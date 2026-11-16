# -*- coding: utf-8 -*-
"""
bt_core.py - Motor de backtest sobre ticks. SIN MetaTrader5.
Magdalena Montoya | Diversifica

Corre en Google Colab, en el VPS o en el Mac. MT5 es solo Windows, asi que
el motor no lo toca: lee los parquet que saca exportar_ticks.py y ya.

LA REGLA QUE HACE QUE UN BACKTEST NO MIENTA
  La senal de la barra i se ejecuta en la apertura de la barra i+1.
  Nunca en el cierre de i. Si se ejecuta en el cierre de la misma barra
  que genero la senal, se esta operando con un precio que en la vida real
  todavia no se conocia, y cualquier estrategia parece rentable.

EL COSTE
  El spread sale del propio archivo de ticks, no de una suposicion.
  Comprar paga el ask, vender cobra el bid. Entrar y salir cuesta un
  spread completo, siempre. Un backtest que no cobre esto es publicidad.

SI EL STOP Y EL OBJETIVO CAEN EN LA MISMA BARRA
  Se asume que salto el STOP. No se sabe que ocurrio primero dentro de la
  barra, y suponer lo bueno es exactamente como se fabrica una curva de
  equity falsa.
"""

import os
import numpy as np
import pandas as pd


# --------------------------------------------------------------------- datos

def cargar_ticks(ruta):
    """Lee el parquet (o csv.gz) que produce exportar_ticks.py."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(ruta)
    if ruta.endswith(".parquet"):
        d = pd.read_parquet(ruta)
    else:
        d = pd.read_csv(ruta)
    faltan = {"time_msc", "bid", "ask"} - set(d.columns)
    if faltan:
        raise ValueError("al archivo le faltan columnas: %s" % faltan)
    d = d[(d["bid"] > 0) & (d["ask"] > 0) & (d["ask"] >= d["bid"])]
    d["time"] = pd.to_datetime(d["time_msc"], unit="ms")
    return d.sort_values("time").reset_index(drop=True)


def hacer_barras(ticks, minutos=5):
    """
    Ticks -> barras OHLC del punto medio, mas el spread mediano de cada barra.

    El medio (bid+ask)/2 es lo correcto para medir el movimiento: usar solo
    el bid mete el spread dentro de la serie de precios y luego se cobra
    otra vez al ejecutar, asi que se pagaria dos veces.
    """
    t = ticks.copy()
    t["mid"] = (t["bid"] + t["ask"]) / 2.0
    t["spr"] = t["ask"] - t["bid"]
    g = t.set_index("time").resample("%dmin" % minutos)
    b = pd.DataFrame({
        "open": g["mid"].first(),
        "high": g["mid"].max(),
        "low": g["mid"].min(),
        "close": g["mid"].last(),
        "spread": g["spr"].median(),
        "ticks": g["mid"].count(),
    })
    b = b[b["ticks"] > 0].copy()
    # el fin de semana deja huecos: eso esta bien, no se rellena.
    # rellenar inventa barras con rango cero y ensucia toda medida de volatilidad.
    b = b.reset_index().rename(columns={"time": "time"})
    b["rel"] = np.log(b["high"].clip(lower=1e-9) / b["low"].clip(lower=1e-9))
    return b


def atr(barras, n=20):
    """ATR en unidades de precio, DESPLAZADO: en la barra i vale el de i-1."""
    tr = (barras["high"] - barras["low"]).to_numpy(dtype=float)
    return pd.Series(tr).rolling(n).mean().shift(1).to_numpy()


# -------------------------------------------------------------------- motor

def correr(barras, senales, stops, objetivos, max_barras=48, etiqueta=""):
    """
    Ejecuta las senales. Todo en unidades de precio del instrumento.

    senales[i]   +1 comprar, -1 vender, 0 nada  (decidido al cierre de i)
    stops[i]     distancia del stop en precio   (NaN = sin stop)
    objetivos[i] distancia del objetivo         (NaN = sin objetivo)
    max_barras   cierre forzoso: ninguna operacion intradia se queda abierta
                 para siempre; sin esto el backtest sobrevive gracias a una
                 posicion que aguanto tres semanas, que no es intradia.

    Devuelve DataFrame de operaciones. Cada fila ya lleva el coste pagado.
    """
    o = barras["open"].to_numpy(dtype=float)
    h = barras["high"].to_numpy(dtype=float)
    l = barras["low"].to_numpy(dtype=float)
    sp = barras["spread"].to_numpy(dtype=float)
    tiempo = barras["time"].to_numpy()
    n = len(o)

    ops = []
    i = 0
    while i < n - 2:
        s = senales[i]
        if not np.isfinite(s) or s == 0:
            i += 1
            continue

        e = i + 1                       # <-- se ejecuta en la barra SIGUIENTE
        medio_spread = sp[e] / 2.0 if np.isfinite(sp[e]) else 0.0
        if s > 0:
            entrada = o[e] + medio_spread          # compra al ask
        else:
            entrada = o[e] - medio_spread          # venta al bid

        dist_stop = stops[i] if np.isfinite(stops[i]) else np.nan
        dist_obj = objetivos[i] if np.isfinite(objetivos[i]) else np.nan
        niv_stop = entrada - s * dist_stop if np.isfinite(dist_stop) else np.nan
        niv_obj = entrada + s * dist_obj if np.isfinite(dist_obj) else np.nan

        salida = np.nan
        motivo = "tiempo"
        j = e
        mfe = 0.0
        mae = 0.0
        while j < n and (j - e) < max_barras:
            favor = (h[j] - entrada) if s > 0 else (entrada - l[j])
            contra = (entrada - l[j]) if s > 0 else (h[j] - entrada)
            mfe = max(mfe, favor)
            mae = max(mae, contra)

            toca_stop = (np.isfinite(niv_stop) and
                         (l[j] <= niv_stop if s > 0 else h[j] >= niv_stop))
            toca_obj = (np.isfinite(niv_obj) and
                        (h[j] >= niv_obj if s > 0 else l[j] <= niv_obj))

            # el stop primero SIEMPRE que los dos caben en la barra.
            # no se sabe el orden dentro de la barra; suponer lo bueno
            # es como se fabrican las curvas de equity falsas.
            if toca_stop:
                salida = niv_stop
                motivo = "stop"
                break
            if toca_obj:
                salida = niv_obj
                motivo = "objetivo"
                break
            j += 1

        if not np.isfinite(salida):
            j = min(j, n - 1)
            salida = o[j]
            motivo = "tiempo"

        medio_salida = sp[min(j, n - 1)] / 2.0 if np.isfinite(sp[min(j, n - 1)]) else 0.0
        if s > 0:
            salida_real = salida - medio_salida     # vende al bid
        else:
            salida_real = salida + medio_salida     # compra al ask

        pnl = s * (salida_real - entrada)
        ops.append({
            "etiqueta": etiqueta,
            "i_senal": i, "i_entrada": e, "i_salida": j,
            "t_entrada": tiempo[e], "t_salida": tiempo[min(j, n - 1)],
            "lado": int(s), "entrada": entrada, "salida": salida_real,
            "pnl": pnl, "pnl_pct": 100.0 * pnl / entrada if entrada else np.nan,
            "coste": (medio_spread + medio_salida),
            "barras": j - e, "motivo": motivo,
            "mfe": mfe, "mae": mae,
        })
        i = j + 1        # una posicion a la vez, como opera ella

    return pd.DataFrame(ops)


# ---------------------------------------------------------------- resultados

def estadisticas(ops, precio_ref=None):
    """
    Lo unico que decide si una estrategia existe: la expectativa por
    operacion y su t-stat. El resto es decoracion.

    t = media / (desviacion / raiz(n)).  Por debajo de 2 no hay nada que
    defender: con 200 operaciones, una estrategia sin ninguna ventaja
    produce |t| > 2 una vez de cada veinte solo por azar.
    """
    if ops is None or len(ops) == 0:
        return {"n": 0}
    p = ops["pnl"].to_numpy(dtype=float)
    pct = ops["pnl_pct"].to_numpy(dtype=float)
    coste = ops["coste"].to_numpy(dtype=float)
    n = len(p)
    sd = p.std(ddof=1) if n > 1 else np.nan
    t = p.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    eq = np.cumsum(p)
    pico = np.maximum.accumulate(eq)
    gana = p > 0
    return {
        "n": n,
        "expectativa": float(p.mean()),
        "expectativa_pct": float(np.nanmean(pct)),
        "t": float(t) if np.isfinite(t) else np.nan,
        "sd": float(sd) if np.isfinite(sd) else np.nan,
        "aciertos": 100.0 * gana.sum() / n,
        "media_gana": float(p[gana].mean()) if gana.any() else 0.0,
        "media_pierde": float(p[~gana].mean()) if (~gana).any() else 0.0,
        "total": float(p.sum()),
        "max_caida": float((pico - eq).max()) if n else 0.0,
        "coste_medio": float(np.nanmean(coste)),
        # cuantos spreads gana cada operacion. Por debajo de 1 la
        # estrategia trabaja para el broker.
        "en_spreads": float(p.mean() / np.nanmean(coste)) if np.nanmean(coste) > 0 else np.nan,
        "barras_media": float(ops["barras"].mean()),
        "pct_stop": 100.0 * (ops["motivo"] == "stop").sum() / n,
        "pct_objetivo": 100.0 * (ops["motivo"] == "objetivo").sum() / n,
        "pct_tiempo": 100.0 * (ops["motivo"] == "tiempo").sum() / n,
        "mfe_medio": float(ops["mfe"].mean()),
        "mae_medio": float(ops["mae"].mean()),
    }


def tabla(resultados, titulo=""):
    """resultados: lista de (nombre, dict de estadisticas)."""
    print()
    print("=" * 100)
    print("RESULTADOS%s" % ((" - " + titulo) if titulo else ""))
    print("=" * 100)
    print()
    print("  %-22s %6s %11s %7s %8s %8s %9s %9s" % (
        "estrategia", "n", "expect.", "t", "aciertos", "spreads", "total", "max caida"))
    print("  " + "-" * 96)
    for nombre, e in resultados:
        if e.get("n", 0) == 0:
            print("  %-22s %6d   sin operaciones" % (nombre, 0))
            continue
        if e["n"] < 5:
            # con 1 o 2 operaciones no hay estadistica que hacer: un t-stat
            # ahi es un numero inventado. Se muestra el total y nada mas.
            print("  %-22s %6d %11.5f %7s %7s %8s %9.2f %9.2f   <- referencia, "
                  "no comparable" % (
                      nombre, e["n"], e["expectativa"], "-", "-", "-",
                      e["total"], e["max_caida"]))
            continue
        print("  %-22s %6d %11.5f %7.2f %7.1f%% %8.2f %9.2f %9.2f" % (
            nombre, e["n"], e["expectativa"], e["t"], e["aciertos"],
            e["en_spreads"], e["total"], e["max_caida"]))
    print()
    print("  expect. = beneficio medio por operacion, en precio, YA con el spread cobrado")
    print("  spreads = cuantos spreads gana cada operacion. Por debajo de 1, el negocio es del broker")
    print("  t       = por debajo de 2 no hay evidencia de nada")
    print()
    aprueban = [n for n, e in resultados
                if e.get("n", 0) > 30 and np.isfinite(e.get("t", np.nan))
                and e["t"] > 2 and e["expectativa"] > 0]
    print("  pasan el filtro (t > 2, expectativa > 0, n > 30): %s" % (
        ", ".join(aprueban) if aprueban else "NINGUNA"))
    if not aprueban:
        print("  Eso es el resultado habitual y es informacion, no un fracaso:")
        print("  significa que ninguna de estas ideas sobrevive a sus propios costes.")
    return aprueban
