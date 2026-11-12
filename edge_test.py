# -*- coding: utf-8 -*-
"""
edge_test.py - Donde esta el edge: direccion vs volatilidad
Magdalena Montoya | Diversifica

Corre:  python edge_test.py
Requiere MetaTrader 5 abierto.

Que hace:
  1. Mide si la DIRECCION de manana es predecible desde hoy
  2. Mide si la VOLATILIDAD de manana es predecible desde hoy
  3. Compara las dos sobre los mismos datos, mismo metodo
  4. Registra el carry (swap) del dia en carry_log.csv para el test futuro

Lo hace para ORO y BRENT.
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd

N_BARRAS = 600   # dias de historial a pedir


# ---------------------------------------------------------------- estadistica
# (sin dependencia de MT5: testeable por separado)

def ols(x, y):
    """Regresion simple. Devuelve (pendiente, R2, t-stat, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 30:
        return np.nan, np.nan, np.nan, n
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, np.nan, n
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    res = y - (a + b * x)
    ss_res = (res ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    se = np.sqrt(ss_res / (n - 2) / sxx) if n > 2 else np.nan
    t = b / se if se and se > 0 else np.nan
    return b, r2, t, n


def acf(x, lags=5):
    """Autocorrelacion hasta 'lags'."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    den = (x * x).sum()
    if den <= 0:
        return [np.nan] * lags
    return [float((x[:-k] * x[k:]).sum() / den) for k in range(1, lags + 1)]


def metricas(close, high, low):
    """
    Nucleo del test. Recibe tres arrays de misma longitud (barras diarias).
    Devuelve un dict con todo lo medido.

    Alineacion (importante, aqui es donde fallan los backtests):
      r[i]  = retorno de la barra i -> i+1
      tr[i] = rango de la barra i+1   (la barra en la que ocurrio r[i])

    OJO: el rango para la REGRESION va en forma relativa, log(high/low).
    Si se usa el rango en dolares (high-low), la regresion acaba midiendo
    el NIVEL del precio -que es un paseo aleatorio muy autocorrelado-
    en vez de la volatilidad, y da un R2 alto que es puro artefacto.
    """
    c = np.asarray(close, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)

    r = np.diff(np.log(c))          # n-1 retornos
    tr = (h - l)[1:]                # n-1 rangos en dolares (para el reporte)
    rel = np.log(np.maximum(h, 1e-9) / np.maximum(l, 1e-9))[1:]   # rango relativo

    atr = pd.Series(rel).rolling(20).mean().values
    lv = np.log(np.maximum(rel, 1e-9))
    la = np.log(np.maximum(atr, 1e-9))

    # DIRECCION: r(t+1) sobre r(t)
    b_d, r2_d, t_d, n_d = ols(r[:-1], r[1:])

    # VOLATILIDAD: log rango(t+1) sobre log ATR20(t)
    b_v, r2_v, t_v, n_v = ols(la[19:-1], lv[20:])

    # persistencia de signo
    s = np.sign(r)
    validos = s[:-1] != 0
    mismo = (s[:-1] == s[1:]) & validos
    pct_signo = 100.0 * mismo.sum() / max(validos.sum(), 1)

    return {
        "n": len(r),
        "dir_r2": r2_d, "dir_t": t_d, "dir_n": n_d,
        "vol_r2": r2_v, "vol_t": t_v, "vol_n": n_v,
        "acf_r": acf(r), "acf_abs": acf(np.abs(r)),
        "pct_signo": pct_signo,
        "rango_med": float(np.median(tr)),
        "rango_p75": float(np.percentile(tr, 75)),
        "rango_p90": float(np.percentile(tr, 90)),
        "vol_anual": float(np.std(r, ddof=1) * np.sqrt(252) * 100),
    }


def reversion(spread):
    """
    Test de valor relativo. Ajusta un Ornstein-Uhlenbeck al spread:
        delta_s(t) = a + b * s(t-1) + ruido
    b negativo = reversion a la media. Vida media = -ln(2)/ln(1+b).
    Devuelve dict con vida media, t-stat de b, sigma diaria del spread.
    """
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 60:
        return None
    ds = np.diff(s)
    b, r2, t, n = ols(s[:-1], ds)
    if not np.isfinite(b) or b >= 0 or (1 + b) <= 0:
        vida = np.inf
    else:
        vida = -np.log(2) / np.log(1 + b)
    return {
        "b": b, "t": t, "n": n, "r2": r2,
        "vida_media": vida,
        "sigma_d": float(np.std(ds, ddof=1)),
        "media": float(np.mean(s)),
        "sd": float(np.std(s, ddof=1)),
        "z_actual": float((s[-1] - np.mean(s)) / np.std(s, ddof=1)),
        "ultimo": float(s[-1]),
    }


def imprimir(nombre, simbolo, m, swap_l=None, swap_s=None, ultima=None):
    print()
    print("=" * 66)
    cab = "%s  [%s]   %d dias" % (nombre, simbolo, m["n"])
    if ultima:
        cab += "   ultima barra %s" % ultima
    print(cab)
    print("=" * 66)
    print()
    print("  QUE ES PREDECIBLE          R2        t      n")
    print("  " + "-" * 44)
    print("  DIRECCION de manana    %8.4f  %7.2f  %5d" % (m["dir_r2"], m["dir_t"], m["dir_n"]))
    print("  VOLATILIDAD de manana  %8.4f  %7.2f  %5d" % (m["vol_r2"], m["vol_t"], m["vol_n"]))
    print()
    # veredicto en castellano llano (|t| > 2 = senal; si no, ruido)
    dir_vale = np.isfinite(m["dir_t"]) and abs(m["dir_t"]) > 2 and m["dir_r2"] > 0.01
    vol_vale = np.isfinite(m["vol_t"]) and abs(m["vol_t"]) > 2 and m["vol_r2"] > 0.05
    print("  VEREDICTO")
    print("    direccion:   %s" % (
        "HAY SENAL -- revisa que no sea artefacto" if dir_vale
        else "ruido. No se puede predecir. No construyas senales direccionales diarias."))
    print("    volatilidad: %s" % (
        "PREDECIBLE. Aqui si hay donde trabajar." if vol_vale
        else "sin senal en estos datos."))
    if dir_vale and vol_vale and m["dir_r2"] > 0:
        print("    la volatilidad es %.0fx mas predecible que la direccion"
              % (m["vol_r2"] / m["dir_r2"]))
    elif vol_vale and not dir_vale:
        print("    el edge esta en el TAMANO de los movimientos, no en su SIGNO")
    print()
    print("  AUTOCORRELACION      lag1    lag2    lag3    lag4    lag5")
    print("  retornos         " + " ".join("%7.3f" % v for v in m["acf_r"]))
    print("  |retornos|       " + " ".join("%7.3f" % v for v in m["acf_abs"]))
    print()
    print("  mismo signo que ayer: %5.1f%%   (azar = 50%%)" % m["pct_signo"])
    print("  rango diario: mediana %.2f | p75 %.2f | p90 %.2f" % (
        m["rango_med"], m["rango_p75"], m["rango_p90"]))
    print("  volatilidad anualizada: %.1f%%" % m["vol_anual"])
    if swap_l is not None:
        print("  swap: largo %s | corto %s" % (swap_l, swap_s))


# ---------------------------------------------------------------- MetaTrader 5

def conectar(mt5):
    if mt5.initialize():
        return True
    rutas = glob.glob(r"C:\Program Files\*\terminal64.exe")
    rutas += glob.glob(os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe"))
    for ruta in rutas:
        if mt5.initialize(path=ruta):
            print("conectado via %s" % ruta)
            return True
    return False


def buscar_simbolo(mt5, claves):
    """Primera clave que aparezca en el nombre de algun simbolo."""
    todos = mt5.symbols_get()
    if not todos:
        return None
    nombres = [s.name for s in todos]
    for clave in claves:
        for n in nombres:
            if clave in n.upper():
                return n
    return None


def barras_diarias(mt5, simbolo, n=N_BARRAS):
    raw = mt5.copy_rates_from_pos(simbolo, mt5.TIMEFRAME_D1, 0, n)
    if raw is None or len(raw) < 80:
        return None
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def registrar_carry(mt5, pares, ruta="carry_log.csv"):
    """
    El carry no se puede backtestear: MT5 no guarda historial de swap.
    Asi que empieza el registro hoy. En 60-90 dias tendras serie para
    testear si la pendiente de la curva predice retornos futuros.
    """
    hoy = datetime.date.today().isoformat()
    filas = []
    for nombre, sim in pares:
        if not sim:
            continue
        info = mt5.symbol_info(sim)
        tick = mt5.symbol_info_tick(sim)
        if info and tick:
            filas.append({
                "fecha": hoy, "instrumento": nombre, "simbolo": sim,
                "bid": tick.bid, "ask": tick.ask,
                "spread": round(tick.ask - tick.bid, 5),
                "swap_long": info.swap_long, "swap_short": info.swap_short,
            })
    if not filas:
        return
    d = pd.DataFrame(filas)
    nuevo = not os.path.exists(ruta)
    d.to_csv(ruta, mode="a", header=nuevo, index=False, encoding="utf-8")
    print()
    print("-" * 66)
    print("carry del dia anadido a %s  (%d filas)" % (os.path.abspath(ruta), len(filas)))
    print("corre esto una vez al dia; en 60-90 dias se puede testear.")


def main():
    import MetaTrader5 as mt5

    if not conectar(mt5):
        raise SystemExit("No conecta a MT5. Abre MetaTrader 5 y repite.")

    oro = buscar_simbolo(mt5, ["XAUUSD", "GOLD"])
    brent = buscar_simbolo(mt5, ["SPOTBRENT", "UKOIL", "BRENT"])
    wti = buscar_simbolo(mt5, ["SPOTCRUDE", "USOIL", "WTI", "CRUDE"])

    print("simbolos encontrados -> oro: %s | brent: %s | wti: %s" % (oro, brent, wti))

    guardados = {}
    for nombre, sim in [("ORO", oro), ("BRENT", brent), ("WTI", wti)]:
        if not sim:
            if nombre != "WTI":
                print("\n%s: no encontrado en tu broker" % nombre)
            continue
        df = barras_diarias(mt5, sim)
        if df is None:
            print("\n%s [%s]: historial insuficiente" % (nombre, sim))
            continue
        guardados[nombre] = df
        if nombre == "WTI":
            continue   # el WTI solo se usa para el spread
        m = metricas(df["close"].values, df["high"].values, df["low"].values)
        info = mt5.symbol_info(sim)
        imprimir(nombre, sim, m,
                 swap_l=info.swap_long if info else None,
                 swap_s=info.swap_short if info else None,
                 ultima=str(df["time"].iloc[-1].date()))

    # ---- valor relativo: spread Brent - WTI
    if "BRENT" in guardados and "WTI" in guardados:
        a = guardados["BRENT"][["time", "close"]].rename(columns={"close": "brent"})
        b = guardados["WTI"][["time", "close"]].rename(columns={"close": "wti"})
        j = pd.merge(a, b, on="time", how="inner")
        rv = reversion((j["brent"] - j["wti"]).values) if len(j) >= 60 else None
        print()
        print("=" * 66)
        print("VALOR RELATIVO  spread BRENT - WTI   (%d dias comunes)" % len(j))
        print("=" * 66)
        if rv is None:
            print("  historial comun insuficiente")
        else:
            vm = rv["vida_media"]
            print()
            print("  spread actual      $%.2f   (media $%.2f, sd $%.2f)" % (
                rv["ultimo"], rv["media"], rv["sd"]))
            print("  z-score actual      %+.2f" % rv["z_actual"])
            print("  vida media         %s" % (
                "sin reversion detectable" if not np.isfinite(vm) or vm > 200
                else "%.1f dias" % vm))
            print("  t-stat reversion   %6.2f   (necesitas < -2)" % rv["t"])
            print("  sigma diaria       $%.3f" % rv["sigma_d"])
            sb = mt5.symbol_info(brent)
            sw = mt5.symbol_info(wti)
            if sb and sw:
                tb, tw = mt5.symbol_info_tick(brent), mt5.symbol_info_tick(wti)
                if tb and tw:
                    coste = (tb.ask - tb.bid) + (tw.ask - tw.bid)
                    print("  coste ida y vuelta $%.3f  (spread de las DOS patas)" % coste)
                    print("  sigma / coste       %.1fx   (por debajo de 3x no hay negocio)"
                          % (rv["sigma_d"] / coste if coste > 0 else 0))

    registrar_carry(mt5, [("ORO", oro), ("BRENT", brent), ("WTI", wti)])
    mt5.shutdown()


if __name__ == "__main__":
    main()
