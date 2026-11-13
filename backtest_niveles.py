# -*- coding: utf-8 -*-
"""
backtest_niveles.py - Tienen edge tus niveles L/H?

=====================================================================
ESTADO: NO USAR TODAVIA PARA DECIDIR NADA.
  El control positivo NO pasa. Al inyectar soportes reales en una serie
  sintetica, el detector los encuentra el 45% de las veces cuando el
  azar da 40%. Es decir: el test no distingue un soporte real del ruido.

  Lo que si pasa: no mira al futuro (caso 3) y no inventa senales donde
  no las hay (caso 1). La fontaneria esta bien; la SENSIBILIDAD no.

  Un test ciego que devuelve "no hay edge" no prueba nada. Por eso esto
  se entrega como trabajo en curso y no como veredicto.

  Causa mas probable: tu senal es NATIVA DE TICKS (umbral 0.25, ventana
  12h, ~134 giros) y esto corre sobre M1, donde el mismo umbral da 2
  giros. Puede que M1 este midiendo otra cosa. Siguiente paso: repetirlo
  sobre ticks con TUS parametros.
=====================================================================

NO necesita MetaTrader 5. Lee un archivo de barras M1 y corre en
cualquier sitio: VPS, Google Colab, o donde sea.

    python backtest_niveles.py datos_SpotBrent_M1.parquet

En Colab:
    from backtest_niveles import correr_backtest, cargar
    df = cargar("datos_SpotBrent_M1.parquet")
    correr_backtest(df, spread=0.034)

HIPOTESIS
  Un nivel con L>=3 y H=0 (tres minimos, ningun maximo) aguanta mas
  que un nivel cualquiera del mismo rango.

LA REGLA QUE LO HACE HONESTO
  Los niveles de cada dia se construyen SOLO con la ventana anterior.
  Nunca con datos del dia que se opera. Si se mezclan, el backtest
  "funciona" siempre y no significa nada.

POR QUE ESTOS PARAMETROS Y NO LOS DE TU SCRIPT DE GIROS
  Tu script corre sobre TICKS con umbral 0.25 y saca ~134 giros en 12h.
  Sobre barras M1 el mismo umbral da 2 giros: la diferencia es que gran
  parte de ese swing tick a tick es el bid/ask rebotando, no precio.
  Es el mismo artefacto de spread que te falseaba el variance ratio.
  Aqui se mide la IDEA (un nivel asimetrico aguanta mas) sobre precio
  limpio, no la implementacion tick a tick. Si quieres probar la tuya
  exacta, pasa --umbral 0.25 --ventana 12 --minev 3 sobre datos de tick.

EL LISTON, FIJADO ANTES DE MIRAR
  El rebote mediano tiene que superar 3 SPREADS. Un rebote de 2 centimos
  sobre un spread de 3.4 no es edge aunque el estadistico salga bonito.
"""

import sys
import numpy as np
import pandas as pd

PASO = 0.10          # tamano del nivel
UMBRAL = 0.10        # giro minimo del zigzag (calibrado para M1, ver nota abajo)
VENTANA_H = 48       # horas de la ventana previa
MIN_EV = 2           # minimos (o maximos) para considerar "con evidencia"
HORIZONTES = [30, 60, 120, 240]   # minutos tras el toque
MULT_LISTON = 3.0    # el rebote debe superar esto x spread


# ---------------------------------------------------------------- estructura

def zigzag(px, umbral):
    """
    Maquina de tres estados (sube / baja / indecisa).
    La version de dos estados se bloquea y devuelve 0 giros.
    Devuelve [(indice, precio, 'L'|'H'), ...]
    """
    px = np.asarray(px, dtype=float)
    if len(px) < 3:
        return []
    swings = []
    dirc = 0
    i_ext = 0
    p_ext = px[0]
    for i in range(1, len(px)):
        p = px[i]
        if dirc >= 0 and p > p_ext:
            i_ext, p_ext = i, p
        elif dirc <= 0 and p < p_ext:
            i_ext, p_ext = i, p
        if dirc >= 0 and p <= p_ext - umbral:
            if dirc > 0:
                swings.append((i_ext, p_ext, "H"))
            dirc = -1
            i_ext, p_ext = i, p
        elif dirc <= 0 and p >= p_ext + umbral:
            if dirc < 0:
                swings.append((i_ext, p_ext, "L"))
            dirc = 1
            i_ext, p_ext = i, p
        elif dirc == 0:
            if p >= px[0] + umbral:
                dirc = 1
            elif p <= px[0] - umbral:
                dirc = -1
    return swings


def contar_niveles(swings, paso=PASO):
    """Cuantos minimos (L) y maximos (H) cayeron en cada nivel."""
    conteo = {}
    for _, precio, tipo in swings:
        k = int(round(precio / paso))
        if k not in conteo:
            conteo[k] = [0, 0]
        conteo[k][0 if tipo == "L" else 1] += 1
    return conteo


def con_evidencia(conteo, min_ev=MIN_EV, paso=PASO):
    """Niveles asimetricos: soporte = L>=min y H==0; resistencia al reves."""
    sop, res = [], []
    for k, (L, H) in conteo.items():
        if L >= min_ev and H == 0:
            sop.append(k * paso)
        elif H >= min_ev and L == 0:
            res.append(k * paso)
    return sop, res


# ---------------------------------------------------------------- excursiones

def primer_toque(low, high, nivel, desde=0):
    """Primer minuto en que el rango de la barra contiene el nivel."""
    n = len(low)
    for i in range(desde, n):
        if low[i] <= nivel <= high[i]:
            return i
    return -1


def excursion(low, high, i0, nivel, favorable_arriba, horizonte):
    """
    Desde el toque: cuanto llego a moverse a favor (MFE) y en contra (MAE).
    Para un soporte lo favorable es ARRIBA.
    """
    j = min(i0 + horizonte, len(low))
    if j <= i0 + 1:
        return np.nan, np.nan
    hi = np.max(high[i0:j])
    lo = np.min(low[i0:j])
    if favorable_arriba:
        return hi - nivel, nivel - lo
    return nivel - lo, hi - nivel


def evaluar(df_dia, niveles, favorable_arriba, horizonte):
    """Un registro por nivel que llego a tocarse ese dia."""
    low = df_dia["low"].to_numpy()
    high = df_dia["high"].to_numpy()
    out = []
    for nv in niveles:
        i0 = primer_toque(low, high, nv)
        if i0 < 0:
            continue
        mfe, mae = excursion(low, high, i0, nv, favorable_arriba, horizonte)
        if np.isfinite(mfe) and np.isfinite(mae):
            out.append((nv, mfe, mae))
    return out


# ---------------------------------------------------------------- backtest

def correr_backtest(df, spread=0.034, paso=PASO, umbral=UMBRAL,
                    ventana_h=VENTANA_H, min_ev=MIN_EV,
                    horizontes=None, semilla=0, verbose=True):
    """
    df: columnas time, open, high, low, close en M1.
    Devuelve un dict con los resultados por horizonte.
    """
    horizontes = horizontes or HORIZONTES
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"])
    d = d.sort_values("time").reset_index(drop=True)
    d["dia"] = d["time"].dt.date
    dias = sorted(d["dia"].unique())
    rng = np.random.default_rng(semilla)

    reales = {h: [] for h in horizontes}
    azar = {h: [] for h in horizontes}
    n_dias_usados = 0
    n_niveles = 0

    for idx in range(1, len(dias)):
        dia = dias[idx]
        fin_ventana = pd.Timestamp(dia)
        ini_ventana = fin_ventana - pd.Timedelta(hours=ventana_h)
        prev = d[(d["time"] >= ini_ventana) & (d["time"] < fin_ventana)]
        hoy = d[d["dia"] == dia]
        if len(prev) < 120 or len(hoy) < 60:
            continue

        # --- niveles SOLO con la ventana previa
        sw = zigzag(prev["close"].to_numpy(), umbral)
        if len(sw) < 4:
            continue
        sop, res = con_evidencia(contar_niveles(sw, paso), min_ev, paso)
        if not sop and not res:
            continue
        n_dias_usados += 1
        n_niveles += len(sop) + len(res)

        # --- control: niveles al azar del mismo rango y misma cantidad
        lo_p, hi_p = prev["low"].min(), prev["high"].max()
        n_fake = len(sop) + len(res)
        fake = rng.uniform(lo_p, hi_p, size=n_fake)
        fake = np.round(fake / paso) * paso
        mitad = max(len(sop), 1)

        for h in horizontes:
            for _, mfe, mae in evaluar(hoy, sop, True, h):
                reales[h].append((mfe, mae))
            for _, mfe, mae in evaluar(hoy, res, False, h):
                reales[h].append((mfe, mae))
            for _, mfe, mae in evaluar(hoy, fake[:mitad], True, h):
                azar[h].append((mfe, mae))
            for _, mfe, mae in evaluar(hoy, fake[mitad:], False, h):
                azar[h].append((mfe, mae))

    res_final = {}
    for h in horizontes:
        R = np.array(reales[h]) if reales[h] else np.empty((0, 2))
        A = np.array(azar[h]) if azar[h] else np.empty((0, 2))
        if len(R) < 30 or len(A) < 30:
            res_final[h] = None
            continue
        mfe_r, mae_r = np.median(R[:, 0]), np.median(R[:, 1])
        mfe_a, mae_a = np.median(A[:, 0]), np.median(A[:, 1])
        # diferencia real - azar, con error estandar por bootstrap
        boot = []
        for _ in range(400):
            br = rng.choice(R[:, 0], size=len(R), replace=True)
            ba = rng.choice(A[:, 0], size=len(A), replace=True)
            boot.append(np.median(br) - np.median(ba))
        boot = np.array(boot)
        res_final[h] = {
            "n_real": len(R), "n_azar": len(A),
            "mfe_real": mfe_r, "mae_real": mae_r,
            "mfe_azar": mfe_a, "mae_azar": mae_a,
            "ventaja": mfe_r - mfe_a,
            "ventaja_sd": boot.std(ddof=1),
            "t": (mfe_r - mfe_a) / boot.std(ddof=1) if boot.std(ddof=1) > 0 else np.nan,
            "mfe_en_spreads": mfe_r / spread if spread > 0 else np.nan,
            "ratio": mfe_r / mae_r if mae_r > 0 else np.nan,
        }

    if verbose:
        imprimir(res_final, spread, n_dias_usados, n_niveles)
    return res_final


def imprimir(res, spread, n_dias, n_niveles):
    print()
    print("=" * 76)
    print("NIVELES L/H  -  %d dias usados, %d niveles evaluados, spread %.3f"
          % (n_dias, n_niveles, spread))
    print("=" * 76)
    print()
    print("  %4s %7s %8s %8s %9s %8s %9s %7s" % (
        "min", "n", "MFE", "MFE azar", "ventaja", "t", "en spreads", "MFE/MAE"))
    print("  " + "-" * 68)
    for h, r in sorted(res.items()):
        if r is None:
            print("  %4d   muestra insuficiente" % h)
            continue
        print("  %4d %7d %8.3f %8.3f %9.3f %8.2f %9.1fx %7.2f" % (
            h, r["n_real"], r["mfe_real"], r["mfe_azar"], r["ventaja"],
            r["t"], r["mfe_en_spreads"], r["ratio"]))

    print()
    print("  EL LISTON (fijado antes de mirar los datos)")
    print("    1. ventaja sobre el azar con |t| > 2")
    print("    2. MFE > %.1f spreads (= %.3f)" % (MULT_LISTON, MULT_LISTON * spread))
    print("    3. MFE/MAE > 1")
    print()
    algun = False
    for h, r in sorted(res.items()):
        if r is None:
            continue
        c1 = abs(r["t"]) > 2
        c2 = r["mfe_en_spreads"] > MULT_LISTON
        c3 = r["ratio"] > 1
        marca = lambda b: "SI " if b else "NO "
        print("    %4d min:  azar %s| liston %s| MFE>MAE %s %s" % (
            h, marca(c1), marca(c2), marca(c3),
            "  <-- PASA LAS TRES" if (c1 and c2 and c3) else ""))
        algun = algun or (c1 and c2 and c3)
    print()
    if not algun:
        print("  VEREDICTO: ningun horizonte pasa las tres. La senal de niveles")
        print("  no sostiene un bot intradia. Descartada, como el calendario.")
    else:
        print("  VEREDICTO: hay al menos un horizonte que pasa. Siguiente paso")
        print("  OBLIGATORIO: walk-forward. Esto sigue siendo dentro de muestra.")


def cargar(ruta):
    if str(ruta).endswith(".parquet"):
        return pd.read_parquet(ruta)
    return pd.read_csv(ruta)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit("Falta el archivo: python backtest_niveles.py datos_M1.parquet")
    sp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.034
    correr_backtest(cargar(sys.argv[1]), spread=sp)
