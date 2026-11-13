# -*- coding: utf-8 -*-
"""
backtest_niveles_ticks.py - Tus niveles L/H, sobre TUS ticks y TUS parametros

No necesita MetaTrader 5. Lee el parquet de exportar_ticks.py.

    python backtest_niveles_ticks.py ticks_SpotBrent_60d.parquet
    python backtest_niveles_ticks.py ticks_XAUUSD_60d.parquet --paso 0.50 --umbral 1.20

En Colab:
    from backtest_niveles_ticks import correr
    correr("ticks_SpotBrent_60d.parquet")

QUE MIDE
  Cuando el precio toca un nivel que la ventana ANTERIOR marco como
  "soporte con evidencia" (L>=3, H=0), se mueve a favor mas que si
  tocara un nivel cualquiera del mismo rango?

TRES COSAS QUE LO MANTIENEN HONESTO
  1. Los niveles del dia D salen SOLO de las 12h previas. Nunca del dia D.
  2. CONTROL NEGATIVO: niveles al azar del mismo rango, misma cantidad.
  3. CONTROL POSITIVO: se inyectan soportes de verdad EN TUS PROPIOS TICKS
     y se comprueba que el detector los encuentra. Si no los encuentra,
     el test esta ciego y un resultado nulo no significaria nada.

  La version sobre barras M1 fallaba el punto 3: encontraba los soportes
  inyectados el 45% de las veces contra un 40% de azar. Por eso esta.

DOS ESTADISTICOS
  - binario   : soportes "con evidencia" contra niveles al azar
  - continuo  : regresion de la excursion sobre la asimetria L-H de CADA
                nivel tocado. Usa toda la informacion en vez de tirar el
                90% en un umbral, y tiene mucha mas potencia.
"""

import os
import sys
import numpy as np
import pandas as pd

PASO = 0.10
UMBRAL = 0.25
VENTANA_H = 12
MIN_EV = 3
HORIZONTES = [15, 30, 60, 120]     # minutos tras el toque
MULT_LISTON = 3.0


# ---------------------------------------------------------------- datos

def cargar_ticks(ruta):
    if str(ruta).endswith(".parquet"):
        df = pd.read_parquet(ruta, columns=["time_msc", "bid", "ask"])
    else:
        df = pd.read_csv(ruta)
    t = df["time_msc"].to_numpy(dtype=np.int64)
    mid = ((df["bid"].to_numpy(dtype=np.float64) +
            df["ask"].to_numpy(dtype=np.float64)) / 2.0)
    spread = float(np.median(df["ask"].to_numpy(dtype=np.float64) -
                             df["bid"].to_numpy(dtype=np.float64)))
    orden = np.argsort(t, kind="stable")
    return t[orden], mid[orden], spread


# ---------------------------------------------------------------- zigzag

def zigzag(px, umbral):
    """
    Tres estados. Igual que tu script de giros: la version de dos estados
    se bloquea en dirc=0 y devuelve 0 giros.
    Devuelve (indices, precios, tipos) con tipos 1=maximo, -1=minimo.
    """
    n = len(px)
    if n < 3:
        return np.empty(0, np.int64), np.empty(0), np.empty(0, np.int8)
    idx = np.empty(n, np.int64)
    val = np.empty(n)
    tip = np.empty(n, np.int8)
    m = 0
    dirc = 0
    i_ext, p_ext = 0, px[0]
    p0 = px[0]
    for i in range(1, n):
        p = px[i]
        if dirc >= 0 and p > p_ext:
            i_ext, p_ext = i, p
        elif dirc <= 0 and p < p_ext:
            i_ext, p_ext = i, p
        if dirc >= 0 and p <= p_ext - umbral:
            if dirc > 0:
                idx[m], val[m], tip[m] = i_ext, p_ext, 1
                m += 1
            dirc = -1
            i_ext, p_ext = i, p
        elif dirc <= 0 and p >= p_ext + umbral:
            if dirc < 0:
                idx[m], val[m], tip[m] = i_ext, p_ext, -1
                m += 1
            dirc = 1
            i_ext, p_ext = i, p
        elif dirc == 0:
            if p >= p0 + umbral:
                dirc = 1
            elif p <= p0 - umbral:
                dirc = -1
    return idx[:m], val[:m], tip[:m]


def contar(val, tip, paso):
    """dict nivel_entero -> [n_minimos, n_maximos]"""
    c = {}
    k = np.round(val / paso).astype(np.int64)
    for kk, tt in zip(k, tip):
        if kk not in c:
            c[kk] = [0, 0]
        c[kk][1 if tt == 1 else 0] += 1
    return c


# ---------------------------------------------------------------- excursiones

def excursion(t, mid, i0, nivel, arriba, minutos):
    """MFE y MAE desde el tick i0, durante 'minutos'."""
    lim = t[i0] + minutos * 60_000
    j = np.searchsorted(t, lim, side="right")
    if j <= i0 + 2:
        return np.nan, np.nan
    seg = mid[i0:j]
    hi, lo = seg.max(), seg.min()
    if arriba:
        return hi - nivel, nivel - lo
    return nivel - lo, hi - nivel


def primer_toque(k_dia, objetivo, desde=0):
    """Primer tick del dia cuyo nivel redondeado es el objetivo."""
    hits = np.flatnonzero(k_dia[desde:] == objetivo)
    return desde + hits[0] if len(hits) else -1


# ---------------------------------------------------------------- inyeccion

def inyectar_soportes(t, mid, cada, aguante_min=60):
    """
    CONTROL POSITIVO sobre datos reales: elige niveles separados 'cada'
    dolares y, tras cada toque, impide que el mid baje de ese nivel durante
    'aguante_min'. Eso crea la asimetria MFE/MAE que el test busca, en
    sitios conocidos, sin tocar nada mas de la serie.
    """
    m = mid.copy()
    base = np.floor(m.min() / cada) * cada
    niveles = np.arange(base, m.max() + cada, cada)
    n = len(m)
    for lv in niveles:
        i = 0
        while i < n:
            hit = np.flatnonzero(m[i:] <= lv)
            if not len(hit):
                break
            i = i + hit[0]
            lim = t[i] + aguante_min * 60_000
            j = np.searchsorted(t, lim, side="right")
            np.maximum(m[i:j], lv, out=m[i:j])
            i = j
    return m, list(niveles)


# ---------------------------------------------------------------- backtest

def _regresion(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 30:
        return np.nan, np.nan, n
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, n
    b = ((x - xm) * (y - ym)).sum() / sxx
    r = y - (ym + b * (x - xm))
    se = np.sqrt((r ** 2).sum() / (n - 2) / sxx)
    return b, (b / se if se > 0 else np.nan), n


def recorrer(t, mid, paso, umbral, ventana_h, min_ev, horizontes, rng):
    """Un paso por los datos. Devuelve observaciones reales, al azar y continuas."""
    dias_ms = 86_400_000
    dia_id = t // dias_ms
    unicos = np.unique(dia_id)
    reales = {h: [] for h in horizontes}
    azar = {h: [] for h in horizontes}
    cont = {h: [] for h in horizontes}
    n_dias = n_niv = 0
    diag = {"dias": 0, "pocos_ticks": 0, "pocos_giros": 0, "sin_evidencia": 0,
            "giros": [], "niveles": [], "evidencia": []}

    for d in unicos[1:]:
        ini_dia = d * dias_ms
        a = np.searchsorted(t, ini_dia - ventana_h * 3_600_000, "left")
        b = np.searchsorted(t, ini_dia, "left")
        c = np.searchsorted(t, ini_dia + dias_ms, "left")
        diag["dias"] += 1
        if b - a < 2000 or c - b < 2000:
            diag["pocos_ticks"] += 1
            continue

        idx, val, tip = zigzag(mid[a:b], umbral)
        diag["giros"].append(len(val))
        if len(val) < 6:
            diag["pocos_giros"] += 1
            continue
        cuenta = contar(val, tip, paso)
        diag["niveles"].append(len(cuenta))
        sop = [k for k, (L, H) in cuenta.items() if L >= min_ev and H == 0]
        res = [k for k, (L, H) in cuenta.items() if H >= min_ev and L == 0]
        diag["evidencia"].append(len(sop) + len(res))
        if not sop and not res:
            diag["sin_evidencia"] += 1
            continue
        n_dias += 1
        n_niv += len(sop) + len(res)

        t_d, m_d = t[b:c], mid[b:c]
        k_d = np.round(m_d / paso).astype(np.int64)

        lo_k = int(np.round(mid[a:b].min() / paso))
        hi_k = int(np.round(mid[a:b].max() / paso))
        n_fake = len(sop) + len(res)
        fake = rng.integers(lo_k, hi_k + 1, size=n_fake)
        mitad = max(len(sop), 1)

        for h in horizontes:
            for kk in sop:
                i0 = primer_toque(k_d, kk)
                if i0 < 0:
                    continue
                mfe, mae = excursion(t_d, m_d, i0, kk * paso, True, h)
                if np.isfinite(mfe):
                    reales[h].append((mfe, mae))
            for kk in res:
                i0 = primer_toque(k_d, kk)
                if i0 < 0:
                    continue
                mfe, mae = excursion(t_d, m_d, i0, kk * paso, False, h)
                if np.isfinite(mfe):
                    reales[h].append((mfe, mae))
            for j, kk in enumerate(fake):
                i0 = primer_toque(k_d, int(kk))
                if i0 < 0:
                    continue
                mfe, mae = excursion(t_d, m_d, i0, kk * paso, j < mitad, h)
                if np.isfinite(mfe):
                    azar[h].append((mfe, mae))
            # estadistico continuo: TODOS los niveles vistos en la ventana
            for kk, (L, H) in cuenta.items():
                i0 = primer_toque(k_d, kk)
                if i0 < 0:
                    continue
                mfe, mae = excursion(t_d, m_d, i0, kk * paso, True, h)
                if np.isfinite(mfe):
                    cont[h].append((L - H, mfe - mae))
    return reales, azar, cont, n_dias, n_niv, diag


def correr(ruta, paso=PASO, umbral=UMBRAL, ventana_h=VENTANA_H, min_ev=MIN_EV,
           horizontes=None, semilla=0, control_positivo=True, verbose=True):
    horizontes = horizontes or HORIZONTES
    rng = np.random.default_rng(semilla)
    t, mid, spread = cargar_ticks(ruta)

    if verbose:
        print()
        print("=" * 80)
        print("%s" % os.path.basename(ruta))
        print("  %s ticks | spread mediano %.5f | paso %.2f umbral %.2f ventana %dh L>=%d"
              % (format(len(t), ","), spread, paso, umbral, ventana_h, min_ev))
        print("=" * 80)

    reales, azar, cont, n_dias, n_niv, diag = recorrer(
        t, mid, paso, umbral, ventana_h, min_ev, horizontes, rng)

    if verbose:
        g = np.array(diag["giros"]) if diag["giros"] else np.array([0])
        nv = np.array(diag["niveles"]) if diag["niveles"] else np.array([0])
        ev = np.array(diag["evidencia"]) if diag["evidencia"] else np.array([0])
        print()
        print("  DIAGNOSTICO DE LA VENTANA  (esto decide si los parametros sirven)")
        print("    dias evaluados               %d" % diag["dias"])
        print("    descartados por pocos ticks  %d" % diag["pocos_ticks"])
        print("    descartados por pocos giros  %d" % diag["pocos_giros"])
        print("    descartados sin evidencia    %d" % diag["sin_evidencia"])
        print("    giros por ventana   mediana %.0f  (p10 %.0f, p90 %.0f)"
              % (np.median(g), np.percentile(g, 10), np.percentile(g, 90)))
        print("    niveles distintos   mediana %.0f" % np.median(nv))
        print("    con evidencia       mediana %.0f" % np.median(ev))
        print()
        print("  %d dias usables | %d niveles con evidencia en total" % (n_dias, n_niv))
        if n_dias < 15:
            print("    AVISO: muy pocos dias. Prueba --umbral mas bajo o --ventana mas larga.")
        print()
        print("  %5s %7s %9s %9s %9s %8s %9s" % (
            "min", "n", "MFE real", "MFE azar", "ventaja", "spreads", "t continuo"))
        print("  " + "-" * 64)

    salida = {}
    for h in horizontes:
        R = np.array(reales[h]) if reales[h] else np.empty((0, 2))
        A = np.array(azar[h]) if azar[h] else np.empty((0, 2))
        C = np.array(cont[h]) if cont[h] else np.empty((0, 2))
        if len(R) < 20 or len(A) < 20:
            salida[h] = None
            if verbose:
                print("  %5d   muestra insuficiente (real %d, azar %d)" % (h, len(R), len(A)))
            continue
        mfe_r = np.median(R[:, 0])
        mfe_a = np.median(A[:, 0])
        boot = np.array([np.median(rng.choice(R[:, 0], len(R), True)) -
                         np.median(rng.choice(A[:, 0], len(A), True)) for _ in range(300)])
        b_c, t_c, n_c = _regresion(C[:, 0], C[:, 1]) if len(C) >= 30 else (np.nan, np.nan, 0)
        salida[h] = {
            "n": len(R), "mfe_real": mfe_r, "mfe_azar": mfe_a,
            "ventaja": mfe_r - mfe_a,
            "t_binario": (mfe_r - mfe_a) / boot.std(ddof=1) if boot.std(ddof=1) > 0 else np.nan,
            "spreads": mfe_r / spread if spread > 0 else np.nan,
            "t_continuo": t_c, "n_continuo": n_c,
            "mae_real": np.median(R[:, 1]),
        }
        if verbose:
            r = salida[h]
            print("  %5d %7d %9.4f %9.4f %9.4f %8.1fx %9.2f" % (
                h, r["n"], r["mfe_real"], r["mfe_azar"], r["ventaja"],
                r["spreads"], r["t_continuo"]))

    if control_positivo:
        if verbose:
            print()
            print("  CONTROL POSITIVO - soportes reales inyectados en estos mismos ticks")
        mid_iny, nivs = inyectar_soportes(t, mid, cada=max(paso * 10, 1.0))
        rp, ap, cp, _, _, _ = recorrer(t, mid_iny, paso, umbral, ventana_h,
                                    min_ev, [horizontes[1]], np.random.default_rng(semilla + 1))
        h = horizontes[1]
        R = np.array(rp[h]) if rp[h] else np.empty((0, 2))
        A = np.array(ap[h]) if ap[h] else np.empty((0, 2))
        C = np.array(cp[h]) if cp[h] else np.empty((0, 2))
        if len(R) >= 20 and len(A) >= 20:
            vent = np.median(R[:, 0]) - np.median(A[:, 0])
            _, t_c, _ = _regresion(C[:, 0], C[:, 1]) if len(C) >= 30 else (0, np.nan, 0)
            ciego = not (vent > 0 and abs(t_c) > 2)
            if verbose:
                print("    %d niveles inyectados cada %.2f" % (len(nivs), max(paso * 10, 1.0)))
                print("    ventaja detectada %+.4f | t continuo %+.2f" % (vent, t_c))
                print("    -> el test %s" % (
                    "ESTA CIEGO: no detecta soportes que SI existen.\n"
                    "       Un resultado nulo arriba no probaria nada." if ciego
                    else "VE. Los resultados de arriba son interpretables."))
            salida["control_positivo"] = {"ventaja": vent, "t": t_c, "ciego": ciego}
        elif verbose:
            print("    muestra insuficiente para el control")

    if verbose:
        print()
        print("  EL LISTON (fijado antes de mirar)")
        print("    1. ventaja sobre el azar, con |t| > 2 en el continuo")
        print("    2. MFE > %.1f spreads (= %.4f)" % (MULT_LISTON, MULT_LISTON * spread))
        print("    3. MFE > MAE")
        print()
    return salida


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit("Falta el archivo de ticks.")
    kw = {}
    for nombre, clave, tipo in [("--paso", "paso", float), ("--umbral", "umbral", float),
                                ("--ventana", "ventana_h", int), ("--minev", "min_ev", int)]:
        if nombre in sys.argv:
            kw[clave] = tipo(sys.argv[sys.argv.index(nombre) + 1])
    correr(sys.argv[1], **kw)
