# -*- coding: utf-8 -*-
"""
backtest_calendario.py - Idea 2: la volatilidad de los dias con evento

Corre:  python backtest_calendario.py
Requiere MetaTrader 5 abierto.

HIPOTESIS
  Los dias con evento programado (EIA, nominas, CPI, FOMC) producen rangos
  sistematicamente mayores, y ese exceso es estable.

POR QUE ESTA ES LA QUE SE PUEDE TESTEAR HOY
  Mediste que la volatilidad es predecible (R2 ~0.30-0.35) y la direccion no.
  Los eventos son el mayor generador de volatilidad Y estan en un calendario
  publico. No hace falta predecir nada: la fecha ya la sabes.

LA TRAMPA QUE EVITA ESTE SCRIPT
  Comparar sin mas "dias de evento" contra "dias normales" NO sirve: los
  eventos pueden caer en periodos ya volatiles y estarias midiendo el
  regimen, no el evento. La especificacion correcta es

      log(rango_t) = a + b*log(ATR20_{t-1}) + c*evento_t

  El coeficiente c es el efecto del evento NETO del regimen que ya habia.
  El ATR va retrasado un dia: si incluyera el dia t seria lookahead.

CONTROLES
  - placebo: dias falsos con la misma frecuencia -> c debe salir ~0
  - direccion: el mismo test sobre el retorno -> no debe haber nada
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd

N_BARRAS = 600
ARCHIVO_EXTRA = "calendario_extra.csv"   # fechas que anadas a mano


# ------------------------------------------------------------------ estadistica

def ols_multi(X, y):
    """
    Regresion multiple. X ya incluye la columna de unos.
    Devuelve (beta, se, t, R2, n).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    n, k = X.shape
    if n <= k + 5:
        return None
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = (resid ** 2).sum() / (n - k)
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return beta, se, t, r2, n


def preparar(df):
    """
    De barras diarias a las series del test.
    Todo relativo (log high/low), nunca en dolares: el rango en dolares
    escala con el nivel del precio y la regresion acabaria midiendo el nivel.
    ATR retrasado un dia para no mirar al futuro.
    """
    d = df.copy()
    d["rel"] = np.log(d["high"] / d["low"])
    d["lrango"] = np.log(np.maximum(d["rel"], 1e-9))
    d["atr20"] = d["rel"].rolling(20).mean().shift(1)     # <- shift: sin lookahead
    d["latr"] = np.log(np.maximum(d["atr20"], 1e-9))
    d["ret"] = np.log(d["close"]).diff()
    d["dow"] = d["time"].dt.dayofweek                      # 0=lunes
    return d


def probar(d, mascara, etiqueta):
    """Efecto del evento sobre el rango, neto del regimen previo."""
    sub = d.dropna(subset=["lrango", "latr"]).copy()
    ev = mascara.reindex(sub.index).fillna(False).to_numpy().astype(float)
    if ev.sum() < 15 or (len(ev) - ev.sum()) < 30:
        return None
    X = np.column_stack([np.ones(len(sub)), sub["latr"].to_numpy(), ev])
    r = ols_multi(X, sub["lrango"].to_numpy())
    if r is None:
        return None
    beta, se, t, r2, n = r
    # el mismo test sobre la direccion (control negativo)
    Xd = np.column_stack([np.ones(len(sub)), ev])
    rd = ols_multi(Xd, sub["ret"].fillna(0).to_numpy())
    t_dir = rd[2][1] if rd else np.nan

    m_ev = sub.loc[ev == 1, "rel"]
    m_no = sub.loc[ev == 0, "rel"]
    return {
        "etiqueta": etiqueta, "n_ev": int(ev.sum()), "n": n,
        "coef": beta[2], "t": t[2],
        "exceso_pct": 100.0 * (np.exp(beta[2]) - 1.0),
        "crudo_pct": 100.0 * (m_ev.median() / m_no.median() - 1.0),
        "t_dir": t_dir,
    }


# ------------------------------------------------------------------ calendario

def primer_viernes(anio, mes):
    d = datetime.date(anio, mes, 1)
    return d + datetime.timedelta(days=(4 - d.weekday()) % 7)


def construir_calendario(fechas):
    """
    Eventos por REGLA (auditables, sin depender de mi memoria):
      EIA   - inventarios semanales, miercoles
      NFP   - nominas, primer viernes de mes
    Las fechas que no siguen regla (FOMC, CPI) van en calendario_extra.csv,
    que puedes rellenar a mano. Mejor un calendario corto y correcto que
    uno largo e inventado.
    """
    idx = pd.DatetimeIndex(fechas)
    cal = {}
    cal["EIA (miercoles)"] = pd.Series(idx.dayofweek == 2, index=range(len(idx)))
    vs = {primer_viernes(d.year, d.month) for d in idx}
    cal["NFP (1er viernes)"] = pd.Series([d.date() in vs for d in idx], index=range(len(idx)))

    if os.path.exists(ARCHIVO_EXTRA):
        try:
            extra = pd.read_csv(ARCHIVO_EXTRA)
            extra["fecha"] = pd.to_datetime(extra["fecha"]).dt.date
            for tipo, g in extra.groupby("tipo"):
                dias = set(g["fecha"])
                cal["%s (manual)" % tipo] = pd.Series(
                    [d.date() in dias for d in idx], index=range(len(idx)))
        except Exception as e:
            print("  aviso: no pude leer %s (%s)" % (ARCHIVO_EXTRA, e))
    return cal


def placebos(n_dias, n_ev, veces, semilla=0):
    """Dias falsos con la misma frecuencia. El efecto debe salir ~0."""
    rng = np.random.default_rng(semilla)
    for _ in range(veces):
        m = np.zeros(n_dias, dtype=bool)
        m[rng.choice(n_dias, size=n_ev, replace=False)] = True
        yield pd.Series(m, index=range(n_dias))


# ------------------------------------------------------------------ MT5

def conectar(mt5):
    if mt5.initialize():
        return True
    rutas = glob.glob(r"C:\Program Files\*\terminal64.exe")
    rutas += glob.glob(os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe"))
    for ruta in rutas:
        if mt5.initialize(path=ruta):
            return True
    return False


def buscar(mt5, claves):
    nombres = [s.name for s in (mt5.symbols_get() or [])]
    for c in claves:
        for n in nombres:
            if c in n.upper():
                return n
    return None


def barras(mt5, sim, n=N_BARRAS):
    raw = mt5.copy_rates_from_pos(sim, mt5.TIMEFRAME_D1, 0, n)
    if raw is None or len(raw) < 120:
        return None
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.reset_index(drop=True)


def analizar(nombre, sim, df):
    d = preparar(df)
    cal = construir_calendario(d["time"])

    print()
    print("=" * 78)
    print("%s  [%s]   %d dias   %s a %s" % (
        nombre, sim, len(d), d["time"].iloc[0].date(), d["time"].iloc[-1].date()))
    print("=" * 78)
    print()
    print("  %-22s %5s %9s %9s %8s %8s" % (
        "evento", "n", "exceso%", "crudo%", "t", "t_dir"))
    print("  " + "-" * 66)

    resultados = []
    for etiqueta, mascara in cal.items():
        r = probar(d, mascara, etiqueta)
        if r is None:
            print("  %-22s  muestra insuficiente" % etiqueta)
            continue
        resultados.append(r)
        print("  %-22s %5d %+8.1f%% %+8.1f%% %+8.2f %+8.2f" % (
            r["etiqueta"], r["n_ev"], r["exceso_pct"], r["crudo_pct"],
            r["t"], r["t_dir"]))

    # --- placebo
    sub = d.dropna(subset=["lrango", "latr"])
    n_ref = resultados[0]["n_ev"] if resultados else 60
    ts, exc = [], []
    for m in placebos(len(d), n_ref, 200):
        r = probar(d, m, "placebo")
        if r:
            ts.append(r["t"]); exc.append(r["exceso_pct"])
    print()
    if ts:
        ts = np.array(ts)
        print("  PLACEBO  200 calendarios falsos de %d dias cada uno" % n_ref)
        print("    t medio %+.2f | |t|>2 en %.1f%% de los casos (esperado ~5%%)"
              % (ts.mean(), 100.0 * (np.abs(ts) > 2).mean()))
        print("    exceso%% p5 %+.1f%%  p95 %+.1f%%  <- el ruido llega hasta aqui"
              % (np.percentile(exc, 5), np.percentile(exc, 95)))

    print()
    print("  COMO LEERLO")
    print("    exceso%% = cuanto mas ancho es el rango ese dia, NETO del regimen")
    print("    crudo%%  = la misma comparacion SIN controlar regimen (suele enganar)")
    print("    t       = |t| > 2 es senal; por debajo es ruido")
    print("    t_dir   = el mismo test sobre la DIRECCION; debe salir ~0")
    for r in resultados:
        if abs(r["t"]) > 2:
            lotes = 1.0 / (1.0 + r["exceso_pct"] / 100.0)
            print()
            print("    -> %s: rango %+.1f%% (t=%.2f). Para igualar riesgo ese dia,"
                  % (r["etiqueta"], r["exceso_pct"], r["t"]))
            print("       el tamano seria %.2fx el normal." % lotes)
    return resultados


def main():
    import MetaTrader5 as mt5
    if not conectar(mt5):
        raise SystemExit("No conecta a MT5. Abre MetaTrader 5 y repite.")

    oro = buscar(mt5, ["XAUUSD", "GOLD"])
    brent = buscar(mt5, ["SPOTBRENT", "UKOIL", "BRENT"])
    print("simbolos -> oro: %s | brent: %s" % (oro, brent))
    if not os.path.exists(ARCHIVO_EXTRA):
        print("\n(sin %s: solo se prueban EIA y NFP, que son por regla)" % ARCHIVO_EXTRA)

    for nombre, sim in [("ORO", oro), ("BRENT", brent)]:
        if not sim:
            continue
        df = barras(mt5, sim)
        if df is None:
            print("\n%s: historial insuficiente" % nombre)
            continue
        analizar(nombre, sim, df)

    mt5.shutdown()


if __name__ == "__main__":
    main()
