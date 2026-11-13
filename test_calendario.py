"""Valida backtest_calendario.py contra series con efecto de evento conocido."""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_calendario import preparar, probar

rng = np.random.default_rng(5)
FALLOS = []


def serie(n=900, efecto_miercoles=0.0, clustering=False):
    """
    Construye barras diarias.
      efecto_miercoles : exceso multiplicativo del rango los miercoles (0.4 = +40%)
      clustering       : volatilidad con memoria (GARCH-ish)
    Devuelve (df, sigma) para que el caso 4 pueda elegir fechas segun el regimen.
    """
    fechas = pd.bdate_range("2023-01-02", periods=n)
    dow = fechas.dayofweek.to_numpy()

    sigma = np.full(n, 0.010)
    if clustering:
        for i in range(1, n):
            sigma[i] = 0.0025 + 0.75 * sigma[i - 1] + 0.20 * abs(rng.normal(0, 0.010))
        sigma = np.clip(sigma, 0.003, 0.08)

    amp = 4.0 * sigma * np.exp(rng.normal(0, 0.15, n))
    if efecto_miercoles:
        amp = amp * np.where(dow == 2, 1.0 + efecto_miercoles, 1.0)

    ret = rng.normal(0, sigma)
    close = 100.0 * np.exp(np.cumsum(ret))
    df = pd.DataFrame({
        "time": fechas,
        "close": close,
        "high": close * (1 + amp / 2),
        "low": close * (1 - amp / 2),
    })
    return df, sigma


def correr(df, mascara=None):
    d = preparar(df)
    if mascara is None:
        mascara = pd.Series(d["time"].dt.dayofweek.to_numpy() == 2, index=d.index)
    else:
        mascara = pd.Series(np.asarray(mascara), index=d.index)
    return probar(d, mascara, "evento")


def chequear(nombre, cond, detalle):
    if not cond:
        FALLOS.append(nombre)
    print("  [%s] %-34s %s" % ("OK" if cond else "FALLO", nombre, detalle))


print("=" * 76)
print("CASO 1 - sin efecto de evento, vol constante")
print("  esperado: exceso ~0%, |t| < 2")
print("=" * 76)
df, _ = serie(); r = correr(df)
print("  exceso %+.1f%%  t=%+.2f  (crudo %+.1f%%)" % (r["exceso_pct"], r["t"], r["crudo_pct"]))
chequear("1-sin falso positivo", abs(r["t"]) < 2.5, "t=%+.2f" % r["t"])

print()
print("=" * 76)
print("CASO 2 - CONTROL POSITIVO: +40% de rango los miercoles")
print("  esperado: exceso ~+40%, t grande. Si no lo ve, el test esta ciego.")
print("=" * 76)
df, _ = serie(efecto_miercoles=0.40); r = correr(df)
print("  exceso %+.1f%%  t=%+.2f  (crudo %+.1f%%)" % (r["exceso_pct"], r["t"], r["crudo_pct"]))
chequear("2-detecta el efecto", r["t"] > 5 and 28 < r["exceso_pct"] < 55,
         "exceso %+.1f%% t=%+.2f" % (r["exceso_pct"], r["t"]))

print()
print("=" * 76)
print("CASO 3 - con clustering de volatilidad Y efecto real de +40%")
print("  esperado: lo sigue estimando bien pese al regimen variable")
print("=" * 76)
df, _ = serie(efecto_miercoles=0.40, clustering=True); r = correr(df)
print("  exceso %+.1f%%  t=%+.2f  (crudo %+.1f%%)" % (r["exceso_pct"], r["t"], r["crudo_pct"]))
chequear("3-robusto al regimen", r["t"] > 5 and 25 < r["exceso_pct"] < 58,
         "exceso %+.1f%%" % r["exceso_pct"])

print()
print("=" * 76)
print("CASO 4 - LA TRAMPA REAL: eventos que se CONVOCAN en epocas volatiles")
print("         (la OPEP se reune mas durante crisis) pero que en si mismos")
print("         no aportan nada. El ATR retrasado debe descartarlos.")
print("  esperado: 'crudo%' enganado al alza, 'exceso%' (controlado) ~0")
print("=" * 76)
df, sigma = serie(clustering=True)
# las fechas del evento se eligen con probabilidad proporcional al regimen YA
# en curso: el evento no causa nada, solo aparece cuando ya hay volatilidad
peso = np.maximum(pd.Series(sigma).rolling(20).mean().shift(1).to_numpy(), 1e-9)
peso[np.isnan(peso)] = np.nanmedian(peso)
peso = peso ** 10
peso = peso / peso.sum()
idx = rng.choice(len(df), size=70, replace=False, p=peso)
m = np.zeros(len(df), dtype=bool); m[idx] = True
r = correr(df, m)
print("  exceso %+.1f%%  t=%+.2f   <- controlado por ATR" % (r["exceso_pct"], r["t"]))
print("  crudo  %+.1f%%             <- sin controlar" % r["crudo_pct"])
chequear("4-el control mata el espejismo", abs(r["t"]) < 3.0,
         "t controlado %+.2f" % r["t"])
chequear("4-efecto controlado no significativo", abs(r["exceso_pct"]) < 6,
         "exceso %+.1f%% (nada real que encontrar)" % r["exceso_pct"])
print("  nota: 'crudo' usa medianas, que ya son bastante robustas, asi que el")
print("        sesgo sin controlar sale moderado. El control lo deja en nada.")

print()
print("=" * 76)
print("RESUMEN: %s" % ("TODO OK" if not FALLOS else "FALLOS -> " + ", ".join(FALLOS)))
print("=" * 76)
sys.exit(0 if not FALLOS else 1)
