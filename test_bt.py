# -*- coding: utf-8 -*-
"""
test_bt.py - Controles del motor de backtest. Corre sin datos y sin MT5.

    python test_bt.py

Si alguno falla, NINGUN resultado del backtest vale, por bonita que sea
la curva de equity.

  1  COSTE       entrar y salir sin movimiento pierde exactamente un spread
  2a NEGATIVO    al azar y sin stops, el motor no inventa ventaja (t_bruto ~ 0)
  2b SESGO       cuanto castiga la regla "el stop salta primero". Es un numero
                 conocido, no un fallo, y cambia como hay que leer la tabla
  3  POSITIVO    con momentum inyectado la ruptura lo encuentra, Y a partir de
                 que fuerza lo encuentra (escalera de calibracion)
  4  EJECUCION   el precio de entrada sale de la barra SIGUIENTE a la senal

El 4 es el error mas comun en backtesting y aqui se comprueba con numeros,
no leyendo el codigo.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_core import correr, estadisticas
from estrategias import ruptura_volatilidad, aleatoria

rng = np.random.default_rng(11)
FALLOS = []


def chequear(nombre, cond, detalle):
    if not cond:
        FALLOS.append(nombre)
    print("  [%s] %-34s %s" % ("OK" if cond else "FALLO", nombre, detalle))


def barras_sinteticas(n=8000, p0=95.0, sigma=0.02, spread=0.030, phi=0.0):
    """
    Barras OHLC directas, sin pasar por ticks, con spread constante conocido.
    phi = autocorrelacion de los retornos: 0 es paseo aleatorio puro,
    y cuanto mas alto, mas se encadenan los movimientos (mas momentum).
    """
    e = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    c = p0 + np.cumsum(r)
    amp = np.abs(rng.normal(0, sigma, n)) * 1.6 + 0.004
    return pd.DataFrame({
        "time": pd.date_range("2026-01-05", periods=n, freq="5min"),
        "open": c, "high": c + amp / 2, "low": c - amp / 2, "close": c,
        "spread": np.full(n, spread), "ticks": 50,
        "rel": np.log((c + amp / 2) / (c - amp / 2)),
    })


def bruto(ops):
    """PnL antes de costes: para ver si el motor inventa ventaja."""
    g = (ops["pnl"] + ops["coste"]).to_numpy(dtype=float)
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
    return g.mean(), t


print("=" * 78)
print("CONTROL 1 - EL COSTE: entrar y salir sin movimiento cuesta un spread")
print("  esperado: pnl = -spread exacto en cada operacion")
print("=" * 78)
SPREAD = 0.030
b = barras_sinteticas(n=400, sigma=0.0, spread=SPREAD)
b["high"] = b["close"] + 0.001
b["low"] = b["close"] - 0.001
s = np.zeros(400)
s[50] = 1
s[200] = -1
sin = np.full(400, np.nan)
ops = correr(b, s, sin, sin, max_barras=1)
pnls = ops["pnl"].to_numpy() if len(ops) else np.array([])
print("  operaciones: %d   pnl: %s   spread: %.5f" % (
    len(ops), ["%+.5f" % v for v in pnls], SPREAD))
chequear("1-cobra exactamente el spread",
         len(ops) == 2 and np.allclose(pnls, -SPREAD, atol=1e-6),
         "pnl medio %+.5f contra -%.5f" % (pnls.mean() if len(pnls) else 0, SPREAD))

print()
print("=" * 78)
print("CONTROL 2a - NEGATIVO: al azar, sin stop ni objetivo, 24 barras fijas")
print("  Aisla la CONTABILIDAD de la gestion de la posicion.")
print("  esperado: pnl bruto (antes de costes) sin ventaja, |t| < 2,5")
print("            y pnl neto = aproximadamente -1 spread")
print("=" * 78)
b = barras_sinteticas(n=8000, spread=SPREAD)
sa, _, _ = aleatoria(b, n_senales=500, semilla=3)
sin = np.full(len(b), np.nan)
ops = correr(b, sa, sin, sin, max_barras=24)
e = estadisticas(ops)
gm, gt = bruto(ops)
print("  n=%d" % e["n"])
print("  BRUTO (sin costes):  %+.5f   t = %+.2f   <- aqui esta la prueba" % (gm, gt))
print("  NETO  (con costes):  %+.5f   = %.2f spreads" % (
    e["expectativa"], e["en_spreads"]))
chequear("2a-el motor no inventa ventaja",
         e["n"] > 50 and np.isfinite(gt) and abs(gt) < 2.5,
         "t bruto = %+.2f" % gt)
chequear("2a-cobra el spread al azar",
         e["expectativa"] < 0 and -1.5 < e["en_spreads"] < -0.5,
         "neto = %.2f spreads" % e["en_spreads"])

print()
print("=" * 78)
print("CONTROL 2b - EL PRECIO DE SER PESIMISTA (no es un fallo: es un numero)")
print("  Cuando el stop y el objetivo caben en la misma barra, el motor")
print("  supone que salto el STOP. No se sabe el orden real dentro de la")
print("  barra, y suponer lo bueno es como se fabrican curvas falsas.")
print("  Esa regla castiga a TODA estrategia con stops. Cuanto:")
print("=" * 78)
sa, st, so = aleatoria(b, n_senales=500, semilla=3)
ops = correr(b, sa, st, so, max_barras=48)
e2 = estadisticas(ops)
gm2, gt2 = bruto(ops)
print("  n=%d   salidas por stop %.0f%% / objetivo %.0f%% / tiempo %.0f%%" % (
    e2["n"], e2["pct_stop"], e2["pct_objetivo"], e2["pct_tiempo"]))
print("  BRUTO con stop 1 ATR / objetivo 2 ATR: %+.5f  t = %+.2f" % (gm2, gt2))
print("  NETO: %.2f spreads por operacion" % e2["en_spreads"])
print()
print("  CONSECUENCIA, y es la razon de que ALEATORIA este en el catalogo:")
print("  el suelo contra el que se compara NO es cero. Es lo que saca")
print("  ALEATORIA con los mismos stops. Una estrategia a -0,5 spreads no")
print("  es mala si el azar con la misma gestion saca -1,1.")
chequear("2b-el sesgo es pesimista, nunca optimista",
         gm2 <= 0.0, "bruto %+.5f (negativo o cero = correcto)" % gm2)
chequear("2b-y esta acotado",
         -3.0 < e2["en_spreads"] < -0.5, "%.2f spreads" % e2["en_spreads"])

print()
print("=" * 78)
print("CONTROL 3 - POSITIVO Y SU CALIBRACION")
print("  Momentum real inyectado como AR(1) en los retornos. phi mide su")
print("  fuerza: phi=0 es azar puro. La escalera dice a partir de que")
print("  fuerza la ruptura lo ve -y por tanto que es capaz de descartar.")
print("=" * 78)
print()
print("  %6s %7s %11s %9s %9s" % ("phi", "n", "expectativa", "t", "aciertos"))
print("  " + "-" * 46)
escalera = []
for phi in [0.00, 0.45, 0.70, 0.85]:
    bb = barras_sinteticas(n=8000, spread=0.010, phi=phi)
    sv, stv, sov = ruptura_volatilidad(bb, ventana=12, k_entrada=0.1)
    o = correr(bb, sv, stv, sov, max_barras=24)
    ee = estadisticas(o)
    escalera.append((phi, ee))
    print("  %6.2f %7d %+11.5f %+9.2f %8.0f%%" % (
        phi, ee["n"], ee["expectativa"], ee["t"], ee["aciertos"]))

d = dict(escalera)
chequear("3-ve el momentum fuerte",
         d[0.85]["n"] > 30 and d[0.85]["t"] > 2,
         "phi=0,85 -> t = %+.2f" % d[0.85]["t"])
chequear("3-y no ve nada donde no hay",
         d[0.00]["t"] < 0,
         "phi=0 -> t = %+.2f (negativo: solo paga costes)" % d[0.00]["t"])

umbral = [phi for phi, ee in escalera if ee["t"] > 2]
print()
if umbral:
    print("  LIMITE DEL TEST: detecta momentum desde phi = %.2f." % min(umbral))
    print("  Por debajo de eso no lo ve. Asi que un resultado plano sobre tus")
    print("  datos reales descarta momentum FUERTE, no descarta momentum debil.")
    print("  Decir mas que eso seria pasarse de lo que el test puede sostener.")
else:
    print("  AVISO: el test no detecta momentum ni con phi alto. Esta ciego.")

print()
print("=" * 78)
print("CONTROL 4 - EJECUCION: la entrada usa la barra SIGUIENTE a la senal")
print("  Se mete un salto enorme solo en la barra posterior a la senal.")
print("  Si el motor ejecutara en la barra de la senal, no cambiaria nada.")
print("=" * 78)
b4 = barras_sinteticas(n=300, sigma=0.0, spread=0.0)
b4["high"] = b4["close"] + 0.001
b4["low"] = b4["close"] - 0.001
s4 = np.zeros(300)
s4[100] = 1
sin4 = np.full(300, np.nan)
ea = float(correr(b4, s4, sin4, sin4, max_barras=1)["entrada"].iloc[0])

b4b = b4.copy()
SALTO = 7.0
b4b.loc[101, ["open", "high", "low", "close"]] += SALTO
eb = float(correr(b4b, s4, sin4, sin4, max_barras=1)["entrada"].iloc[0])

print("  entrada sin salto %.4f | con salto en la barra 101 %.4f | dif %+.4f" % (
    ea, eb, eb - ea))
chequear("4-ejecuta en la barra i+1", abs((eb - ea) - SALTO) < 1e-6,
         "se movio %.4f, igual al salto inyectado" % (eb - ea))

b4c = b4.copy()
b4c.loc[100, ["open", "high", "low"]] -= 3.0      # close intacto: misma senal
ec = float(correr(b4c, s4, sin4, sin4, max_barras=1)["entrada"].iloc[0])
chequear("4b-la barra de la senal no fija el precio", abs(ec - ea) < 1e-6,
         "entrada %.4f, sin cambio" % ec)

print()
print("=" * 78)
print("RESUMEN: %s" % ("TODO OK" if not FALLOS else "FALLOS -> " + ", ".join(FALLOS)))
print("=" * 78)
if FALLOS:
    print("No uses ningun resultado del backtest hasta arreglar esto.")
sys.exit(0 if not FALLOS else 1)
