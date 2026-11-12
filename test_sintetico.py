"""Verifica edge_test.py contra series sinteticas de respuesta conocida."""
import os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_test import metricas

rng = np.random.default_rng(42)
FALLOS = []


def serie(ret, sigma=None, p0=100.0, ruido_rango=0.18):
    """
    OHLC a partir de retornos.
    El rango diario sigue la volatilidad del dia (como en el mercado real,
    aprox. Parkinson: rango ~ 4*sigma) con ruido multiplicativo modesto.
    """
    ret = np.asarray(ret, dtype=float)
    c = np.concatenate([[p0], p0 * np.exp(np.cumsum(ret))])
    if sigma is None:
        sigma = np.full(len(ret), np.std(ret))
    sigma = np.asarray(sigma, dtype=float)
    amp = 4.0 * sigma * np.exp(rng.normal(0, ruido_rango, len(sigma)))
    amp = np.concatenate([[float(np.median(amp))], amp])
    return c, c * (1 + amp / 2), c * (1 - amp / 2)


def chequear(etiqueta, cond, detalle):
    estado = "OK" if cond else "FALLO"
    if not cond:
        FALLOS.append(etiqueta)
    print("  [%s] %s -- %s" % (estado, etiqueta, detalle))


print("=" * 72)
print("CASO 1 - paseo aleatorio, volatilidad CONSTANTE")
print("  esperado: direccion R2 ~ 0  Y  volatilidad R2 ~ 0 (nada que predecir)")
print("=" * 72)
r = rng.normal(0, 0.01, 2500)
m = metricas(*serie(r))
print("  direccion   R2=%.4f  t=%6.2f" % (m["dir_r2"], m["dir_t"]))
print("  volatilidad R2=%.4f  t=%6.2f" % (m["vol_r2"], m["vol_t"]))
print("  acf |r| lag1-3: %s" % ["%.3f" % v for v in m["acf_abs"][:3]])
chequear("1-direccion nula", abs(m["dir_r2"]) < 0.01, "R2=%.4f" % m["dir_r2"])
chequear("1-volatilidad nula", m["vol_r2"] < 0.05, "R2=%.4f" % m["vol_r2"])

print()
print("=" * 72)
print("CASO 2 - paseo aleatorio con CLUSTERING de volatilidad (GARCH 1,1)")
print("  esperado: direccion R2 ~ 0  PERO  volatilidad R2 alto")
print("=" * 72)
n = 2500
om, al, be = 1e-6, 0.10, 0.88
h2 = np.zeros(n)
h2[0] = om / (1 - al - be)
e = np.zeros(n)
for i in range(1, n):
    h2[i] = om + al * e[i - 1] ** 2 + be * h2[i - 1]
    e[i] = rng.normal(0, np.sqrt(h2[i]))
sig = np.sqrt(h2)
m2 = metricas(*serie(e, sigma=sig))
print("  direccion   R2=%.4f  t=%6.2f" % (m2["dir_r2"], m2["dir_t"]))
print("  volatilidad R2=%.4f  t=%6.2f" % (m2["vol_r2"], m2["vol_t"]))
print("  acf |r| lag1-3: %s" % ["%.3f" % v for v in m2["acf_abs"][:3]])
chequear("2-direccion nula", abs(m2["dir_r2"]) < 0.02, "R2=%.4f" % m2["dir_r2"])
chequear("2-volatilidad predecible", m2["vol_r2"] > 0.30, "R2=%.4f" % m2["vol_r2"])
chequear("2-vol >> dir", m2["vol_r2"] > 20 * max(m2["dir_r2"], 1e-6),
         "ratio=%.0fx" % (m2["vol_r2"] / max(m2["dir_r2"], 1e-6)))

print()
print("=" * 72)
print("CASO 3 - CONTROL POSITIVO: retornos con momentum AR(1) real")
print("  esperado: direccion R2 claramente > 0. Si sale 0, el test esta ciego.")
print("=" * 72)
r3 = np.zeros(2500)
for i in range(1, 2500):
    r3[i] = 0.35 * r3[i - 1] + rng.normal(0, 0.01)
m3 = metricas(*serie(r3))
print("  direccion   R2=%.4f  t=%6.2f  pendiente esperada ~0.35" % (m3["dir_r2"], m3["dir_t"]))
chequear("3-detecta momentum", m3["dir_r2"] > 0.05 and m3["dir_t"] > 5,
         "R2=%.4f t=%.1f" % (m3["dir_r2"], m3["dir_t"]))

print()
print("=" * 72)
print("CASO 4 - CONTROL DE ALINEACION: el rango de manana ES el de hoy")
print("  esperado: volatilidad R2 muy alto (~1). Verifica que no hay desfase.")
print("=" * 72)
r4 = rng.normal(0, 0.01, 2500)
s4 = np.abs(np.sin(np.arange(2500) / 40.0)) * 0.012 + 0.002   # vol ciclica, muy predecible
m4 = metricas(*serie(r4, sigma=s4, ruido_rango=0.01))
print("  volatilidad R2=%.4f  t=%7.1f" % (m4["vol_r2"], m4["vol_t"]))
chequear("4-alineacion sin desfase", m4["vol_r2"] > 0.60, "R2=%.4f" % m4["vol_r2"])

print()
print("=" * 72)
print("RESUMEN: %s" % ("TODO OK" if not FALLOS else "FALLOS -> " + ", ".join(FALLOS)))
print("=" * 72)
sys.exit(0 if not FALLOS else 1)
