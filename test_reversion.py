"""Valida reversion() contra procesos OU de vida media conocida."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_test import reversion

rng = np.random.default_rng(7)
fallos = []

def ou(vida_media, n=3000, sigma=0.3, mu=11.0):
    """Genera OU con vida media objetivo."""
    phi = 0.5 ** (1.0 / vida_media)      # coef AR(1): phi^vida = 0.5
    s = np.zeros(n); s[0] = mu
    for i in range(1, n):
        s[i] = mu + phi * (s[i-1] - mu) + rng.normal(0, sigma)
    return s

print("=" * 66)
print("Valida el estimador de vida media del spread")
print("=" * 66)
for objetivo in [3, 10, 30]:
    r = reversion(ou(objetivo))
    err = abs(r["vida_media"] - objetivo) / objetivo
    ok = err < 0.25
    if not ok: fallos.append("vida=%d" % objetivo)
    print("  objetivo %3d dias -> estimado %6.1f  (error %4.0f%%)  t=%7.1f  [%s]"
          % (objetivo, r["vida_media"], err*100, r["t"], "OK" if ok else "FALLO"))

print()
print("Control negativo: paseo aleatorio (sin reversion)")
rw = np.cumsum(rng.normal(0, 0.3, 3000)) + 11.0
r = reversion(rw)
ok = (r["vida_media"] > 200) or not np.isfinite(r["vida_media"])
if not ok: fallos.append("paseo-aleatorio")
print("  vida media estimada: %s   t=%6.2f   [%s]"
      % (("inf" if not np.isfinite(r["vida_media"]) else "%.0f dias" % r["vida_media"]),
         r["t"], "OK" if ok else "FALLO"))

print()
print("=" * 66)
print("RESUMEN: %s" % ("TODO OK" if not fallos else "FALLOS -> " + ", ".join(fallos)))
print("=" * 66)
sys.exit(0 if not fallos else 1)
