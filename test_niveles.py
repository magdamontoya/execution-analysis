"""Valida backtest_niveles.py contra series M1 sinteticas de respuesta conocida."""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_niveles import correr_backtest, zigzag, contar_niveles, con_evidencia

rng = np.random.default_rng(13)
FALLOS = []


def chequear(nombre, cond, detalle):
    if not cond:
        FALLOS.append(nombre)
    print("  [%s] %-36s %s" % ("OK" if cond else "FALLO", nombre, detalle))


def serie_m1(dias=90, p0=95.0, sigma_min=0.012):
    """Paseo aleatorio M1, 20h por sesion. Sin estructura de ningun tipo."""
    por_dia = 20 * 60
    n = dias * por_dia
    px = p0 + np.cumsum(rng.normal(0, sigma_min, n))
    amp = np.abs(rng.normal(0, sigma_min, n)) * 1.5 + 0.004
    t0 = pd.Timestamp("2026-01-05 00:00")
    tiempos = []
    for d in range(dias):
        base = t0 + pd.Timedelta(days=d)
        tiempos.extend(base + pd.Timedelta(minutes=m) for m in range(por_dia))
    return pd.DataFrame({"time": tiempos, "open": px, "high": px + amp / 2,
                         "low": px - amp / 2, "close": px})


def inyectar_soportes(df, cada=0.50, aguante=120):
    """
    CONTROL POSITIVO, hecho por post-proceso y no por dinamica.

    Primer intento (barrera reflectante dentro del paseo): fallaba. Rebotar
    hacia arriba mete deriva alcista, el precio se escapa de la zona de los
    niveles y deja de visitarlos. Y al contenerlo con reversion a la media,
    el rango se estrecha tanto que TODOS los niveles del rango son soporte
    y el control aleatorio ve la misma estructura: no hay contraste.

    Esto es directo: se eligen niveles separados 'cada' dolares (esparcidos
    frente al paso de 0.10, para que el azar casi nunca los pille) y, tras
    cada toque, se impide que el minimo baje del nivel durante 'aguante'
    minutos. Eso crea exactamente la asimetria MFE/MAE que el backtest mide,
    en sitios conocidos, sin tocar nada mas de la serie.
    """
    d = df.copy()
    lo = d["low"].to_numpy().copy()
    hi = d["high"].to_numpy().copy()
    base = np.floor(d["low"].min() / cada) * cada
    niveles = np.arange(base, d["high"].max() + cada, cada)
    n = len(lo)
    for lv in niveles:
        i = 0
        while i < n:
            if lo[i] <= lv <= hi[i]:
                j = min(i + aguante, n)
                lo[i:j] = np.maximum(lo[i:j], lv)      # no puede perforar
                hi[i:j] = np.maximum(hi[i:j], lo[i:j])
                i = j
            else:
                i += 1
    d["low"], d["high"] = lo, hi
    d["close"] = np.clip(d["close"].to_numpy(), lo, hi)
    return d, list(niveles)


print("=" * 76)
print("CASO 0 - el zigzag de tres estados no se bloquea")
print("=" * 76)
px = 95 + np.cumsum(rng.normal(0, 0.02, 5000))
sw = zigzag(px, 0.25)
chequear("0-produce giros", len(sw) > 20, "%d giros en 5000 puntos" % len(sw))
cuenta = contar_niveles(sw)
sop, res = con_evidencia(cuenta)
chequear("0-clasifica niveles", len(cuenta) > 5,
         "%d niveles, %d soportes, %d resistencias" % (len(cuenta), len(sop), len(res)))

print()
print("=" * 76)
print("CASO 1 - CONTROL NEGATIVO: paseo aleatorio puro, sin rebotes")
print("  esperado: la ventaja sobre el azar ~0 y |t| < 2 en todos los horizontes")
print("=" * 76)
df = serie_m1(dias=90)
r = correr_backtest(df, spread=0.034, semilla=1, verbose=False)
ts = [v["t"] for v in r.values() if v]
vent = [v["ventaja"] for v in r.values() if v]
print("  t por horizonte:       %s" % ["%+.2f" % t for t in ts])
print("  ventaja por horizonte: %s" % ["%+.4f" % v for v in vent])
chequear("1-sin falso positivo", len(ts) > 0 and max(abs(t) for t in ts) < 2.5,
         "max |t| = %.2f" % (max(abs(t) for t in ts) if ts else -1))

print()
print("=" * 76)
print("CASO 2 - CONTROL POSITIVO: rebote real inyectado en niveles redondos")
print("  esperado: ventaja clara y t grande. Si no lo ve, el test esta ciego.")
print("=" * 76)
df, niveles = inyectar_soportes(serie_m1(dias=90))
print("  niveles con soporte real inyectado: %d, separados 0.50" % len(niveles))
r = correr_backtest(df, spread=0.034, semilla=2, verbose=False)
ts = [v["t"] for v in r.values() if v]
vent = [v["ventaja"] for v in r.values() if v]
print("  t por horizonte:       %s" % ["%+.2f" % t for t in ts])
print("  ventaja por horizonte: %s" % ["%+.4f" % v for v in vent])
chequear("2-detecta el rebote", len(ts) > 0 and max(ts) > 2.5,
         "max t = %.2f, max ventaja = %.4f" % (max(ts) if ts else -1,
                                               max(vent) if vent else 0))

print()
print("=" * 76)
print("CASO 3 - SIN MIRAR AL FUTURO: los niveles del dia D salen de D-1")
print("  Se comprueba que el backtest no usa barras del propio dia que opera.")
print("=" * 76)
df = serie_m1(dias=60)
# si hubiera lookahead, romper el dia siguiente no cambiaria el resultado;
# si no lo hay, alterar SOLO la ventana previa si debe cambiarlo
df2 = df.copy()
mask = df2["time"].dt.date == df2["time"].dt.date.unique()[10]
df2.loc[mask, ["open", "high", "low", "close"]] += 5.0
r1 = correr_backtest(df, spread=0.034, semilla=3, verbose=False)
r2 = correr_backtest(df2, spread=0.034, semilla=3, verbose=False)
cambio = any((r1[h] is None) != (r2[h] is None) or
             (r1[h] and r2[h] and abs(r1[h]["mfe_real"] - r2[h]["mfe_real"]) > 1e-9)
             for h in r1)
chequear("3-la ventana previa influye", cambio,
         "alterar un dia cambia el resultado (no esta congelado)")

print()
print("=" * 76)
print("RESUMEN: %s" % ("TODO OK" if not FALLOS else "FALLOS -> " + ", ".join(FALLOS)))
print("=" * 76)
sys.exit(0 if not FALLOS else 1)
