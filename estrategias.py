# -*- coding: utf-8 -*-
"""
estrategias.py - Las ideas a probar, todas con la misma firma
Magdalena Montoya | Diversifica

    estrategia(barras, **params) -> (senales, stops, objetivos)

Tres arrays de la misma longitud que 'barras'. La senal de la posicion i
se decide con datos de la barra i y de antes, NUNCA de i+1. El motor
(bt_core.correr) la ejecuta en la apertura de i+1.

POR QUE TODAS CON LA MISMA FIRMA
  Para que compitan en igualdad. Si cada estrategia trae su propio motor,
  su propio coste y su propia forma de contar, la que gana es la que tiene
  el motor mas generoso. Un solo motor, un solo coste, una sola tabla.

LA ALEATORIA NO ES BROMA
  Es el control negativo y es la mas importante de las seis. Entra al azar
  con la misma frecuencia que las demas. Su resultado es el suelo: si una
  estrategia no le gana con claridad, lo que parecia una senal era el
  numero de operaciones, no la idea.
"""

import numpy as np
import pandas as pd

from bt_core import atr


def _rsi(close, n=14):
    c = np.asarray(close, dtype=float)
    d = np.diff(c, prepend=c[0])
    sube = pd.Series(np.where(d > 0, d, 0.0)).rolling(n).mean()
    baja = pd.Series(np.where(d < 0, -d, 0.0)).rolling(n).mean()
    rs = sube / baja.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).to_numpy()


def _vacio(n):
    return np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)


# ------------------------------------------------------- 1. ruptura por vol

def ruptura_volatilidad(barras, ventana=24, k_entrada=0.25, k_stop=1.0,
                        k_objetivo=2.0, n_atr=20):
    """
    La unica de las seis apoyada en algo que medimos: la volatilidad de
    manana SI es predecible (R2 ~ 0,30). La direccion no. Asi que la
    volatilidad no se usa para decidir el lado, se usa para decidir el
    tamano del movimiento que cuenta como ruptura y donde va el stop.

    Entra si el cierre supera el maximo de las 'ventana' barras previas
    por mas de k_entrada * ATR. Al revés para los cortos.
    """
    c = barras["close"].to_numpy(dtype=float)
    n = len(c)
    a = atr(barras, n_atr)
    maxp = pd.Series(barras["high"]).rolling(ventana).max().shift(1).to_numpy()
    minp = pd.Series(barras["low"]).rolling(ventana).min().shift(1).to_numpy()

    s = np.zeros(n)
    # comparar con NaN da False, asi que las barras sin ATR ni maximo
    # previo no generan senal por si solas. El filtro de abajo lo asegura.
    with np.errstate(invalid="ignore"):
        larga = c > (maxp + k_entrada * a)
        corta = c < (minp - k_entrada * a)
    s[larga] = 1
    s[corta] = -1
    s[~np.isfinite(a) | ~np.isfinite(maxp) | ~np.isfinite(minp)] = 0
    return s, k_stop * a, k_objetivo * a


# -------------------------------------------------- 2. ruptura de apertura

def ruptura_apertura(barras, barras_rango=12, k_stop=1.0, k_objetivo=2.0,
                     n_atr=20):
    """
    Las primeras 'barras_rango' barras de cada dia definen un rango; el
    resto del dia se operan sus roturas. Es la estructura intradia mas
    vieja que hay y merece una prueba limpia antes de descartarla.

    El rango se cierra ANTES de poder operar, asi que no hay forma de
    mirar al futuro: a media manana el rango de la apertura ya es pasado.
    """
    c = barras["close"].to_numpy(dtype=float)
    n = len(c)
    a = atr(barras, n_atr)
    dia = pd.to_datetime(barras["time"]).dt.date.to_numpy()

    s = np.zeros(n)
    i = 0
    while i < n:
        fin = i
        while fin < n and dia[fin] == dia[i]:
            fin += 1
        if fin - i > barras_rango:
            hi = barras["high"].to_numpy()[i:i + barras_rango].max()
            lo = barras["low"].to_numpy()[i:i + barras_rango].min()
            for j in range(i + barras_rango, fin):
                if c[j] > hi:
                    s[j] = 1
                elif c[j] < lo:
                    s[j] = -1
        i = fin
    s[~np.isfinite(a)] = 0
    return s, k_stop * a, k_objetivo * a


# ------------------------------------------------ 3. reversion intradia

def reversion_intradia(barras, ventana=48, z_entrada=2.0, k_stop=1.5,
                       n_atr=20):
    """
    Contra el movimiento: si el precio se aleja mas de z_entrada
    desviaciones de su media de 'ventana' barras, apostar a que vuelve.
    El objetivo es la media, no un multiplo del ATR: si la idea es que
    vuelve a la media, el objetivo tiene que ser la media.
    """
    c = barras["close"].to_numpy(dtype=float)
    n = len(c)
    a = atr(barras, n_atr)
    med = pd.Series(c).rolling(ventana).mean().to_numpy()
    sd = pd.Series(c).rolling(ventana).std(ddof=1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (c - med) / sd

    s = np.zeros(n)
    with np.errstate(invalid="ignore"):
        s[z > z_entrada] = -1      # muy arriba -> corto
        s[z < -z_entrada] = 1      # muy abajo -> largo
    s[~np.isfinite(a) | ~np.isfinite(z)] = 0
    objetivo = np.abs(c - med)                          # distancia a la media
    objetivo[~np.isfinite(objetivo)] = np.nan
    return s, k_stop * a, objetivo


# --------------------------------------------- 4. su operativa manual

def cruce_ma_rsi(barras, rapida=10, lenta=20, rsi_n=14, rsi_alto=55,
                 rsi_bajo=45, k_stop=1.0, k_objetivo=2.0, n_atr=20):
    """
    Version medible de lo que operas a mano: MA10 contra MA20 con filtro
    de RSI. No es tu criterio exacto -el Squeeze y el flujo de ordenes no
    se pueden reconstruir desde ticks de bid/ask- pero es la parte que SI
    se puede escribir sin ambiguedad, y es la que se puede falsar.

    Si esto sale plano, no demuestra que tu operativa sea plana. Demuestra
    que la parte mecanica de tu operativa es plana, y entonces lo que
    aporta valor es la parte que no esta aqui.
    """
    c = barras["close"].to_numpy(dtype=float)
    n = len(c)
    a = atr(barras, n_atr)
    mr = pd.Series(c).rolling(rapida).mean().to_numpy()
    ml = pd.Series(c).rolling(lenta).mean().to_numpy()
    r = _rsi(c, rsi_n)

    arriba = mr > ml
    cruce_alza = np.zeros(n, dtype=bool)
    cruce_baja = np.zeros(n, dtype=bool)
    cruce_alza[1:] = arriba[1:] & ~arriba[:-1]
    cruce_baja[1:] = ~arriba[1:] & arriba[:-1]

    s = np.zeros(n)
    s[cruce_alza & (r > rsi_alto)] = 1
    s[cruce_baja & (r < rsi_bajo)] = -1
    s[~np.isfinite(a) | ~np.isfinite(r)] = 0
    return s, k_stop * a, k_objetivo * a


# ------------------------------------------- 5. CONTROL NEGATIVO: al azar

def aleatoria(barras, n_senales=200, semilla=0, k_stop=1.0, k_objetivo=2.0,
              n_atr=20):
    """
    Entra al azar, con la misma gestion de stop y objetivo que las demas.
    Es el suelo contra el que se mide todo lo anterior.

    Lo que tiene que salir: expectativa negativa de aproximadamente un
    spread por operacion, y |t| por debajo de 2. Si la aleatoria sale
    rentable, el motor tiene un fallo y ningun otro resultado vale nada.
    """
    n = len(barras)
    a = atr(barras, n_atr)
    rng = np.random.default_rng(semilla)
    validas = np.where(np.isfinite(a))[0]
    validas = validas[validas < n - 2]
    s = np.zeros(n)
    if len(validas):
        elegidas = rng.choice(validas, size=min(n_senales, len(validas)),
                              replace=False)
        s[elegidas] = rng.choice([-1.0, 1.0], size=len(elegidas))
    return s, k_stop * a, k_objetivo * a


# ------------------------------------------ 6. REFERENCIA: comprar y aguantar

def comprar_mantener(barras, **kw):
    """
    Una sola compra en la primera barra con ATR valido, sin stop ni
    objetivo. Es la referencia honesta: si una estrategia que opera 300
    veces no le gana a comprar una vez y no mirar, las 300 operaciones
    solo han servido para pagar spreads.

    Hay que correrla con max_barras alto (el motor cierra por tiempo).
    """
    n = len(barras)
    a = atr(barras, 20)
    s = np.zeros(n)
    validas = np.where(np.isfinite(a))[0]
    if len(validas):
        s[validas[0]] = 1
    return s, np.full(n, np.nan), np.full(n, np.nan)


CATALOGO = {
    "ruptura_vol": ruptura_volatilidad,
    "ruptura_apertura": ruptura_apertura,
    "reversion": reversion_intradia,
    "ma_rsi": cruce_ma_rsi,
    "ALEATORIA": aleatoria,
    "comprar_mantener": comprar_mantener,
}
