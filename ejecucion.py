# -*- coding: utf-8 -*-
# ================================================================
# EJECUCION — lo que ningun competidor mide
#
# Cruza cada operacion del usuario con los ticks reales del mercado en
# ese milisegundo y responde:
#
#   1. Cuanto pago de spread, en dinero y como % de lo que gano
#   2. Si le dieron el precio que habia en el mercado (deslizamiento)
#   3. Cuanto habria ahorrado entrando con orden limitada
#   4. En que horas su ejecucion se degrada
#   5. Que parte de su resultado se la come la ejecucion
#
# Los ticks se leen de una carpeta con <SIMBOLO>_ticks.parquet.
# Si no hay ticks de un instrumento, ese queda fuera y se avisa.
# ================================================================

import os

import numpy as np
import pandas as pd

TAM_CONTRATO = {'XAUUSD': 100.0, 'NAS100': 1.0, 'US30': 1.0, 'SPOTBRENT': 100.0,
                'SpotBrent': 100.0, 'BTCUSD': 1.0, 'ETHUSD': 1.0}


def _valor_punto(simbolo):
    return TAM_CONTRATO.get(simbolo, TAM_CONTRATO.get(str(simbolo).upper(), 1.0))


def cargar_ticks(carpeta, simbolo):
    for nombre in (simbolo, str(simbolo).upper(), str(simbolo).capitalize()):
        for ext in ('.parquet', '.csv.gz'):
            ruta = os.path.join(carpeta, nombre + '_ticks' + ext)
            if os.path.exists(ruta):
                d = pd.read_parquet(ruta) if ext == '.parquet' else pd.read_csv(ruta)
                d = d.sort_values('time_msc').reset_index(drop=True)
                return d
    return None


def _buscar(ticks, momentos):
    """Para cada momento, el tick vigente en ese milisegundo."""
    t = ticks['time_msc'].values
    # la unidad de los datetime cambia entre versiones de pandas: se fuerza a nanosegundos
    idx = pd.DatetimeIndex(pd.to_datetime(momentos))
    try:
        idx = idx.as_unit('ns')
    except AttributeError:
        pass
    ms = (idx.astype('int64') // 10 ** 6).values
    pos = np.searchsorted(t, ms, side='right') - 1
    ok = (pos >= 0) & (pos < len(t))
    pos = np.clip(pos, 0, len(t) - 1)
    bid = ticks['bid'].values[pos]; ask = ticks['ask'].values[pos]
    dentro = ok & (np.abs(t[pos] - ms) < 5 * 60 * 1000)      # a menos de 5 minutos
    return bid, ask, dentro


def analizar(ops, carpeta_ticks):
    """Devuelve el analisis de ejecucion operacion a operacion."""
    if not os.path.isdir(carpeta_ticks):
        return None
    filas = []
    sin_ticks = []
    for sim, g in ops.groupby('simbolo'):
        ticks = cargar_ticks(carpeta_ticks, sim)
        if ticks is None or len(ticks) < 1000:
            sin_ticks.append(sim); continue
        vp = _valor_punto(sim)
        g = g.dropna(subset=['abierta']).copy()
        if len(g) == 0: continue
        bid_a, ask_a, ok_a = _buscar(ticks, g['abierta'])
        bid_c, ask_c, ok_c = _buscar(ticks, g['cerrada'].fillna(g['abierta']))
        medio_a = (bid_a + ask_a) / 2
        spread_a = ask_a - bid_a
        spread_c = ask_c - bid_c
        compra = (g['tipo'] == 'COMPRA').values
        entrada = g['entrada'].values
        lotes = g['lotes'].values
        # precio que le tocaba pagar: ask si compra, bid si vende
        debido = np.where(compra, ask_a, bid_a)
        # deslizamiento: lo que pago de mas respecto a lo que habia
        desliz = np.where(compra, entrada - debido, debido - entrada)
        # si el precio del usuario y el del tick difieren mucho mas que el spread, no son
        # comparables (otro broker, otro contrato): ese deslizamiento no se puede medir
        fiable = np.abs(desliz) <= 10 * np.maximum(spread_a, 1e-9)
        desliz = np.where(fiable, desliz, np.nan)
        coste_spread = (spread_a + spread_c) * vp * lotes       # entrar y salir
        filas.append(pd.DataFrame(dict(
            simbolo=sim, abierta=g['abierta'].values, tipo=g['tipo'].values, lotes=lotes,
            pnl=g['pnl_neto'].values if 'pnl_neto' in g else g['pnl'].values,
            spread_entrada=spread_a, spread_salida=spread_c,
            spread_medio_sim=float(np.median(ticks['ask'].values - ticks['bid'].values)),
            coste_spread=coste_spread,
            deslizamiento=desliz * vp * lotes,
            desliz_medible=fiable,
            precio_medio=medio_a, valido=ok_a & ok_c)))
    if not filas:
        return dict(ok=False, sin_ticks=sin_ticks,
                    aviso='No hay ticks para ninguno de los instrumentos operados.')
    d = pd.concat(filas, ignore_index=True)
    d = d[d['valido']]
    if len(d) == 0:
        return dict(ok=False, sin_ticks=sin_ticks,
                    aviso='Los ticks disponibles no cubren las fechas de las operaciones.')
    return dict(ok=True, detalle=d, sin_ticks=sin_ticks)


def resumen(res):
    if not res or not res.get('ok'):
        return None
    d = res['detalle']
    bruto = float(d['pnl'].sum())
    coste = float(d['coste_spread'].sum())
    desliz = float(d['deslizamiento'].sum(skipna=True))
    medibles = int(d['desliz_medible'].sum()) if 'desliz_medible' in d else len(d)
    ganado = float(d.loc[d['pnl'] > 0, 'pnl'].sum())
    return dict(operaciones=len(d),
                coste_spread_total=round(coste, 2),
                coste_por_operacion=round(coste / max(len(d), 1), 2),
                deslizamiento_total=round(desliz, 2),
                operaciones_con_deslizamiento_medible=medibles,
                deslizamiento_por_operacion=round(desliz / max(medibles, 1), 3),
                resultado_sin_costes=round(bruto + coste, 2),
                resultado_real=round(bruto, 2),
                pct_de_lo_ganado=round(coste / max(ganado, 1) * 100, 1),
                veces_el_resultado=round(abs(coste / bruto), 1) if bruto else None)


def por_instrumento(res):
    if not res or not res.get('ok'): return pd.DataFrame()
    d = res['detalle']
    g = d.groupby('simbolo').agg(
        operaciones=('pnl', 'size'),
        resultado=('pnl', 'sum'),
        coste_spread=('coste_spread', 'sum'),
        coste_medio=('coste_spread', 'mean'),
        spread_tipico=('spread_medio_sim', 'first'),
        deslizamiento=('deslizamiento', 'sum')).round(2).reset_index()
    g['coste_vs_resultado'] = np.where(g['resultado'] != 0,
                                       (g['coste_spread'] / g['resultado'].abs()).round(2), None)
    return g.sort_values('coste_spread', ascending=False)


def por_hora(res):
    """A que horas la ejecucion es peor: spread mas ancho y mas deslizamiento."""
    if not res or not res.get('ok'): return pd.DataFrame()
    d = res['detalle'].copy()
    d['hora'] = pd.to_datetime(d['abierta']).dt.hour
    g = d.groupby('hora').agg(
        operaciones=('pnl', 'size'),
        spread_medio=('spread_entrada', 'mean'),
        coste_medio=('coste_spread', 'mean'),
        deslizamiento_medio=('deslizamiento', 'mean'),
        resultado=('pnl', 'sum')).round(3).reset_index()
    if len(g):
        base = g['spread_medio'].median()
        g['veces_lo_normal'] = (g['spread_medio'] / base).round(2) if base else None
    return g


def limitada_vs_mercado(res, ahorro_por_spread=0.5):
    """Cuanto habria ahorrado entrando con orden limitada en lugar de a mercado.

    Una limitada se llena, de media, medio spread mejor que una orden a mercado.
    """
    if not res or not res.get('ok'): return None
    d = res['detalle']
    ahorro = float((d['spread_entrada'] * ahorro_por_spread).sum() *
                   0)  # se calcula abajo con lotes y valor punto
    # recalcular con el coste real: el spread de entrada ya viene en dinero dentro de coste_spread
    mitad = float((d['coste_spread'] * ahorro_por_spread / 2).sum())
    bruto = float(d['pnl'].sum())
    return dict(ahorro_estimado=round(mitad, 2),
                resultado_con_limitadas=round(bruto + mitad, 2),
                por_operacion=round(mitad / max(len(d), 1), 2),
                cambia_el_signo=bool(bruto < 0 <= bruto + mitad))


def veredicto(res):
    """Las frases que van en el informe."""
    r = resumen(res)
    if not r: return []
    out = []
    out.append('Pagaste ' + str(abs(r['coste_spread_total'])) + ' USD de spread en ' +
               str(r['operaciones']) + ' operaciones: ' + str(r['coste_por_operacion']) +
               ' USD cada una.')
    if r['veces_el_resultado'] and r['veces_el_resultado'] >= 1:
        out.append('Ese coste es ' + str(r['veces_el_resultado']) +
                   ' veces tu resultado final: la ejecucion pesa mas que tus aciertos.')
    elif r['pct_de_lo_ganado'] > 0:
        out.append('Representa el ' + str(r['pct_de_lo_ganado']) +
                   '% de todo lo que ganaste en tus operaciones positivas.')
    out.append('Sin costes de ejecucion tu resultado habria sido ' + str(r['resultado_sin_costes']) +
               ' USD, frente a los ' + str(r['resultado_real']) + ' reales.')
    if r['deslizamiento_total'] < -1 and r.get('operaciones_con_deslizamiento_medible', 0) >= 10:
        out.append('Ademas perdiste ' + str(abs(r['deslizamiento_total'])) + ' USD de deslizamiento en ' +
                   str(r['operaciones_con_deslizamiento_medible']) + ' operaciones: te ejecutaron peor '
                   'que el precio que habia en el mercado en ese milisegundo.')
    lv = limitada_vs_mercado(res)
    if lv and lv['ahorro_estimado'] > 0:
        out.append('Entrando con ordenes limitadas habrias ahorrado unos ' + str(lv['ahorro_estimado']) +
                   ' USD' + (', y tu resultado pasaria de negativo a positivo.' if lv['cambia_el_signo'] else '.'))
    h = por_hora(res)
    if len(h) and 'veces_lo_normal' in h.columns:
        peor = h.nlargest(1, 'veces_lo_normal')
        if len(peor) and float(peor['veces_lo_normal'].iloc[0]) > 1.5:
            out.append('Tu peor hora para ejecutar es las ' + str(int(peor['hora'].iloc[0])) +
                       ':00, con el spread ' + str(round(float(peor['veces_lo_normal'].iloc[0]), 2)) +
                       ' veces lo normal.')
    return out
