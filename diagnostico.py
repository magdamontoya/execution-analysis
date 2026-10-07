# -*- coding: utf-8 -*-
# ================================================================
# DIAGNOSTICO — que dicen las operaciones de quien las hizo
#
# Recibe el DataFrame que devuelve lector.leer() y calcula todo sin
# tocar MetaTrader: funciona con el archivo que suba el usuario.
#
#   resumen()        aciertos, ganancia y perdida media, cuantas borra cada perdida
#   tabla()          resultado por duracion, tamano, hora, instrumento, dia
#   monte_carlo()    probabilidad de ruina con SU distribucion real
#   reglas()         las reglas que salen de sus propios numeros
#   curva()          su curva real frente a la que tendria sin las peores
# ================================================================

import numpy as np
import pandas as pd

HORAS_LARGA = 4.0        # a partir de aqui una operacion se considera larga


def preparar(d, capital=None):
    """Añade las columnas que necesitan los analisis."""
    d = d.copy()
    if 'pnl_neto' not in d.columns:
        d['pnl_neto'] = d['pnl']
    d['gana'] = (d['pnl_neto'] > 0).astype(int)
    d['hora'] = pd.to_datetime(d['abierta']).dt.hour + pd.to_datetime(d['abierta']).dt.minute / 60
    d['dia'] = pd.to_datetime(d['abierta']).dt.normalize()
    d['dia_semana'] = pd.to_datetime(d['abierta']).dt.dayofweek
    d['orden_del_dia'] = d.groupby('dia').cumcount() + 1
    d['tras_perdida'] = (d['pnl_neto'].shift(1) < 0).astype(int)
    d['pnl_acumulado_dia'] = d.groupby('dia')['pnl_neto'].cumsum() - d['pnl_neto']
    # capital estimado si no lo dan: el mayor resultado acumulado, como suelo
    if capital is None:
        eq = d['pnl_neto'].cumsum()
        capital = max(float(-eq.min()) * 2, abs(float(d['pnl_neto'].sum())) * 2, 1000.0)
    d.attrs['capital'] = float(capital)
    d['exposicion'] = d['lotes'] * d['entrada']
    d['veces_capital'] = (d['exposicion'] / capital).round(2)
    d['grupo_minutos'] = pd.cut(d['minutos'], [-1, 5, 15, 60, HORAS_LARGA * 60, 1e9],
                                labels=['0-5 min', '5-15 min', '15-60 min', '1-4 horas', 'mas de 4 horas'])
    d['grupo_tamano'] = pd.cut(d['veces_capital'], [-1, 1, 3, 6, 1e9],
                               labels=['menos de 1x', '1-3x', '3-6x', 'mas de 6x'])
    d['grupo_orden'] = pd.cut(d['orden_del_dia'], [0, 1, 3, 6, 1e9],
                              labels=['1a del dia', '2a-3a', '4a-6a', '7a o mas'])
    d['franja'] = pd.cut(d['hora'], [-0.01, 8, 13, 16, 19, 24],
                         labels=['0-8', '8-13', '13-16', '16-19', '19-24'])
    return d


def resumen(d):
    g = d.loc[d['pnl_neto'] > 0, 'pnl_neto']; p = d.loc[d['pnl_neto'] < 0, 'pnl_neto']
    ganado = float(g.sum()) if len(g) else 0.0
    return dict(operaciones=len(d),
                aciertos_pct=round(float(d['gana'].mean() * 100), 1),
                resultado_total=round(float(d['pnl_neto'].sum()), 2),
                ganancia_media=round(float(g.mean()), 2) if len(g) else 0.0,
                perdida_media=round(float(p.mean()), 2) if len(p) else 0.0,
                cuantas_borra=round(abs(float(p.mean())) / float(g.mean()), 1) if len(g) and len(p) and g.mean() else None,
                mejor=round(float(d['pnl_neto'].max()), 2),
                peor=round(float(d['pnl_neto'].min()), 2),
                peor_pct_de_lo_ganado=round(abs(float(d['pnl_neto'].min())) / max(ganado, 1) * 100, 1),
                minutos_mediana=int(d['minutos'].median()) if d['minutos'].notna().any() else None,
                dias_operados=int(d['dia'].nunique()),
                operaciones_por_dia=round(len(d) / max(d['dia'].nunique(), 1), 1),
                costes_totales=round(float(d['comision'].sum() + d['swap'].sum()), 2))


def tabla(d, col, minimo=3):
    filas = []
    for k, g in d.groupby(col, observed=True):
        if len(g) < minimo: continue
        filas.append({col: str(k), 'operaciones': len(g),
                      'aciertos_pct': round(float(g['gana'].mean() * 100), 1),
                      'resultado': round(float(g['pnl_neto'].sum()), 2),
                      'media': round(float(g['pnl_neto'].mean()), 2),
                      'peor': round(float(g['pnl_neto'].min()), 2),
                      'minutos_mediana': int(g['minutos'].median()) if g['minutos'].notna().any() else None})
    if not filas: return pd.DataFrame()
    return pd.DataFrame(filas).sort_values('resultado', ascending=False)


def monte_carlo(pnls, capital, n=250, sims=4000, semilla=0):
    """Remuestrea SUS resultados reales: nada de suponer porcentajes."""
    x = np.asarray(pnls, float)
    if len(x) < 10: return None
    rng = np.random.default_rng(semilla)
    eq = capital + np.cumsum(rng.choice(x, (sims, n), replace=True), axis=1)
    eq = np.c_[np.full(sims, capital), eq]
    fin = eq[:, -1]; pico = np.maximum.accumulate(eq, axis=1)
    dd = ((eq - pico) / np.where(pico == 0, 1, pico)).min(axis=1)
    return dict(capital_inicial=round(float(capital), 2),
                final_mediano=round(float(np.median(fin)), 2),
                peor_5pct=round(float(np.percentile(fin, 5)), 2),
                mejor_5pct=round(float(np.percentile(fin, 95)), 2),
                prob_positivo_pct=round(float((fin > capital).mean() * 100), 1),
                prob_perder_mitad_pct=round(float((fin < capital / 2).mean() * 100), 1),
                prob_quiebra_pct=round(float((eq.min(axis=1) <= 0).mean() * 100), 1),
                caida_max_mediana_pct=round(float(np.median(dd) * 100), 1))


def comparar_sin_peores(d, cuantas=5):
    cap = d.attrs.get('capital', 1000.0)
    con = monte_carlo(d['pnl_neto'].values, cap)
    sin = monte_carlo(d.drop(d.nsmallest(cuantas, 'pnl_neto').index)['pnl_neto'].values, cap)
    return con, sin


def curva(d, cuantas=5):
    cap = d.attrs.get('capital', 1000.0)
    x = d.sort_values('cerrada')
    peores = x.nsmallest(cuantas, 'pnl_neto').index
    return (pd.DataFrame(dict(fecha=x['cerrada'], real=cap + x['pnl_neto'].cumsum())),
            pd.DataFrame(dict(fecha=x.drop(peores)['cerrada'],
                              sin_las_peores=cap + x.drop(peores)['pnl_neto'].cumsum())))


def reglas(d):
    """Las reglas que salen de SUS numeros, cada una con la cifra que la justifica."""
    r = []
    if len(d) < 20: return ['Hacen falta al menos 20 operaciones para sacar reglas fiables.']
    largas = d[d['minutos'] > HORAS_LARGA * 60]
    cortas = d[d['minutos'] <= HORAS_LARGA * 60]
    if len(largas) >= 3 and largas['pnl_neto'].sum() < 0:
        r.append('Cerrar antes de ' + str(int(HORAS_LARGA)) + ' horas: tus ' + str(len(largas)) +
                 ' operaciones mas largas suman ' + str(round(largas['pnl_neto'].sum(), 2)) +
                 ', frente a ' + str(round(cortas['pnl_neto'].sum(), 2)) + ' del resto.')
    peor = d.nsmallest(1, 'pnl_neto').iloc[0]
    ganado = d.loc[d['pnl_neto'] > 0, 'pnl_neto'].sum()
    r.append('Stop siempre: tu peor operacion (' + str(round(peor['pnl_neto'], 2)) + ' en ' + peor['simbolo'] +
             ') se llevo el ' + str(round(abs(peor['pnl_neto']) / max(ganado, 1) * 100, 1)) +
             '% de todo lo que ganaste.')
    por_sim = d.groupby('simbolo')['pnl_neto'].sum().sort_values()
    if len(por_sim) and por_sim.iloc[0] < 0:
        r.append('Revisar ' + por_sim.index[0] + ': suma ' + str(round(por_sim.iloc[0], 2)) +
                 ' en ' + str(int((d['simbolo'] == por_sim.index[0]).sum())) + ' operaciones.')
    por_franja = d.groupby('franja', observed=True)['pnl_neto'].sum()
    malas = [str(k) for k, v in por_franja.items() if v < 0]
    if malas:
        r.append('Evitar las franjas ' + ', '.join(malas) + ': son las que suman negativo.')
    tam = d.groupby('grupo_tamano', observed=True)['pnl_neto'].sum()
    grandes = [str(k) for k, v in tam.items() if v < 0 and k in ('3-6x', 'mas de 6x')]
    if grandes:
        r.append('Bajar el tamano: las operaciones de ' + ', '.join(grandes) +
                 ' tu capital suman ' + str(round(sum(tam[k] for k in grandes), 2)) + '.')
    tras = d.groupby('tras_perdida')['pnl_neto'].mean()
    if 1 in tras.index and 0 in tras.index and tras[1] < tras[0]:
        r.append('Parar tras una perdida: las operaciones que vienen despues promedian ' +
                 str(round(tras[1], 2)) + ' frente a ' + str(round(tras[0], 2)) + ' del resto.')
    return r


def diagnostico_completo(d, capital=None, carpeta_ticks=None):
    """Todo de una vez, listo para el informe.

    Si se le pasa una carpeta con ticks, anade el analisis de coste de ejecucion.
    """
    d = preparar(d, capital)
    con, sin = comparar_sin_peores(d)
    ej = None
    if carpeta_ticks:
        try:
            import ejecucion as EJ
            res = EJ.analizar(d, carpeta_ticks)
            if res and res.get('ok'):
                ej = dict(resumen=EJ.resumen(res), veredicto=EJ.veredicto(res),
                          por_instrumento=EJ.por_instrumento(res), por_hora=EJ.por_hora(res),
                          limitada=EJ.limitada_vs_mercado(res))
        except Exception:
            ej = None
    return dict(datos=d, resumen=resumen(d), reglas=reglas(d), ejecucion=ej,
                por_duracion=tabla(d, 'grupo_minutos'), por_tamano=tabla(d, 'grupo_tamano'),
                por_orden=tabla(d, 'grupo_orden'), por_franja=tabla(d, 'franja'),
                por_instrumento=tabla(d, 'simbolo'), montecarlo_con=con, montecarlo_sin=sin)
