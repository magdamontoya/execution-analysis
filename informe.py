# -*- coding: utf-8 -*-
# ================================================================
# INFORME — convierte el diagnostico en algo que se lee y se manda
#
#   texto(dx)   version para la terminal
#   html(dx)    version para la web y para guardar como PDF
# ================================================================

import pandas as pd


def _usd(v):
    if v is None: return '-'
    return ('+' if v >= 0 else '') + str(round(v, 2)) + ' USD'


def _bloque_ejecucion_texto(dx):
    ej = dx.get('ejecucion')
    if not ej or not ej.get('veredicto'): return []
    L = ['\n' + '-' * 78, 'LO QUE TE CUESTA EJECUTAR (lo que ningun diario mide)', '-' * 78]
    for x in ej['veredicto']: L.append('  - ' + x)
    if ej.get('por_instrumento') is not None and len(ej['por_instrumento']):
        L.append('')
        L.append(ej['por_instrumento'].to_string(index=False))
    return L


def texto(dx):
    r = dx['resumen']; con = dx['montecarlo_con']; sin = dx['montecarlo_sin']
    L = []
    L.append('=' * 78)
    L.append('DIAGNOSTICO DE TU OPERATIVA')
    L.append('=' * 78)
    L.append('  operaciones          ' + str(r['operaciones']) + '  en ' + str(r['dias_operados']) +
             ' dias (' + str(r['operaciones_por_dia']) + ' al dia)')
    L.append('  aciertos             ' + str(r['aciertos_pct']) + '%')
    L.append('  resultado total      ' + _usd(r['resultado_total']))
    L.append('  ganancia media       ' + _usd(r['ganancia_media']) + '   |  perdida media ' + _usd(r['perdida_media']))
    if r['cuantas_borra']:
        L.append('  cada perdida borra   ' + str(r['cuantas_borra']) + ' operaciones ganadoras')
    L.append('  mejor / peor         ' + _usd(r['mejor']) + ' / ' + _usd(r['peor']))
    L.append('  tu peor operacion    se llevo el ' + str(r['peor_pct_de_lo_ganado']) + '% de todo lo que ganaste')
    L.append('  duracion mediana     ' + str(r['minutos_mediana']) + ' minutos')
    L.append('  costes               ' + _usd(r['costes_totales']))

    L.append('\n' + '-' * 78)
    L.append('LAS REGLAS QUE SALEN DE TUS PROPIOS NUMEROS')
    L.append('-' * 78)
    for x in dx['reglas']: L.append('  - ' + x)

    for titulo, clave in [('POR DURACION', 'por_duracion'), ('POR TAMANO', 'por_tamano'),
                          ('POR ORDEN DEL DIA', 'por_orden'), ('POR FRANJA HORARIA', 'por_franja'),
                          ('POR INSTRUMENTO', 'por_instrumento')]:
        t = dx[clave]
        if t is None or len(t) == 0: continue
        L.append('\n' + '-' * 78); L.append(titulo); L.append('-' * 78)
        L.append(t.to_string(index=False))

    L += _bloque_ejecucion_texto(dx)

    if con and sin:
        L.append('\n' + '-' * 78)
        L.append('SI REPITIERAS 250 OPERACIONES COMO ESTAS')
        L.append('-' * 78)
        comp = pd.DataFrame([con, sin], index=['con todas', 'sin las 5 peores']).T
        L.append(comp.to_string())
        if con['prob_quiebra_pct'] > 10:
            L.append('\n  AVISO: repitiendo esta operativa, la probabilidad de quedarte sin cuenta es del ' +
                     str(con['prob_quiebra_pct']) + '%. Sin las cinco peores operaciones baja al ' +
                     str(sin['prob_quiebra_pct']) + '%.')
    L.append('\n' + '=' * 78)
    return '\n'.join(L)


def html(dx, titulo='Diagnostico de tu operativa'):
    r = dx['resumen']; con = dx['montecarlo_con']; sin = dx['montecarlo_sin']
    est = """<style>
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#111}
    h1{font-size:26px;margin-bottom:4px} h2{font-size:18px;margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
    .k{display:inline-block;min-width:230px;color:#555} .v{font-weight:600}
    table{border-collapse:collapse;width:100%;margin-top:10px;font-size:14px}
    th,td{border-bottom:1px solid #eee;padding:7px 9px;text-align:right} th{text-align:right;background:#fafafa}
    td:first-child,th:first-child{text-align:left}
    .pos{color:#16a34a} .neg{color:#dc2626}
    .aviso{background:#fef2f2;border-left:4px solid #dc2626;padding:12px 16px;margin-top:16px}
    .regla{background:#f8fafc;border-left:4px solid #2563eb;padding:10px 14px;margin:8px 0}
    </style>"""

    def tt(df):
        if df is None or len(df) == 0: return '<p>Sin datos suficientes.</p>'
        h = '<table><tr>' + ''.join('<th>' + str(c) + '</th>' for c in df.columns) + '</tr>'
        for _, row in df.iterrows():
            h += '<tr>' + ''.join(
                '<td class="' + ('pos' if isinstance(v, (int, float)) and v > 0 and c in ('resultado', 'media') else
                                 ('neg' if isinstance(v, (int, float)) and v < 0 and c in ('resultado', 'media', 'peor') else '')) +
                '">' + str(v) + '</td>' for c, v in zip(df.columns, row)) + '</tr>'
        return h + '</table>'

    p = ['<html><head><meta charset="utf-8"><title>' + titulo + '</title>' + est + '</head><body>']
    p.append('<h1>' + titulo + '</h1>')
    p.append('<p>' + str(r['operaciones']) + ' operaciones en ' + str(r['dias_operados']) + ' dias.</p>')
    p.append('<h2>Resumen</h2>')
    for k, v in [('Aciertos', str(r['aciertos_pct']) + '%'), ('Resultado total', _usd(r['resultado_total'])),
                 ('Ganancia media', _usd(r['ganancia_media'])), ('Perdida media', _usd(r['perdida_media'])),
                 ('Cada perdida borra', str(r['cuantas_borra']) + ' ganadoras' if r['cuantas_borra'] else '-'),
                 ('Peor operacion', _usd(r['peor']) + ' (' + str(r['peor_pct_de_lo_ganado']) + '% de lo ganado)'),
                 ('Duracion mediana', str(r['minutos_mediana']) + ' minutos'),
                 ('Costes', _usd(r['costes_totales']))]:
        p.append('<div><span class="k">' + k + '</span><span class="v">' + str(v) + '</span></div>')

    p.append('<h2>Las reglas que salen de tus numeros</h2>')
    for x in dx['reglas']: p.append('<div class="regla">' + x + '</div>')

    for titulo_s, clave in [('Por duracion', 'por_duracion'), ('Por tamano', 'por_tamano'),
                            ('Por orden del dia', 'por_orden'), ('Por franja horaria', 'por_franja'),
                            ('Por instrumento', 'por_instrumento')]:
        p.append('<h2>' + titulo_s + '</h2>' + tt(dx[clave]))

    ej = dx.get('ejecucion')
    if ej and ej.get('veredicto'):
        p.append('<h2>Lo que te cuesta ejecutar</h2>')
        for x in ej['veredicto']:
            p.append('<div class="regla">' + x + '</div>')
        if ej.get('por_instrumento') is not None and len(ej['por_instrumento']):
            p.append(tt(ej['por_instrumento']))

    if con and sin:
        p.append('<h2>Si repitieras 250 operaciones como estas</h2>')
        comp = pd.DataFrame([con, sin], index=['con todas', 'sin las 5 peores']).T.reset_index()
        comp.columns = ['metrica', 'con todas', 'sin las 5 peores']
        p.append(tt(comp))
        if con['prob_quiebra_pct'] > 10:
            p.append('<div class="aviso">Repitiendo esta operativa, la probabilidad de quedarte sin cuenta es del <b>' +
                     str(con['prob_quiebra_pct']) + '%</b>. Sin las cinco peores operaciones baja al <b>' +
                     str(sin['prob_quiebra_pct']) + '%</b>.</div>')
    p.append('</body></html>')
    return '\n'.join(p)
