# -*- coding: utf-8 -*-
# ================================================================
# LECTOR — convierte el historial de cualquier broker al formato unico
#
# Acepta:
#   - Informe de MetaTrader 5 en HTML  (Historial -> Informe)
#   - Informe de MetaTrader 4 en HTM   (Cuenta -> Informe detallado)
#   - CSV exportado de MT5 o del dashboard
#   - Excel (.xlsx) con las mismas columnas
#
# Devuelve SIEMPRE el mismo DataFrame, con estas columnas:
#   abierta, cerrada, simbolo, tipo, lotes, entrada, salida,
#   sl, tp, comision, swap, pnl, minutos
#
# Uso:
#   from lector import leer
#   ops = leer('informe.html')
# ================================================================

import io
import os
import re

import numpy as np
import pandas as pd

COLUMNAS = ['abierta', 'cerrada', 'simbolo', 'tipo', 'lotes', 'entrada', 'salida',
            'sl', 'tp', 'comision', 'swap', 'pnl', 'minutos']

# como se llama cada columna en cada idioma y version
ALIAS = {
    'abierta': ['open time', 'time', 'hora de apertura', 'apertura', 'fecha apertura', 'open_time', 'opentime'],
    'cerrada': ['close time', 'hora de cierre', 'cierre', 'fecha cierre', 'close_time', 'closetime'],
    'simbolo': ['symbol', 'simbolo', 'símbolo', 'instrumento', 'item'],
    'tipo': ['type', 'tipo', 'direccion', 'dirección'],
    'lotes': ['volume', 'volumen', 'lots', 'lotes', 'size', 'tamano', 'tamaño'],
    'entrada': ['price', 'precio', 'open price', 'precio de apertura', 'entry', 'entrada'],
    'salida': ['close price', 'precio de cierre', 'exit', 'salida'],
    'sl': ['s / l', 's/l', 'sl', 'stop loss', 'stop'],
    'tp': ['t / p', 't/p', 'tp', 'take profit', 'objetivo'],
    'comision': ['commission', 'comision', 'comisión', 'fee'],
    'swap': ['swap', 'rollover', 'financiacion', 'financiación'],
    'pnl': ['profit', 'beneficio', 'ganancia', 'resultado', 'p&l', 'pnl', 'net p&l', 'utilidad'],
}


def _norm(x):
    x = str(x).strip().lower()
    x = x.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', x)


def _a_numero(s):
    """Convierte texto a numero aguantando 1.234,56 / 1,234.56 / espacios / parentesis."""
    if isinstance(s, (int, float, np.number)):
        return float(s)
    t = str(s).strip().replace('\xa0', '').replace(' ', '')
    if t in ('', '-', '—', 'nan', 'none'): return np.nan
    neg = t.startswith('(') and t.endswith(')')
    t = t.strip('()')
    if ',' in t and '.' in t:
        t = t.replace('.', '').replace(',', '.') if t.rfind(',') > t.rfind('.') else t.replace(',', '')
    elif ',' in t:
        t = t.replace(',', '.') if len(t.split(',')[-1]) <= 2 else t.replace(',', '')
    try:
        v = float(re.sub(r'[^0-9.\-]', '', t))
    except ValueError:
        return np.nan
    return -v if neg else v


def _mapear(df):
    """Renombra las columnas del archivo a los nombres estandar."""
    cols = {_norm(c): c for c in df.columns}
    ren = {}
    usadas = set()
    for destino, nombres in ALIAS.items():
        for n in nombres:
            for c_norm, c_real in cols.items():
                if c_real in usadas: continue
                if c_norm == n or c_norm.startswith(n + ' ') or c_norm.endswith(' ' + n):
                    ren[c_real] = destino; usadas.add(c_real); break
            if destino in ren.values(): break
    return df.rename(columns=ren)


def _desde_html(ruta_o_texto):
    """Informes de MetaTrader 4 y 5 en HTML: busca la tabla que tenga columnas de operaciones."""
    try:
        tablas = pd.read_html(ruta_o_texto, header=0)
    except Exception:
        tablas = pd.read_html(io.StringIO(ruta_o_texto if isinstance(ruta_o_texto, str) else ''), header=0)
    mejor, puntos_mejor = None, 0
    for t in tablas:
        if len(t) < 2: continue
        t = t.dropna(axis=1, how='all')
        # MT5 mete la cabecera dentro del cuerpo: si las columnas son numeros, busca la fila cabecera
        if all(str(c).isdigit() or str(c).startswith('Unnamed') for c in t.columns):
            for i in range(min(8, len(t))):
                fila = [_norm(x) for x in t.iloc[i].tolist()]
                if any('symbol' in x or 'simbolo' in x or 'símbolo' in x for x in fila):
                    t = t.iloc[i + 1:].reset_index(drop=True)
                    t.columns = [str(x) for x in fila]
                    break
        m = _mapear(t)
        puntos = sum(1 for c in ('simbolo', 'tipo', 'lotes', 'pnl') if c in m.columns)
        if puntos > puntos_mejor:
            mejor, puntos_mejor = m, puntos
    if mejor is None or puntos_mejor < 3:
        raise ValueError('No se encontro una tabla de operaciones en el archivo HTML.')
    return mejor


def _desde_tabla(ruta):
    ext = os.path.splitext(str(ruta))[1].lower()
    if ext in ('.xlsx', '.xls'):
        return _mapear(pd.read_excel(ruta))
    for sep in (None, ';', '\t', ','):
        try:
            d = pd.read_csv(ruta, sep=sep, engine='python')
            if d.shape[1] >= 4:
                return _mapear(d)
        except Exception:
            continue
    raise ValueError('No se pudo leer el archivo como tabla.')


def leer(ruta, moneda_cuenta='USD'):
    """Lee cualquier historial y devuelve el DataFrame estandar."""
    ext = os.path.splitext(str(ruta))[1].lower()
    if ext in ('.html', '.htm'):
        d = _desde_html(ruta)
    else:
        d = _desde_tabla(ruta)

    for c in COLUMNAS:
        if c not in d.columns: d[c] = np.nan
    d = d[COLUMNAS].copy()

    # limpiar
    for c in ('lotes', 'entrada', 'salida', 'sl', 'tp', 'comision', 'swap', 'pnl'):
        d[c] = d[c].map(_a_numero)
    for c in ('abierta', 'cerrada'):
        d[c] = pd.to_datetime(d[c], errors='coerce', dayfirst=False)
    d['simbolo'] = d['simbolo'].astype(str).str.strip().str.upper()
    d['tipo'] = d['tipo'].astype(str).str.strip().str.lower().map(
        lambda x: 'COMPRA' if x.startswith(('buy', 'compra', 'long')) else
        ('VENTA' if x.startswith(('sell', 'venta', 'short')) else np.nan))

    # quitar lo que no son operaciones (depositos, resumenes, filas vacias)
    d = d[d['tipo'].notna() & d['simbolo'].notna() & (d['simbolo'] != 'NAN')]
    d = d[d['lotes'].notna() & (d['lotes'] > 0)]
    d = d[d['pnl'].notna()]
    d = d[d['simbolo'].str.len().between(3, 20)]

    # resultado neto: si el informe separa comision y swap, se suman
    d['comision'] = d['comision'].fillna(0.0)
    d['swap'] = d['swap'].fillna(0.0)
    d['pnl_neto'] = d['pnl'] + d['comision'] + d['swap']

    # duracion
    d['minutos'] = (d['cerrada'] - d['abierta']).dt.total_seconds() / 60
    d.loc[d['minutos'] < 0, 'minutos'] = np.nan

    d = d.sort_values('abierta').reset_index(drop=True)
    d.attrs['moneda'] = moneda_cuenta
    return d


def resumen_lectura(d):
    """Lo que hay que enseñar al usuario justo despues de subir el archivo."""
    if len(d) == 0:
        return {'operaciones': 0, 'aviso': 'No se encontraron operaciones en el archivo.'}
    return dict(operaciones=len(d),
                desde=str(d['abierta'].min())[:10], hasta=str(d['cerrada'].max())[:10],
                instrumentos=int(d['simbolo'].nunique()),
                lista_instrumentos=', '.join(sorted(d['simbolo'].unique())[:10]),
                resultado_total=round(float(d['pnl_neto'].sum()), 2),
                con_duracion=int(d['minutos'].notna().sum()),
                sin_duracion=int(d['minutos'].isna().sum()),
                comisiones=round(float(d['comision'].sum()), 2),
                swaps=round(float(d['swap'].sum()), 2))


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Uso: python lector.py <archivo>'); sys.exit(1)
    ops = leer(sys.argv[1])
    print('=' * 70)
    for k, v in resumen_lectura(ops).items():
        print('  ' + k.ljust(20) + str(v))
    print('=' * 70)
    print(ops.head(10).to_string(index=False))
