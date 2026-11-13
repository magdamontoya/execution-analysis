# -*- coding: utf-8 -*-
"""
exportar_ticks.py - Saca ticks de MT5 a un archivo para analizar donde sea

Corre en el VPS, con MetaTrader 5 abierto:

    python exportar_ticks.py                  # 60 dias de SpotBrent y XAUUSD
    python exportar_ticks.py --dias 90
    python exportar_ticks.py --simbolo SpotBrent --dias 30

Produce  ticks_<SIMBOLO>_<dias>d.parquet  en la carpeta actual.

POR QUE TICKS Y NO BARRAS M1
  Tu script de giros corre sobre ticks: umbral 0.25 sobre 12h le da ~134
  giros. El mismo umbral sobre barras M1 da 2. Son dos cosas distintas,
  asi que para testear TU senal hay que usar TUS datos.

POR QUE UN ARCHIVO Y NO LEER MT5 DIRECTO
  MT5 solo existe en Windows. Con el archivo, el backtest corre igual en
  el VPS, en Google Colab o en cualquier portatil. Y lo mas importante:
  el dato queda CONGELADO, asi que dos ejecuciones del mismo backtest dan
  el mismo resultado. Leyendo en vivo nunca se puede reproducir nada.

Guarda solo lo necesario (time_msc, bid, ask) en float32 para que el
archivo sea manejable: ~60 dias de Brent caben en unas decenas de MB.
"""

import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd

DIAS = 60
SIMBOLOS = ["SpotBrent", "XAUUSD"]


def conectar(mt5):
    if mt5.initialize():
        return True
    rutas = glob.glob(r"C:\Program Files\*\terminal64.exe")
    rutas += glob.glob(os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe"))
    for ruta in rutas:
        if mt5.initialize(path=ruta):
            return True
    return False


def resolver(mt5, pedido):
    nombres = [s.name for s in (mt5.symbols_get() or [])]
    if pedido in nombres:
        return pedido
    p = pedido.upper()
    for n in nombres:
        if n.upper() == p:
            return n
    for n in nombres:
        if n.upper().startswith(p):
            return n
    return None


def bajar(mt5, simbolo, dias):
    """
    Dia a dia, no de una. Pedir 60 dias en una sola llamada hace que MT5
    devuelva None sin decir por que; troceando se ve exactamente que dia
    falla y cuanto trae cada uno.
    """
    fin = datetime.datetime.now()
    trozos = []
    vacios = 0
    for d in range(dias, 0, -1):
        a = fin - datetime.timedelta(days=d)
        b = fin - datetime.timedelta(days=d - 1)
        t = mt5.copy_ticks_range(simbolo, a, b, mt5.COPY_TICKS_ALL)
        if t is None or len(t) == 0:
            vacios += 1
            continue
        df = pd.DataFrame(t)[["time_msc", "bid", "ask"]]
        trozos.append(df)
        if d % 10 == 0:
            print("   ... faltan %d dias (%s: %d ticks)" % (d, a.date(), len(df)))
    if not trozos:
        return None, vacios
    out = pd.concat(trozos, ignore_index=True)
    out = out[(out["bid"] > 0) & (out["ask"] > 0)]
    out["bid"] = out["bid"].astype("float32")
    out["ask"] = out["ask"].astype("float32")
    out["time_msc"] = out["time_msc"].astype("int64")
    return out.sort_values("time_msc").reset_index(drop=True), vacios


def main():
    import MetaTrader5 as mt5

    dias = DIAS
    simbolos = SIMBOLOS
    if "--dias" in sys.argv:
        dias = int(sys.argv[sys.argv.index("--dias") + 1])
    if "--simbolo" in sys.argv:
        simbolos = [sys.argv[sys.argv.index("--simbolo") + 1]]

    if not conectar(mt5):
        raise SystemExit("No conecta a MT5. Abre MetaTrader 5 y repite.")

    print("exportando %d dias de: %s" % (dias, ", ".join(simbolos)))
    for pedido in simbolos:
        sim = resolver(mt5, pedido)
        if sim is None:
            print("\n%s: no esta en tu broker" % pedido)
            continue
        print("\n%s  ->  bajando..." % sim)
        df, vacios = bajar(mt5, sim, dias)
        if df is None or df.empty:
            print("   sin datos (%d dias vacios)" % vacios)
            continue

        ruta = "ticks_%s_%dd.parquet" % (sim, dias)
        try:
            df.to_parquet(ruta, index=False, compression="snappy")
        except Exception as e:
            ruta = "ticks_%s_%dd.csv.gz" % (sim, dias)
            df.to_csv(ruta, index=False, compression="gzip")
            print("   (parquet no disponible: %s -> guardado como csv.gz)" % type(e).__name__)

        mb = os.path.getsize(ruta) / 1e6
        t0 = pd.to_datetime(df["time_msc"].iloc[0], unit="ms")
        t1 = pd.to_datetime(df["time_msc"].iloc[-1], unit="ms")
        dias_reales = max((t1 - t0).days, 1)
        spread = (df["ask"] - df["bid"]).median()
        print("   %s" % os.path.abspath(ruta))
        print("   %s ticks | %.1f MB | %s a %s" % (
            format(len(df), ","), mb, t0.date(), t1.date()))
        print("   %s ticks/dia de media | spread mediano %.5f" % (
            format(int(len(df) / dias_reales), ","), spread))
        if vacios:
            print("   (%d dias sin datos: fines de semana y festivos)" % vacios)

    mt5.shutdown()
    print("\nListo. Mandame las lineas de resumen de cada simbolo.")


if __name__ == "__main__":
    main()
