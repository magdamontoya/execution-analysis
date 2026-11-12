# -*- coding: utf-8 -*-
"""
swap_posiciones.py - Que te ha costado (o pagado) mantener cada posicion

Corre:  python swap_posiciones.py
Requiere MetaTrader 5 abierto.

Para cada posicion abierta muestra:
  - fecha de apertura y dias que lleva abierta
  - swap acumulado hasta hoy
  - la tasa REAL que has pagado (swap acumulado / dias) -- empirica
  - breakeven incluyendo el swap
Y proyecta lo que costara mantenerla 7, 30 y 90 dias mas.

La tasa empirica es la que manda: sale de tu propio dinero, no de una
conversion de unidades que puede estar mal.
"""

import os
import glob
import datetime

SIMBOLO = "SpotBrent"
DIAS_SEM = {0: "domingo", 1: "lunes", 2: "martes", 3: "miercoles",
            4: "jueves", 5: "viernes", 6: "sabado"}


def conectar(mt5):
    if mt5.initialize():
        return True
    rutas = glob.glob(r"C:\Program Files\*\terminal64.exe")
    rutas += glob.glob(os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe"))
    for ruta in rutas:
        if mt5.initialize(path=ruta):
            return True
    return False


def main():
    import MetaTrader5 as mt5

    if not conectar(mt5):
        raise SystemExit("No conecta a MT5. Abre MetaTrader 5 y repite.")

    info = mt5.symbol_info(SIMBOLO)
    tick = mt5.symbol_info_tick(SIMBOLO)
    if info is None or tick is None:
        raise SystemExit("No encuentro %s" % SIMBOLO)

    precio = (tick.bid + tick.ask) / 2.0
    tam = info.trade_contract_size              # barriles por lote
    unidad = info.point * tam                   # 1 punto de swap -> dinero por lote
    tasa_largo = info.swap_long * unidad        # por lote y noche
    tasa_corto = info.swap_short * unidad
    d3 = getattr(info, "swap_rollover3days", None)
    ahora = datetime.datetime.now()
    posiciones = mt5.positions_get(symbol=SIMBOLO) or []
    if not posiciones:
        print("Sin posiciones abiertas en %s" % SIMBOLO)
        mt5.shutdown()
        return

    print()
    print("=" * 94)
    print("POSICIONES %s    precio %.3f    %s" % (
        SIMBOLO, precio, ahora.strftime("%d-%m-%Y %H:%M")))
    print("=" * 94)
    print("  tasa del broker ahora:  largo %+.2f/lote/noche   corto %+.2f/lote/noche"
          % (tasa_largo, tasa_corto))
    print("  (campo crudo: largo %+.2f  corto %+.2f puntos | 1 punto = %.5f x %s barriles)"
          % (info.swap_long, info.swap_short, info.point, int(tam)))
    if d3 is not None and d3 in DIAS_SEM:
        print("  swap TRIPLE los %s  -> 4 dias normales + 1 triple = 7 noches por semana," % DIAS_SEM[d3])
        print("     asi que multiplicar por dias naturales sale correcto.")
    print()

    cab = "  %-5s %6s %9s %12s %6s %10s %10s %10s %10s"
    print(cab % ("tipo", "lotes", "entrada", "abierta", "dias",
                 "flotante", "swap", "swap/dia", "breakeven"))
    print("  " + "-" * 90)

    grupos = {"BUY": [], "SELL": []}
    for p in posiciones:
        tipo = "SELL" if p.type == 1 else "BUY"
        abierta = datetime.datetime.fromtimestamp(p.time)
        dias = max((ahora - abierta).days, 0)
        valor_dolar = p.volume * tam          # dinero por cada $1 de movimiento
        # breakeven incluyendo el swap ya pagado/cobrado
        if tipo == "BUY":
            be = p.price_open - p.swap / valor_dolar
        else:
            be = p.price_open + p.swap / valor_dolar
        por_dia = (p.swap / dias) if dias > 0 else 0.0
        print(cab % (tipo, "%.2f" % p.volume, "%.3f" % p.price_open,
                     abierta.strftime("%d-%m-%Y"), str(dias),
                     "%+.2f" % p.profit, "%+.2f" % p.swap,
                     "%+.2f" % por_dia, "%.3f" % be))
        grupos[tipo].append({"vol": p.volume, "swap": p.swap, "dias": dias,
                             "flot": p.profit, "entrada": p.price_open,
                             "vd": valor_dolar})

    print()
    for tipo, etiqueta, tasa in (("BUY", "LARGOS", tasa_largo),
                                 ("SELL", "CORTOS", tasa_corto)):
        g = grupos[tipo]
        if not g:
            continue
        lotes = sum(x["vol"] for x in g)
        swap_ac = sum(x["swap"] for x in g)
        flot = sum(x["flot"] for x in g)
        dias_pond = (sum(x["dias"] * x["vol"] for x in g) / lotes) if lotes else 0
        emp = (swap_ac / dias_pond / lotes) if dias_pond and lotes else 0.0
        por_noche = tasa * lotes

        print("=" * 94)
        print("%s  -  %.2f lotes" % (etiqueta, lotes))
        print("=" * 94)
        print("  swap acumulado desde la apertura   %+10.2f" % swap_ac)
        print("  dias abiertos (ponderado)          %10.1f" % dias_pond)
        print("  tasa REAL pagada  (empirica)       %+10.2f  por lote y noche" % emp)
        print("  tasa del broker hoy                %+10.2f  por lote y noche" % tasa)
        if abs(tasa) > 1e-9:
            dif = 100.0 * (emp - tasa) / abs(tasa)
            print("  diferencia                         %+9.1f%%  %s" % (
                dif,
                "(la tasa cambio desde que abriste)" if abs(dif) > 15 else "(coherente)"))
        print()
        print("  flotante de precio (sin swap)      %+10.2f" % (flot - swap_ac))
        print("  swap                               %+10.2f" % swap_ac)
        print("  flotante total                     %+10.2f" % flot)
        print()
        print("  si no tocas nada, a la tasa de hoy:  %+.2f por noche" % por_noche)
        for n in (7, 30, 90):
            print("     %3d dias mas -> %+10.2f   (precio de breakeven se mueve %+.3f)" % (
                n, por_noche * n,
                (-por_noche * n / (lotes * tam)) if tipo == "BUY"
                else (por_noche * n / (lotes * tam))))
        print()

    # ---- resumen del libro
    lotes_l = sum(x["vol"] for x in grupos["BUY"])
    lotes_c = sum(x["vol"] for x in grupos["SELL"])
    neto = lotes_l - lotes_c
    noche = tasa_largo * lotes_l + tasa_corto * lotes_c
    print("=" * 94)
    print("LIBRO COMPLETO")
    print("=" * 94)
    print("  neto %+.2f lotes  ->  %.0f USD por cada dolar de movimiento" % (
        neto, abs(neto) * tam))
    print("  swap acumulado total  %+.2f" % sum(p.swap for p in posiciones))
    print("  swap por noche a la tasa de hoy  %+.2f" % noche)
    print("  al mes (30 noches)               %+.2f" % (noche * 30))
    ac = mt5.account_info()
    if ac and ac.equity > 0:
        print("  eso es el %.1f%% de tu equity ($%.2f) al mes" % (
            abs(noche) * 30 / ac.equity * 100, ac.equity))
    print()

    mt5.shutdown()


if __name__ == "__main__":
    main()
