# -*- coding: utf-8 -*-
"""
swap_posiciones.py - Que te ha costado (o pagado) mantener cada posicion

Corre:  python swap_posiciones.py              todas las posiciones abiertas
        python swap_posiciones.py SpotBrent    solo ese instrumento

Requiere MetaTrader 5 abierto.

Para cada posicion abierta muestra:
  - fecha de apertura y dias que lleva abierta
  - swap acumulado hasta hoy
  - la tasa REAL que has pagado (swap acumulado / dias) -- empirica
  - breakeven incluyendo el swap
Y proyecta lo que costara mantenerla 7, 30 y 90 dias mas.

LA TASA EMPIRICA ES LA QUE MANDA
  Sale de tu propio dinero, no de una conversion de unidades que puede
  estar mal. Si la empirica y la del broker no coinciden, o la tasa
  cambio desde que abriste, o la conversion de puntos a dinero esta mal.
  En ambos casos conviene saberlo.

OJO CON LAS POSICIONES DE HOY
  Una posicion abierta hoy puede traer swap y llevar 0 dias. Metida en
  la media, infla la tasa empirica hasta valores absurdos. Aqui queda
  fuera del calculo de la tasa (y se avisa), pero su swap si cuenta en
  los totales, porque es dinero real.
"""

import os
import sys
import glob
import datetime

DIAS_SEM = {0: "domingo", 1: "lunes", 2: "martes", 3: "miercoles",
            4: "jueves", 5: "viernes", 6: "sabado"}


def conectar(mt5):
    if mt5.initialize():
        return True
    rutas = glob.glob(r"C:\Program Files\*\terminal64.exe")
    rutas += glob.glob(os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe"))
    for ruta in rutas:
        if mt5.initialize(path=ruta):
            print("conectado via %s" % ruta)
            return True
    return False


def analizar_simbolo(mt5, simbolo, posiciones, ahora):
    """Imprime el bloque de un instrumento. Devuelve sus totales."""
    info = mt5.symbol_info(simbolo)
    tick = mt5.symbol_info_tick(simbolo)
    if info is None:
        print("\n%s: no encuentro el simbolo" % simbolo)
        return None

    precio = (tick.bid + tick.ask) / 2.0 if tick else float("nan")
    tam = info.trade_contract_size               # unidades por lote
    unidad = info.point * tam                    # 1 punto de swap -> dinero por lote
    tasa_largo = info.swap_long * unidad         # por lote y noche
    tasa_corto = info.swap_short * unidad
    d3 = getattr(info, "swap_rollover3days", None)

    print()
    print("=" * 94)
    print("%s    precio %.5f    %d posicion(es)" % (
        simbolo, precio, len(posiciones)))
    print("=" * 94)
    print("  tasa del broker ahora:  largo %+.2f/lote/noche   corto %+.2f/lote/noche"
          % (tasa_largo, tasa_corto))
    print("  (campo crudo: largo %+.2f  corto %+.2f puntos | 1 punto = %.5f x %s unidades)"
          % (info.swap_long, info.swap_short, info.point, int(tam)))
    if d3 is not None and d3 in DIAS_SEM:
        print("  swap TRIPLE los %s  -> 4 noches normales + 1 triple = 7 por semana,"
              % DIAS_SEM[d3])
        print("     asi que multiplicar por dias naturales sale correcto.")
    print()

    cab = "  %-5s %6s %12s %12s %6s %10s %10s %10s %12s"
    print(cab % ("tipo", "lotes", "entrada", "abierta", "dias",
                 "flotante", "swap", "swap/dia", "breakeven"))
    print("  " + "-" * 90)

    grupos = {"BUY": [], "SELL": []}
    recien = 0
    for p in posiciones:
        tipo = "SELL" if p.type == 1 else "BUY"
        abierta = datetime.datetime.fromtimestamp(p.time)
        dias = max((ahora - abierta).days, 0)
        valor_dolar = p.volume * tam             # dinero por cada 1.0 de movimiento
        if tipo == "BUY":
            be = p.price_open - p.swap / valor_dolar
        else:
            be = p.price_open + p.swap / valor_dolar
        por_dia = (p.swap / dias) if dias > 0 else float("nan")
        if dias == 0:
            recien += 1
        print(cab % (tipo, "%.2f" % p.volume, "%.5f" % p.price_open,
                     abierta.strftime("%d-%m-%Y"), str(dias),
                     "%+.2f" % p.profit, "%+.2f" % p.swap,
                     ("%+.2f" % por_dia) if dias > 0 else "-",
                     "%.5f" % be))
        grupos[tipo].append({"vol": p.volume, "swap": p.swap, "dias": dias,
                             "flot": p.profit, "entrada": p.price_open})

    if recien:
        print()
        print("  (%d posicion(es) abiertas hoy: fuera del calculo de la tasa empirica,"
              % recien)
        print("   porque llevan 0 dias. Su swap si cuenta en los totales.)")

    for tipo, etiqueta, tasa in (("BUY", "LARGOS", tasa_largo),
                                 ("SELL", "CORTOS", tasa_corto)):
        g = grupos[tipo]
        if not g:
            continue
        lotes = sum(x["vol"] for x in g)
        swap_ac = sum(x["swap"] for x in g)
        flot = sum(x["flot"] for x in g)
        # la tasa empirica solo con las que llevan al menos un dia
        viejas = [x for x in g if x["dias"] > 0]
        lotes_v = sum(x["vol"] for x in viejas)
        swap_v = sum(x["swap"] for x in viejas)
        dias_pond = (sum(x["dias"] * x["vol"] for x in viejas) / lotes_v) if lotes_v else 0
        emp = (swap_v / dias_pond / lotes_v) if dias_pond and lotes_v else float("nan")
        por_noche = tasa * lotes

        print()
        print("-" * 94)
        print("%s %s  -  %.2f lotes" % (simbolo, etiqueta, lotes))
        print("-" * 94)
        print("  swap acumulado desde la apertura   %+10.2f" % swap_ac)
        if dias_pond:
            print("  dias abiertos (ponderado)          %10.1f" % dias_pond)
            print("  tasa REAL pagada  (empirica)       %+10.2f  por lote y noche" % emp)
        else:
            print("  (todas abiertas hoy: aun no hay tasa empirica)")
        print("  tasa del broker hoy                %+10.2f  por lote y noche" % tasa)
        if dias_pond and abs(tasa) > 1e-9:
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
            mov_be = (-por_noche * n / (lotes * tam)) if tipo == "BUY" \
                else (por_noche * n / (lotes * tam))
            print("     %3d dias mas -> %+10.2f   (el breakeven se mueve %+.5f)" % (
                n, por_noche * n, mov_be))

    lotes_l = sum(x["vol"] for x in grupos["BUY"])
    lotes_c = sum(x["vol"] for x in grupos["SELL"])
    return dict(simbolo=simbolo, tam=tam,
                lotes_l=lotes_l, lotes_c=lotes_c, neto=lotes_l - lotes_c,
                swap_ac=sum(p.swap for p in posiciones),
                flot=sum(p.profit for p in posiciones),
                noche=tasa_largo * lotes_l + tasa_corto * lotes_c)


def main():
    import MetaTrader5 as mt5

    filtro = sys.argv[1] if len(sys.argv) > 1 else None

    if not conectar(mt5):
        raise SystemExit("No conecta a MT5. Abre MetaTrader 5 y repite.")

    todas = mt5.positions_get() or []
    if filtro:
        todas = [p for p in todas if p.symbol.upper() == filtro.upper()]
    if not todas:
        print("Sin posiciones abiertas%s." % ((" en " + filtro) if filtro else ""))
        mt5.shutdown()
        return

    ahora = datetime.datetime.now()
    por_simbolo = {}
    for p in todas:
        por_simbolo.setdefault(p.symbol, []).append(p)

    print()
    print("%d posicion(es) abierta(s) en %d instrumento(s): %s" % (
        len(todas), len(por_simbolo), ", ".join(sorted(por_simbolo))))
    print("fecha %s" % ahora.strftime("%d-%m-%Y %H:%M"))

    totales = []
    for sim in sorted(por_simbolo):
        t = analizar_simbolo(mt5, sim, por_simbolo[sim], ahora)
        if t:
            totales.append(t)

    if not totales:
        mt5.shutdown()
        return

    print()
    print("=" * 94)
    print("LIBRO COMPLETO  (%d instrumento(s))" % len(totales))
    print("=" * 94)
    print("  %-12s %9s %9s %9s %12s %12s" % (
        "instrumento", "largos", "cortos", "neto", "swap acum.", "swap/noche"))
    print("  " + "-" * 70)
    for t in totales:
        print("  %-12s %9.2f %9.2f %9.2f %12.2f %12.2f" % (
            t["simbolo"], t["lotes_l"], t["lotes_c"], t["neto"],
            t["swap_ac"], t["noche"]))
    print("  " + "-" * 70)
    swap_total = sum(t["swap_ac"] for t in totales)
    noche_total = sum(t["noche"] for t in totales)
    flot_total = sum(t["flot"] for t in totales)
    print("  %-12s %9s %9s %9s %12.2f %12.2f" % (
        "TOTAL", "", "", "", swap_total, noche_total))
    print()
    print("  swap acumulado hasta hoy          %+10.2f" % swap_total)
    print("  swap por noche a la tasa de hoy   %+10.2f" % noche_total)
    print("  al mes (30 noches)                %+10.2f" % (noche_total * 30))
    print("  flotante total del libro          %+10.2f" % flot_total)

    ac = mt5.account_info()
    if ac and ac.equity > 0:
        print()
        print("  equity   %10.2f" % ac.equity)
        print("  el carry de un mes es el %.1f%% de tu equity" % (
            abs(noche_total) * 30 / ac.equity * 100))
        if noche_total < 0:
            meses = ac.equity / abs(noche_total * 30) if noche_total else 0
            print("  a esta tasa, el carry solo se come la cuenta en %.1f meses" % meses)
        else:
            print("  el carry esta A TU FAVOR: cobras por mantener el libro")
    print()

    mt5.shutdown()


if __name__ == "__main__":
    main()
