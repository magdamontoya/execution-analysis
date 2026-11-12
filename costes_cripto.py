# -*- coding: utf-8 -*-
"""
costes_cripto.py - Que cuesta de verdad operar cada cripto en tu broker

Corre:  python costes_cripto.py
        python costes_cripto.py --lote 0.1
Requiere MetaTrader 5 abierto.

Para cada simbolo calcula, sobre UN lote:
  - valor nocional
  - spread: en dinero, en % del nocional, y cuanto tiene que moverse
    el precio solo para empatar la ida y vuelta
  - swap largo y corto por noche, y su equivalente ANUALIZADO en %
    (asi 41 instrumentos distintos se pueden comparar entre si)
  - margen requerido y apalancamiento efectivo
  - cuantos dias de swap equivalen a un spread de ida y vuelta

La columna que manda es 'carry %/ano': es lo unico que hace comparables
instrumentos con precios de $0.00002 y de $90,000.
"""

import sys

REF = 1000.0   # nocional de referencia en USD: todo se compara por $1,000

SIMBOLOS = [
    "ADAUSD", "AVAXUSD", "BCHUSD", "BNBUSD", "COMPUSD", "DOGEUSD", "DOTUSD",
    "ETCUSD", "GLMRUSD", "KSMUSD", "LINKUSD", "SOLUSD", "UNIUSD", "XLMUSD",
    "XRPUSD", "XTZUSD", "TRUMPUSD", "AAVUSD", "ATMUSD", "FLOUSD", "JUPUSD",
    "NERUSD", "ONDUSD", "PEPUSD", "SHBUSD", "TRXUSD", "WIFUSD", "ARBUSD",
    "BNKUSD", "MANUSD", "SANUSD", "POLUSD", "SONUSD", "HBARUSD", "SUIUSD",
    "TONUSD", "APTUSD", "HYPEUSD", "INJUSD", "RENDERUSD", "FETUSD", "USCUSD",
]


def conectar(mt5):
    import os, glob
    if mt5.initialize():
        return True
    rutas = glob.glob(r"C:\Program Files\*\terminal64.exe")
    rutas += glob.glob(os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe"))
    for ruta in rutas:
        if mt5.initialize(path=ruta):
            return True
    return False


def resolver(mt5, pedido, todos):
    """Encuentra el nombre real: exacto, o con sufijo/prefijo del broker."""
    if pedido in todos:
        return pedido
    p = pedido.upper()
    for n in todos:
        if n.upper() == p:
            return n
    # el broker a veces anade sufijos: BTCUSD.r, BTCUSD-ECN ...
    for n in todos:
        u = n.upper()
        if u.startswith(p) and len(u) - len(p) <= 5:
            return n
    return None


def analizar(mt5, nombre, lote):
    info = mt5.symbol_info(nombre)
    if info is None:
        return None
    if not info.visible:
        mt5.symbol_select(nombre, True)
        info = mt5.symbol_info(nombre)
    tick = mt5.symbol_info_tick(nombre)
    if tick is None or tick.ask <= 0:
        return None

    precio = (tick.bid + tick.ask) / 2.0
    tam = info.trade_contract_size
    noc_lote = precio * tam           # nocional de UN lote
    if noc_lote <= 0:
        return None

    # --- EL SPREAD ES LO UNICO QUE NO DEPENDE DE SUPUESTOS DE UNIDADES:
    # sale de bid/ask en vivo. Fiable siempre.
    spread_px = tick.ask - tick.bid
    spread_pct = 100.0 * spread_px / precio

    # --- TODO LO DEMAS SE NORMALIZA A $1,000 DE NOCIONAL.
    # Los contract_size van de 1 a 100,000 entre estos simbolos, asi que
    # "un lote" significa $1 en uno y $99,970 en otro: comparar por lote
    # no tiene sentido. Por $1,000 si.
    lotes_1k = REF / noc_lote

    unidad = info.point * tam         # 1 punto de swap -> dinero por lote
    swap_l_lote = info.swap_long * unidad
    swap_s_lote = info.swap_short * unidad
    swap_l = swap_l_lote * lotes_1k   # por noche, por $1,000
    swap_s = swap_s_lote * lotes_1k
    carry_l = 100.0 * swap_l * 365.0 / REF
    carry_s = 100.0 * swap_s * 365.0 / REF

    margen = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, nombre, lote, tick.ask)
    apal = (noc_lote * lote / margen) if margen else float("nan")

    spread_1k = REF * spread_pct / 100.0
    peor = min(swap_l, swap_s)
    dias_eq = (spread_1k / abs(peor)) if peor < 0 else float("inf")

    # --- marca de fiabilidad: con nocionales por lote diminutos, el campo
    # de swap (2 decimales) tiene tan poca resolucion que el % se dispara.
    dudoso = []
    if noc_lote < 50:
        dudoso.append("nocional/lote $%.2f" % noc_lote)
    if max(abs(carry_l), abs(carry_s)) > 150:
        dudoso.append("carry absurdo")

    return {
        "sim": nombre, "precio": precio, "noc_lote": noc_lote,
        "spread_pct": spread_pct, "spread_1k": spread_1k,
        "swap_l": swap_l, "swap_s": swap_s,
        "carry_l": carry_l, "carry_s": carry_s,
        "margen": margen, "apal": apal, "dias_eq": dias_eq,
        "lote_min": info.volume_min, "dudoso": dudoso,
    }


def main():
    import MetaTrader5 as mt5

    lote = 1.0
    if "--lote" in sys.argv:
        lote = float(sys.argv[sys.argv.index("--lote") + 1])

    if not conectar(mt5):
        raise SystemExit("No conecta a MT5. Abre MetaTrader 5 y repite.")

    todos = [s.name for s in (mt5.symbols_get() or [])]
    filas, faltan = [], []
    for s in SIMBOLOS:
        real = resolver(mt5, s, todos)
        if real is None:
            faltan.append(s)
            continue
        r = analizar(mt5, real, lote)
        if r is None:
            faltan.append(s + " (sin cotizacion)")
        else:
            filas.append(r)

    if not filas:
        print("No se pudo leer ningun simbolo.")
        mt5.shutdown()
        return

    print()
    print("=" * 118)
    print("COSTE DE OPERAR CRIPTOS   lote = %s   (%d de %d simbolos)"
          % (lote, len(filas), len(SIMBOLOS)))
    print("=" * 118)
    print()
    cab = "%-11s %12s %12s %8s %10s %10s %9s %9s %9s %7s"
    print("Todo por $1,000 de nocional, salvo precio, margen (1 lote) y apalancamiento.")
    print()
    print(cab % ("simbolo", "precio", "noc/lote", "spread%", "swap L", "swap S",
                 "carry L%", "carry S%", "margen", "apal"))
    print("-" * 118)

    def fmt_px(p):
        if p < 0.01:
            return "%.8f" % p
        if p < 1:
            return "%.5f" % p
        return format(p, ",.2f")

    for r in sorted(filas, key=lambda x: x["spread_pct"]):
        print(cab % (
            r["sim"],
            fmt_px(r["precio"]),
            "%.0f" % r["noc_lote"],
            "%.3f" % r["spread_pct"],
            "%+.2f" % r["swap_l"],
            "%+.2f" % r["swap_s"],
            "%+.1f" % r["carry_l"],
            "%+.1f" % r["carry_s"],
            ("%.0f" % r["margen"]) if r["margen"] else "n/d",
            ("%.0f:1" % r["apal"]) if r["apal"] == r["apal"] else "n/d",
        ) + ("   <- %s" % "; ".join(r["dudoso"]) if r["dudoso"] else ""))

    print()
    print("=" * 118)
    print("LOS QUE IMPORTAN")
    print("=" * 118)

    barato = sorted(filas, key=lambda x: x["spread_pct"])[:5]
    caro = sorted(filas, key=lambda x: -x["spread_pct"])[:5]
    print("\n  Spread mas BARATO (% del nocional, ida y vuelta se paga una vez):")
    for r in barato:
        print("    %-11s %6.3f%%   -> el precio debe moverse %.3f%% solo para empatar"
              % (r["sim"], r["spread_pct"], r["spread_pct"]))
    print("\n  Spread mas CARO:")
    for r in caro:
        print("    %-11s %6.3f%%" % (r["sim"], r["spread_pct"]))

    fiables = [f for f in filas if not f["dudoso"]]
    pagan_l = sorted(fiables, key=lambda x: x["carry_l"])[:5]
    cobran_s = sorted(fiables, key=lambda x: -x["carry_s"])[:5]
    print("\n  Mas CARO estar LARGO (carry anualizado):")
    for r in pagan_l:
        print("    %-11s %+8.1f%%/ano   (%+.2f por noche)" % (r["sim"], r["carry_l"], r["swap_l"]))
    print("\n  Te PAGAN mas por estar CORTO:")
    for r in cobran_s:
        print("    %-11s %+8.1f%%/ano   (%+.2f por noche)" % (r["sim"], r["carry_s"], r["swap_s"]))

    print("\n  Donde el SPREAD se come el swap mas rapido:")
    for r in sorted([f for f in fiables if f["dias_eq"] < 1e6], key=lambda x: x["dias_eq"])[:5]:
        print("    %-11s un spread = %.1f dias de swap" % (r["sim"], r["dias_eq"]))

    if faltan:
        print()
        print("  No encontrados en tu broker (%d): %s" % (len(faltan), ", ".join(faltan)))

    print()
    n_malos = len([f for f in filas if f["dudoso"]])
    print("  Filas marcadas <- : el nocional de 1 lote es tan pequeno que el campo de")
    print("  swap (2 decimales) no tiene resolucion. Su carry %% no es de fiar y quedan")
    print("  fuera de los rankings de carry. Son %d de %d. El SPREAD si es fiable en todas:" % (n_malos, len(filas)))
    print("  sale de bid/ask en vivo y no depende de ningun supuesto de unidades.")
    print()
    mt5.shutdown()


if __name__ == "__main__":
    main()
