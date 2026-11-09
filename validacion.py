"""
validacion.py  -  Validación del RK4 con la solución analítica
================================================================
Caso obligatorio (Entrega 1): sin viento, y0 = 0, v0 = 0.

    v(t) = v_t tanh(g t / v_t)
    y(t) = (v_t^2 / g) ln cosh(g t / v_t),        v_t = sqrt(m g / c)

Se calcula el error global  e(h) = max_k |y(t_k) - y_k|  para h, h/2, h/4, ...
y la razón e(2h)/e(h), que debe acercarse a 2^4 = 16 si el método es de orden 4.

Uso:
    python validacion.py              # imprime tablas y guarda figura y CSV
    python validacion.py --sin-ventana  # igual, pero sin abrir la ventana gráfica
"""

import csv
import math
import os
import sys

import numpy as np
import matplotlib

if "--sin-ventana" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from nucleo import (coef_k, velocidad_terminal, crear_viento,
                    integrar_rk4, solucion_analitica)

# Parámetros del ejemplo (paracaidista): m = 75 kg, A = 0.7 m^2, Cd = 1.0
PARAMS = dict(m=75.0, A=0.7, Cd=1.0, rho=1.225, g=9.81)
PASOS = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
T_VALID = 10.0


def error_global(g, k, wspec, y0, v0, T, h):
    """e(h) = max_k |y(t_k) - y_k| contra la solución analítica por tramos."""
    w, tramos = crear_viento(*wspec)
    t, y, v, h_ef = integrar_rk4(g, k, w, y0, v0, T, h)
    y_ex, v_ex = solucion_analitica(t, y0, v0, g, k, tramos)
    return float(np.max(np.abs(y_ex - y))), float(np.max(np.abs(v_ex - v))), h_ef


def tabla_convergencia(wspec, y0, v0, T=T_VALID, pasos=PASOS, params=PARAMS):
    """Devuelve filas (h, e_y(h), e_v(h), razon, orden_observado)."""
    g = params["g"]
    k = coef_k(params["m"], params["A"], params["Cd"], params["rho"])
    filas, e_prev = [], None
    for h in pasos:
        e_y, e_v, h_ef = error_global(g, k, wspec, y0, v0, T, h)
        razon = (e_prev / e_y) if e_prev else None
        orden = math.log2(razon) if razon else None
        filas.append((h_ef, e_y, e_v, razon, orden))
        e_prev = e_y
    return filas


def texto_tabla(filas):
    """Tabla de convergencia como texto (para consola y para la interfaz)."""
    lineas = ["%-9s %-14s %-14s %-12s %-8s" %
              ("h (s)", "e_y(h) (m)", "e_v(h) (m/s)", "e(2h)/e(h)", "orden p")]
    for h, ey, ev, r, p in filas:
        lineas.append("%-9.5f %-14.4e %-14.4e %-12s %-8s" %
                      (h, ey, ev, "-" if r is None else "%.2f" % r,
                       "-" if p is None else "%.2f" % p))
    return "\n".join(lineas)


def imprimir_tabla(titulo, filas):
    print("\n" + titulo)
    for linea in texto_tabla(filas).split("\n"):
        print("  " + linea)


def guardar_csv(ruta, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["h_s", "e_y_m", "e_v_m_s", "razon_e2h_eh", "orden_observado"])
        for h, ey, ev, r, p in filas:
            wr.writerow([h, ey, ev, "" if r is None else r, "" if p is None else p])


def figura_validacion(filas, params=PARAMS, T=T_VALID, fig=None):
    """Gráfica 1: numérica vs analítica. Gráfica 2: e(h) en escala log-log."""
    g = params["g"]
    k = coef_k(params["m"], params["A"], params["Cd"], params["rho"])
    vt = velocidad_terminal(g, k)
    w, tramos = crear_viento("ninguno")
    if fig is None:
        fig = plt.figure(figsize=(11, 4.5))
    fig.clear()
    ax1, ax2 = fig.subplots(1, 2)

    # (a) solución analítica (línea) y RK4 con un paso grueso (puntos)
    t_fino = np.linspace(0.0, T, 600)
    y_ex, _ = solucion_analitica(t_fino, 0.0, 0.0, g, k, tramos)
    t, y, v, h_ef = integrar_rk4(g, k, w, 0.0, 0.0, T, 1.0)
    ax1.plot(t_fino, y_ex, "k-", lw=1.8, label="Analítica: (v_t²/g)·ln cosh(g t/v_t)")
    ax1.plot(t, y, "o", ms=6, color="tab:red", label="RK4 propio, h = 1 s")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("y (m, positivo hacia abajo)")
    ax1.set_title("Sin viento, y0 = 0, v0 = 0  (v_t = %.2f m/s)" % vt)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # (b) error global frente a h, con pendiente 4 de referencia
    hs = np.array([f[0] for f in filas])
    es = np.array([f[1] for f in filas])
    ax2.loglog(hs, es, "o-", color="tab:red", label="e(h) = max|y(t_k) - y_k|")
    ref = es[0] * (hs / hs[0]) ** 4
    ax2.loglog(hs, ref, "k--", label="Pendiente 4 (referencia)")
    ax2.set_xticks(hs)
    ax2.set_xticklabels(["%g" % h for h in hs], rotation=45)
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("h (s)")
    ax2.set_ylabel("Error global e(h) (m)")
    ax2.set_title("Convergencia de RK4")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    return fig


def main():
    carpeta = os.path.dirname(os.path.abspath(__file__))
    print("=" * 66)
    print(" VALIDACIÓN DEL RK4 - Caída con viento")
    print(" Parámetros:", PARAMS, " T = %.1f s" % T_VALID)
    print("=" * 66)

    # --- Caso obligatorio: sin viento, reposo --------------------------
    filas = tabla_convergencia(("ninguno",), 0.0, 0.0)
    imprimir_tabla("CASO OBLIGATORIO: sin viento, y0 = 0, v0 = 0", filas)
    guardar_csv(os.path.join(carpeta, "tabla_convergencia.csv"), filas)
    ordenes = [f[4] for f in filas if f[4] is not None]
    print("  -> orden observado promedio: %.2f  (teórico = 4, razón teórica = 16)"
          % (sum(ordenes) / len(ordenes)))

    # --- Casos adicionales con solución analítica ----------------------
    extra = [
        ("Adicional 1: sin viento, v0 = 60 m/s (> v_t, rama coth)",
         ("ninguno",), 0.0, 60.0),
        ("Adicional 2: viento constante w = -15 m/s (ascendente), v0 = 20 m/s",
         ("constante", -15.0), 0.0, 20.0),
    ]
    for titulo, wspec, y0, v0 in extra:
        imprimir_tabla(titulo, tabla_convergencia(wspec, y0, v0))

    # --- Casos informativos: pérdida de orden por falta de suavidad ----
    info = [
        ("Informativo 1: viento constante w = +15 m/s, v0 = 0  "
         "(u = v - w cruza 0: |u|u no es C^2)",
         ("constante", 15.0), 0.0, 0.0),
        ("Informativo 2: pulso rectangular de viento w = -15 m/s "
         "(w(t) discontinua)",
         ("pulso", -15.0, 0.5, 5.0, 5.0), 0.0, 0.0),
    ]
    for titulo, wspec, y0, v0 in info:
        imprimir_tabla(titulo, tabla_convergencia(wspec, y0, v0, T=12.0))
    print("\nNota: RK4 solo alcanza orden 4 si el lado derecho es suficientemente"
          " suave.\nEn los casos informativos el orden baja (aprox. 3 y 1) y es"
          " el comportamiento esperado.")

    fig = figura_validacion(filas)
    ruta_png = os.path.join(carpeta, "validacion_rk4.png")
    fig.savefig(ruta_png, dpi=150)
    print("\nFigura guardada en:", ruta_png)
    print("Tabla guardada en:", os.path.join(carpeta, "tabla_convergencia.csv"))
    if "--sin-ventana" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
