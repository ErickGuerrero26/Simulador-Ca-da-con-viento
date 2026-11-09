"""
nucleo.py  -  Caída con viento: modelo, método RK4 y solución analítica
=======================================================================
Proyecto final de Ecuaciones Diferenciales (Universidad Mariana).

Modelo (convención del documento del grupo: y y v POSITIVAS HACIA ABAJO):

    y'' = g - (c/m) * |y' - w(t)| * (y' - w(t)),     c = (1/2) * rho * Cd * A

    w(t) > 0  -> viento hacia abajo (corriente descendente)
    w(t) < 0  -> viento hacia arriba (corriente ascendente)

Sistema de primer orden (v = y'):

    dy/dt = v
    dv/dt = f_v(t, v) = g - k * |v - w(t)| * (v - w(t)),     k = c/m

Solo se usan math y numpy. NO se usa scipy.integrate ni sympy.dsolve.
"""

import math
import numpy as np


# ----------------------------------------------------------------------
# 1. Parámetros físicos
# ----------------------------------------------------------------------
def coef_k(m, A, Cd, rho):
    """k = c/m = rho*Cd*A / (2 m)   [1/m]."""
    return 0.5 * rho * Cd * A / m


def velocidad_terminal(g, k):
    """v_t = sqrt(m g / c) = sqrt(g / k)   [m/s]."""
    return math.sqrt(g / k)


# ----------------------------------------------------------------------
# 2. Excitación: viento vertical w(t)
# ----------------------------------------------------------------------
TIPOS_VIENTO = ["ninguno", "constante", "pulso", "senoidal", "rafaga suave (1-cos)"]


def crear_viento(tipo, w0=0.0, f=0.5, t_ini=5.0, dur=5.0):
    """
    Devuelve (w, tramos):
      w(t)   : función Python con la velocidad del viento en el instante t.
      tramos : lista [(t_inicio, w_constante), ...] si el viento es constante
               por tramos (entonces existe solución analítica); None si no.
    """
    if tipo == "ninguno":
        return (lambda t: 0.0), [(0.0, 0.0)]

    if tipo == "constante":
        return (lambda t: w0), [(0.0, w0)]

    if tipo == "pulso":
        def w(t):
            return w0 if (t_ini <= t < t_ini + dur) else 0.0
        return w, [(0.0, 0.0), (t_ini, w0), (t_ini + dur, 0.0)]

    if tipo == "senoidal":
        return (lambda t: w0 * math.sin(2.0 * math.pi * f * t)), None

    if tipo == "rafaga suave (1-cos)":
        def w(t):
            if t_ini <= t <= t_ini + dur:
                return 0.5 * w0 * (1.0 - math.cos(2.0 * math.pi * (t - t_ini) / dur))
            return 0.0
        return w, None

    raise ValueError("tipo de viento desconocido: " + str(tipo))


# ----------------------------------------------------------------------
# 3. Método numérico: Runge-Kutta de orden 4 (RK4) para el sistema (y, v)
# ----------------------------------------------------------------------
def f_v(t, v, g, k, w):
    """Aceleración: f_v(t, v) = g - k |v - w(t)| (v - w(t))."""
    u = v - w(t)                 # velocidad relativa al aire
    return g - k * abs(u) * u


def paso_rk4(t, y, v, h, g, k, w):
    """
    Un paso de RK4 de t_k a t_{k+1} = t_k + h.
    Corresponde 1 a 1 con el algoritmo del documento:
        ky1 = v_k                      kv1 = f_v(t_k,       v_k)
        ky2 = v_k + (h/2) kv1          kv2 = f_v(t_k + h/2, v_k + (h/2) kv1)
        ky3 = v_k + (h/2) kv2          kv3 = f_v(t_k + h/2, v_k + (h/2) kv2)
        ky4 = v_k +  h    kv3          kv4 = f_v(t_k + h,   v_k +  h    kv3)
        y_{k+1} = y_k + (h/6)(ky1 + 2 ky2 + 2 ky3 + ky4)
        v_{k+1} = v_k + (h/6)(kv1 + 2 kv2 + 2 kv3 + kv4)
    """
    # k1: pendientes al inicio del intervalo
    ky1 = v
    kv1 = f_v(t, v, g, k, w)
    # k2: primer punto medio (usa k1)
    ky2 = v + 0.5 * h * kv1
    kv2 = f_v(t + 0.5 * h, v + 0.5 * h * kv1, g, k, w)
    # k3: segundo punto medio (usa k2)
    ky3 = v + 0.5 * h * kv2
    kv3 = f_v(t + 0.5 * h, v + 0.5 * h * kv2, g, k, w)
    # k4: extremo final (usa k3)
    ky4 = v + h * kv3
    kv4 = f_v(t + h, v + h * kv3, g, k, w)
    # promedio ponderado de Simpson
    y_nuevo = y + (h / 6.0) * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4)
    v_nuevo = v + (h / 6.0) * (kv1 + 2.0 * kv2 + 2.0 * kv3 + kv4)
    return y_nuevo, v_nuevo


def integrar_rk4(g, k, w, y0, v0, T, h):
    """
    Integra desde t = 0 hasta t = T con paso h (bucle temporal propio).
    Si T/h no es entero, se usa N = round(T/h) pasos y h_ef = T/N.
    Devuelve t, y, v (arreglos numpy) y h_ef.
    """
    N = max(1, int(round(T / h)))
    h_ef = T / N
    t = np.empty(N + 1)
    y = np.empty(N + 1)
    v = np.empty(N + 1)
    t[0], y[0], v[0] = 0.0, y0, v0
    for i in range(N):                      # t_k -> t_{k+1}
        t[i + 1] = (i + 1) * h_ef
        y[i + 1], v[i + 1] = paso_rk4(t[i], y[i], v[i], h_ef, g, k, w)
    return t, y, v, h_ef


# ----------------------------------------------------------------------
# 4. Solución analítica (viento constante por tramos)
# ----------------------------------------------------------------------
def _lncosh(x):
    """ln(cosh x) estable para |x| grande."""
    x = np.abs(x)
    return x + np.log1p(np.exp(-2.0 * x)) - math.log(2.0)


def _lnsinh(x):
    """ln(sinh x) estable para x > 0 grande."""
    return x + np.log1p(-np.exp(-2.0 * x)) - math.log(2.0)


def _tramo_constante(tau, y_s, v_s, w, g, k):
    """
    Solución exacta con viento CONSTANTE w, desde (y_s, v_s) en tau = 0.
    Con u = v - w:  u' = g - k |u| u   (misma ecuación que sin viento).
      |u0| < vt : u = vt tanh(a tau + b)      (a = sqrt(g k), b = atanh(u0/vt))
       u0  > vt : u = vt coth(a tau + b)      (b = atanh(vt/u0))
       u0  < 0  : u = vt tan(a tau + th0) hasta u = 0 y luego rama tanh.
    y(tau) = y_s + w tau + integral de u.
    """
    tau = np.asarray(tau, dtype=float)
    vt = math.sqrt(g / k)
    a = math.sqrt(g * k)          # = g / vt
    L = vt / a                    # = vt^2 / g
    u0 = v_s - w

    if abs(u0 - vt) < 1e-12 * vt:                 # u0 = vt: velocidad constante
        u = np.full_like(tau, vt)
        du = vt * tau
    elif u0 > vt:                                 # rama coth
        b = math.atanh(vt / u0)
        u = vt / np.tanh(a * tau + b)
        du = L * (_lnsinh(a * tau + b) - float(_lnsinh(b)))
    elif u0 >= 0.0:                               # rama tanh (0 <= u0 < vt)
        b = math.atanh(u0 / vt)
        u = vt * np.tanh(a * tau + b)
        du = L * (_lncosh(a * tau + b) - float(_lncosh(b)))
    else:                                         # u0 < 0: rama tan, luego tanh
        th0 = math.atan(u0 / vt)                  # en (-pi/2, 0)
        tau_c = -th0 / a                          # instante en que u = 0
        antes = tau <= tau_c
        ta = np.where(antes, tau, 0.0)
        u_a = vt * np.tan(a * ta + th0)
        du_a = -L * (np.log(np.cos(a * ta + th0)) - math.log(math.cos(th0)))
        tb = np.where(antes, 0.0, tau - tau_c)
        u_b = vt * np.tanh(a * tb)
        du_b = L * math.log(math.cos(th0)) + L * _lncosh(a * tb)
        u = np.where(antes, u_a, u_b)
        du = np.where(antes, du_a, du_b)

    return y_s + w * tau + du, w + u


def solucion_analitica(t, y0, v0, g, k, tramos):
    """
    Solución exacta y(t), v(t) para viento constante por tramos
    tramos = [(t_inicio, w), ...] ordenados. En cada cambio de viento, y y v
    son continuas, así que la solución se "pega" tramo a tramo.
    """
    t = np.asarray(t, dtype=float)
    y = np.empty_like(t)
    v = np.empty_like(t)
    ts, ys, vs = 0.0, y0, v0
    for j, (t_ini, w) in enumerate(tramos):
        t_fin = tramos[j + 1][0] if j + 1 < len(tramos) else math.inf
        if t_fin <= ts:                       # tramo de duración nula
            continue
        if j == 0:
            ts = t_ini
        mask = (t >= ts) & (t < t_fin) if math.isfinite(t_fin) else (t >= ts)
        if np.any(mask):
            y[mask], v[mask] = _tramo_constante(t[mask] - ts, ys, vs, w, g, k)
        if math.isfinite(t_fin):
            y_e, v_e = _tramo_constante(np.array([t_fin - ts]), ys, vs, w, g, k)
            ts, ys, vs = t_fin, float(y_e[0]), float(v_e[0])
    return y, v


def tiempo_impacto_sin_viento(H, g, k):
    """Caso w=0, y0=0, v0=0: T = (vt/g) arccosh(exp(g H / vt^2))."""
    vt = math.sqrt(g / k)
    return (vt / g) * math.acosh(math.exp(g * H / vt ** 2))
