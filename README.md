# Simulador: Caída con viento (RK4 propio)

Proyecto final de Ecuaciones Diferenciales – Entrega 2. Universidad Mariana.

Modelo (y y v **positivas hacia abajo**; w > 0 = viento hacia abajo, w < 0 = hacia arriba):

    y'' = g - (c/m) |y' - w(t)| (y' - w(t)),     c = ½ ρ Cd A

Sistema de primer orden: `dy/dt = v`, `dv/dt = g - k |v - w(t)| (v - w(t))`, con `k = c/m`.
Método: **Runge-Kutta de orden 4**, programado desde cero (solo `math` y `numpy`).
No se usa `scipy.integrate` ni `sympy.dsolve`.

## Instalación

Python 3.9 o superior.

    pip install numpy matplotlib

`tkinter` viene con Python en Windows y macOS. En Linux: `sudo apt install python3-tk`.

## Cómo ejecutar

    python simulador.py       # interfaz con deslizadores
    python validacion.py      # tabla de errores, orden, figura y CSV

Los tres archivos `.py` deben estar en la misma carpeta.

## Archivos

| Archivo | Contenido |
|---|---|
| `nucleo.py` | Modelo, tipos de viento, **RK4 propio** (`f_v`, `paso_rk4`, `integrar_rk4`) y solución analítica |
| `validacion.py` | e(h) = max\|y(t_k) − y_k\| para h, h/2, h/4…, razón e(2h)/e(h), orden observado, figura y CSV |
| `simulador.py` | Interfaz (tkinter + matplotlib): parámetros físicos, excitación, condiciones iniciales, h y T |
| `tabla_convergencia.csv`, `validacion_rk4.png` | Resultados de la validación (se regeneran al ejecutar `validacion.py`) |

## Correspondencia pseudocódigo ↔ código (`nucleo.py`)

| Pseudocódigo | Código |
|---|---|
| `f_v(t, v) = g − (c/m)·\|v − w(t)\|·(v − w(t))` | `f_v(t, v, g, k, w)` |
| `ky1…ky4`, `kv1…kv4` | variables con el mismo nombre en `paso_rk4` |
| `y_{k+1} = y_k + h/6 (ky1 + 2ky2 + 2ky3 + ky4)` | última parte de `paso_rk4` |
| `v_{k+1} = v_k + h/6 (kv1 + 2kv2 + 2kv3 + kv4)` | última parte de `paso_rk4` |
| Bucle temporal t_k → t_{k+1} | `for i in range(N)` en `integrar_rk4` |

## Qué se valida

1. **Caso obligatorio:** sin viento, y0 = 0, v0 = 0.
   `y(t) = (v_t²/g)·ln cosh(g t / v_t)`, `v_t = √(m g / c)`.
   La razón e(2h)/e(h) se acerca a **16** (orden 4).
2. Casos adicionales con solución exacta: v0 > v_t (rama coth) y viento constante.
3. Casos **informativos** donde el orden baja, y es lo esperado:
   * viento constante que hace u = v − w cruzar 0: `|u|·u` no es suave → orden ≈ 3;
   * pulso rectangular: w(t) discontinua → orden ≈ 1.

Valores de referencia (m = 75 kg, A = 0.7 m², Cd = 1, ρ = 1.225, g = 9.81; T = 10 s):
v_t = 41.43 m/s, τ = v_t/g = 4.22 s.

## Guion para la revisión (15 min)

| Tiempo | Qué mostrar |
|---|---|
| 2 min | Abrir `nucleo.py`: sistema (y, v), `paso_rk4` (k1…k4), fórmulas de actualización, por qué es de orden 4 |
| 3 min | Botón «Validación RK4 vs analítica» (o `validacion.py`): gráfica, tabla de errores, razón ≈ 16 |
| 7 min | Simulador: cambiar m, A, Cd; viento (pulso, constante, senoidal, 1−cos); y0, v0; h grande y pequeño; T |
| 3 min | Preguntas: ver lista abajo |

Preguntas probables:

* ¿Por qué `|u|·u` y no `u²`? (el arrastre siempre se opone a la velocidad relativa)
* ¿Qué pasa con una ráfaga ascendente? (v_eq = w + v_t; si w = −v_t el objeto flota)
* ¿Por qué el error baja ≈ 16 veces al dividir h entre 2?
* ¿Por qué con pulso rectangular el orden baja?
* ¿Cómo se pasó la EDO de segundo orden a un sistema de primer orden?

## Notas para el documento del método

* La función de aceleración debe llevar el valor absoluto: `g − (c/m)·|v − w(t)|·(v − w(t))`.
* En el algoritmo, la última pendiente es `kv4 = f_v(t_k + h, v_k + h·kv3)` (no `ky4`).
* Indica en el documento que y y v son positivas hacia abajo.

## Bitácora de IA

El uso de IA para este código debe registrarse en la bitácora con el prompt completo y la validación
(por ejemplo: ejecutar `validacion.py` y ver que la razón de error es ≈ 16).
