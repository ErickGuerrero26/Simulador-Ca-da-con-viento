"""
simulador.py  -  Simulador interactivo: Caída con viento (RK4 propio)
======================================================================
Ejecutar:   python simulador.py

Permite variar en tiempo real:
  * Parámetros físicos: m, A, Cd, rho, g
  * Excitación: tipo de viento, amplitud, frecuencia, inicio y duración
  * Condiciones iniciales: y0, v0
  * Método numérico: paso h, tiempo de simulación T (y altura de caída H)
y muestra y(t), v(t) y la excitación w(t).

ANIMACIÓN INTERACTIVA (panel central):
  * Reproducir / pausar / reiniciar, velocidad de reproducción y ventana de vista.
  * Deslizador de tiempo y clic (o arrastre) sobre las gráficas para ir a un instante.
  * Muestra el objeto, la corriente de aire relativa, el peso, el arrastre y el viento.

Convención: y y v son POSITIVAS HACIA ABAJO.
            w > 0 = viento hacia abajo,  w < 0 = viento hacia arriba.
"""

import math
import time
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from nucleo import (TIPOS_VIENTO, coef_k, velocidad_terminal, crear_viento,
                    integrar_rk4, solucion_analitica)
import validacion

# Valores por defecto: ejemplo del paracaidista (m = 75 kg, A = 0.7 m^2, Cd = 1)
DEFAULTS = dict(m=75.0, A=0.7, Cd=1.0, rho=1.225, g=9.81,
                y0=0.0, v0=0.0,
                w0=-15.0, f=0.5, t_ini=5.0, dur=5.0,
                h=0.05, T=30.0, H=500.0)
TIPO_DEFECTO = "pulso"

# ----------------------------------------------------------------------
# Tema visual (solo apariencia: no afecta al cálculo)
# ----------------------------------------------------------------------
TEMA = dict(
    fondo="#eef2f7", tarjeta="#ffffff", borde="#d3dbe6",
    texto="#1f2937", texto_suave="#64748b",
    cabecera="#0f2742", cab_sub="#9fb8d8", acento="#2563eb", acento_osc="#1d4ed8",
    cielo_arriba="#74bdf0", cielo_abajo="#e8f5ff",
    peso="#dc2626", arrastre="#16a34a", viento="#f59e0b", aire="#3b82f6",
    suelo="#8b6b4a", pasto="#4caf50", cursor="#e11d48",
    analitica="#111827", gris="#6b7280",
)


def _mezcla(c1, c2, t):
    """Color intermedio entre dos colores '#rrggbb' (t = 0 -> c1, t = 1 -> c2)."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))

# clave: (mínimo, máximo, resolución, etiqueta con unidades)
RANGOS = {
    "m":     (0.05, 150.0, 0.05, "Masa  m  (kg)"),
    "A":     (0.001, 1.5, 0.001, "Área frontal  A  (m²)"),
    "Cd":    (0.1, 2.0, 0.01, "Coeficiente de arrastre  Cd  (adimensional)"),
    "rho":   (0.5, 1.3, 0.005, "Densidad del aire  ρ  (kg/m³)"),
    "g":     (1.0, 25.0, 0.01, "Gravedad  g  (m/s²)"),
    "y0":    (-100.0, 100.0, 0.5, "Posición inicial  y0  (m)"),
    "v0":    (-50.0, 80.0, 0.5, "Velocidad inicial  v0  (m/s, + hacia abajo)"),
    "w0":    (-30.0, 30.0, 0.5, "Amplitud del viento  w0  (m/s, + hacia abajo)"),
    "f":     (0.05, 5.0, 0.05, "Frecuencia (senoidal)  f  (Hz)"),
    "t_ini": (0.0, 60.0, 0.5, "Inicio de la ráfaga  t1  (s)"),
    "dur":   (0.1, 30.0, 0.1, "Duración de la ráfaga  Δt  (s)"),
    "h":     (0.005, 1.0, 0.005, "Paso de malla  h  (s)"),
    "T":     (1.0, 120.0, 1.0, "Tiempo de simulación  T  (s)"),
    "H":     (10.0, 5000.0, 10.0, "Altura de caída  H  (m)"),
}

# Qué controles de viento usa cada tipo de excitación
USA = {
    "ninguno": [],
    "constante": ["w0"],
    "pulso": ["w0", "t_ini", "dur"],
    "senoidal": ["w0", "f"],
    "rafaga suave (1-cos)": ["w0", "t_ini", "dur"],
}


class Simulador(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador: caída con viento (RK4 propio)")
        self._aplicar_tema()
        ancho = min(1500, self.winfo_screenwidth() - 40)
        alto = min(860, self.winfo_screenheight() - 80)
        self.geometry("%dx%d" % (ancho, alto))
        self.minsize(min(1200, ancho), min(680, alto))

        self.vars = {}
        self.escalas = {}
        self.valores_lbl = {}
        self._after_id = None
        self.tipo = tk.StringVar(value=TIPO_DEFECTO)
        self.ver_analitica = tk.BooleanVar(value=True)
        self.parar_suelo = tk.BooleanVar(value=False)

        # --- estado de la animación ---
        self.datos = None                # resultados de la última simulación
        self.t_anim = 0.0                # instante mostrado en la animación (s)
        self.reproduciendo = False
        self._t_ultimo = None
        self._interno = False            # evita bucles al mover el deslizador por código
        self._bg = None                  # fondo para blitting de los cursores
        self.var_t = tk.DoubleVar(value=0.0)
        self.var_vel = tk.DoubleVar(value=1.0)
        self.var_vista = tk.DoubleVar(value=150.0)
        self.ver_fuerzas = tk.BooleanVar(value=True)
        self.ver_aire = tk.BooleanVar(value=True)
        self.repetir = tk.BooleanVar(value=False)

        self._construir_cabecera()
        self._construir_panel()
        self._construir_animacion()
        self._construir_grafica()
        self._actualizar_estado_controles()
        self.actualizar()

    # ------------------------------------------------------------------
    # Apariencia
    # ------------------------------------------------------------------
    def _aplicar_tema(self):
        """Colores, fuentes y estilos ttk. Solo cambia el aspecto."""
        disponibles = set(tkfont.families(self))
        self.fuente = tkfont.nametofont("TkDefaultFont").actual("family")
        for f in ("Segoe UI", "Helvetica Neue", "Inter", "Ubuntu", "Noto Sans", "DejaVu Sans", "Arial"):
            if f in disponibles:
                self.fuente = f
                break
        for nombre in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(nombre).configure(family=self.fuente, size=10)
        fu = self.fuente
        self.configure(bg=TEMA["fondo"])

        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=TEMA["fondo"], foreground=TEMA["texto"], font=(fu, 10))
        st.configure("TFrame", background=TEMA["fondo"])
        st.configure("TLabel", background=TEMA["fondo"], foreground=TEMA["texto"])
        st.configure("Titulo.TLabel", font=(fu, 11, "bold"), foreground=TEMA["acento"])
        st.configure("TLabelframe", background=TEMA["fondo"], bordercolor=TEMA["borde"])
        st.configure("TLabelframe.Label", background=TEMA["fondo"],
                     foreground=TEMA["acento"], font=(fu, 10, "bold"))
        st.configure("TNotebook", background=TEMA["fondo"], borderwidth=0, tabmargins=(2, 4, 2, 0))
        st.configure("TNotebook.Tab", padding=(10, 6), background="#dbe3ee",
                     foreground=TEMA["texto_suave"], borderwidth=0, font=(fu, 10, "bold"))
        st.map("TNotebook.Tab", background=[("selected", TEMA["acento"])],
               foreground=[("selected", "#ffffff")])
        st.configure("TButton", padding=(10, 6), background="#e2e8f0", foreground=TEMA["texto"],
                     borderwidth=0, focusthickness=0, focuscolor="none")
        st.map("TButton", background=[("active", "#cbd5e1")])
        st.configure("Acento.TButton", padding=(12, 6), background=TEMA["acento"],
                     foreground="#ffffff", font=(fu, 10, "bold"))
        st.map("Acento.TButton", background=[("active", TEMA["acento_osc"]),
                                             ("pressed", TEMA["acento_osc"])])
        st.configure("TCheckbutton", background=TEMA["fondo"], focuscolor=TEMA["fondo"])
        st.map("TCheckbutton", background=[("active", TEMA["fondo"])],
               indicatorcolor=[("selected", TEMA["acento"])])
        st.configure("TCombobox", fieldbackground="#ffffff", padding=4)

        # estilo común de los deslizadores tk.Scale
        # (la manija usa el color "bg"; el valor se muestra en una etiqueta aparte)
        self.kw_escala = dict(bg=TEMA["acento"], troughcolor="#d5deea", showvalue=False,
                              highlightthickness=0, bd=0, activebackground=TEMA["acento_osc"],
                              sliderrelief="flat", sliderlength=26, width=14)

    def _construir_cabecera(self):
        cab = tk.Frame(self, bg=TEMA["cabecera"])
        cab.pack(side="top", fill="x")
        tk.Frame(cab, bg=TEMA["acento"], height=3).pack(side="bottom", fill="x")
        izq = tk.Frame(cab, bg=TEMA["cabecera"])
        izq.pack(side="left", padx=16, pady=7)
        tk.Label(izq, text="Caída con viento", bg=TEMA["cabecera"], fg="#ffffff",
                 font=(self.fuente, 18, "bold")).pack(anchor="w")
        tk.Label(izq, text="Simulador de ecuaciones diferenciales  ·  Runge-Kutta de orden 4 (RK4) propio",
                 bg=TEMA["cabecera"], fg=TEMA["cab_sub"], font=(self.fuente, 10)).pack(anchor="w")
        tk.Label(cab, text="y'' = g − (c/m)·|y' − w(t)|·(y' − w(t))", bg=TEMA["cabecera"],
                 fg="#cfe3ff", font=("Courier", 12, "bold")).pack(side="right", padx=18)

    # ------------------------------------------------------------------
    # Interfaz
    # ------------------------------------------------------------------
    def _fila_escala(self, padre, etiqueta, var, desde, hasta, res, comando, largo):
        """Etiqueta + valor numérico + deslizador. Devuelve (escala, etiqueta, valor)."""
        marco = ttk.Frame(padre)
        marco.pack(fill="x", padx=6, pady=(3, 4))
        cabecera = ttk.Frame(marco)
        cabecera.pack(fill="x")
        lbl = ttk.Label(cabecera, text=etiqueta)
        lbl.pack(side="left")
        decimales = next(d for d in range(0, 6)
                         if abs(res * 10 ** d - round(res * 10 ** d)) < 1e-9)
        vlbl = ttk.Label(cabecera, text="", foreground=TEMA["acento"],
                         font=(self.fuente, 10, "bold"))
        vlbl.pack(side="right")

        def _mostrar(*_):
            vlbl.configure(text="%.*f" % (decimales, var.get()))
        var.trace_add("write", _mostrar)
        _mostrar()
        esc = tk.Scale(marco, from_=desde, to=hasta, resolution=res, orient="horizontal",
                       variable=var, length=largo, command=comando, **self.kw_escala)
        esc.pack(fill="x", pady=(2, 0))
        return esc, lbl, vlbl

    def _slider(self, padre, clave):
        desde, hasta, res, etiqueta = RANGOS[clave]
        var = tk.DoubleVar(value=DEFAULTS[clave])
        self.vars[clave] = var
        esc, lbl, vlbl = self._fila_escala(padre, etiqueta, var, desde, hasta, res,
                                           lambda _v: self.programar(), 310)
        self.escalas[clave] = (esc, lbl)
        self.valores_lbl[clave] = vlbl

    def _construir_panel(self):
        panel = ttk.Frame(self, padding=(8, 6, 4, 6))
        panel.pack(side="left", fill="y")
        ttk.Label(panel, text="Parámetros", style="Titulo.TLabel").pack(anchor="w", pady=(0, 3))

        libro = ttk.Notebook(panel)
        libro.pack(fill="x")

        tab1 = ttk.Frame(libro)
        libro.add(tab1, text="Modelo físico")
        for c in ("m", "A", "Cd", "rho", "g"):
            self._slider(tab1, c)

        tab2 = ttk.Frame(libro)
        libro.add(tab2, text="Excitación (viento)")
        ttk.Label(tab2, text="Tipo de viento vertical  w(t):").pack(anchor="w", padx=6, pady=(6, 0))
        cb = ttk.Combobox(tab2, textvariable=self.tipo, values=TIPOS_VIENTO,
                          state="readonly", width=28)
        cb.pack(anchor="w", padx=6, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._cambio_tipo())
        for c in ("w0", "f", "t_ini", "dur"):
            self._slider(tab2, c)

        tab3 = ttk.Frame(libro)
        libro.add(tab3, text="Condiciones y método")
        for c in ("y0", "v0", "h", "T", "H"):
            self._slider(tab3, c)

        opciones = ttk.LabelFrame(panel, text="Opciones", padding=4)
        opciones.pack(fill="x", pady=4)
        ttk.Checkbutton(opciones, text="Mostrar solución analítica (si existe)",
                        variable=self.ver_analitica,
                        command=self.actualizar).pack(anchor="w")
        ttk.Checkbutton(opciones, text="Detener al llegar al suelo (y = H)",
                        variable=self.parar_suelo,
                        command=self.actualizar).pack(anchor="w")

        botones = ttk.Frame(panel)
        botones.pack(fill="x", pady=2)
        ttk.Button(botones, text="Restablecer valores",
                   command=self.restablecer).pack(side="left", padx=2)
        ttk.Button(botones, text="Validación RK4 vs analítica", style="Acento.TButton",
                   command=self.abrir_validacion).pack(side="left", padx=2)

        info = ttk.LabelFrame(panel, text="Resultados", padding=4)
        info.pack(fill="both", expand=True, pady=4)
        self.texto_info = tk.Label(info, text="", justify="left", anchor="nw",
                                   font=("Courier", 9), wraplength=320, bg=TEMA["tarjeta"],
                                   fg=TEMA["texto"], padx=8, pady=6,
                                   highlightthickness=1, highlightbackground=TEMA["borde"])
        self.texto_info.pack(fill="both", expand=True)

    def _construir_animacion(self):
        col = ttk.Frame(self, padding=(4, 6))
        col.pack(side="left", fill="y")
        ttk.Label(col, text="Animación interactiva", style="Titulo.TLabel").pack(
            side="top", anchor="w", pady=(0, 3))

        # los controles se anclan abajo; el lienzo ocupa el espacio que sobre
        ctrl = ttk.Frame(col)
        ctrl.pack(side="bottom", fill="x")
        self.lienzo = tk.Canvas(col, width=280, height=300, bg=TEMA["cielo_abajo"],
                                highlightthickness=1, highlightbackground=TEMA["borde"])
        self.lienzo.pack(side="top", fill="both", expand=True)
        self.lienzo.bind("<Configure>", lambda e: self.dibujar_animacion())

        fila = ttk.Frame(ctrl)
        fila.pack(fill="x", pady=(4, 0))
        self.btn_play = ttk.Button(fila, text="▶ Reproducir", style="Acento.TButton",
                                   command=self.alternar_reproduccion)
        self.btn_play.pack(side="left", padx=2)
        ttk.Button(fila, text="↺ Reiniciar", command=self.reiniciar_animacion).pack(side="left", padx=2)
        ttk.Checkbutton(fila, text="Bucle", variable=self.repetir).pack(side="left", padx=4)

        self.escala_t, _, _ = self._fila_escala(
            ctrl, "Tiempo de la animación  t  (s)", self.var_t, 0.0, 30.0, 0.01,
            lambda v: self._mover_tiempo(float(v)), 280)
        self._fila_escala(ctrl, "Velocidad de reproducción  (× tiempo real)", self.var_vel,
                          0.25, 10.0, 0.25, None, 280)
        self._fila_escala(ctrl, "Ventana de vista  (m visibles en pantalla)", self.var_vista,
                          30.0, 600.0, 10.0, lambda v: self.dibujar_animacion(), 280)
        ttk.Checkbutton(ctrl, text="Mostrar fuerzas (peso y arrastre)",
                        variable=self.ver_fuerzas,
                        command=self.dibujar_animacion).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Mostrar aire relativo al objeto",
                        variable=self.ver_aire,
                        command=self.dibujar_animacion).pack(anchor="w")

    def _construir_grafica(self):
        marco = ttk.Frame(self, padding=(4, 6, 8, 6))
        marco.pack(side="left", fill="both", expand=True)
        ttk.Label(marco, text="Respuesta del sistema", style="Titulo.TLabel").pack(
            side="top", anchor="w", pady=(0, 3))
        self.fig = Figure(figsize=(8, 7), dpi=100, constrained_layout=True, facecolor="#ffffff")
        self.ax_y, self.ax_v = self.fig.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=marco)
        self.barra = NavigationToolbar2Tk(self.canvas, marco)
        self.barra.update()
        try:                                   # barra de herramientas con el color del tema
            self.barra.configure(background=TEMA["fondo"])
            for hijo in self.barra.winfo_children():
                try:
                    hijo.configure(background=TEMA["fondo"])
                except tk.TclError:
                    pass
        except tk.TclError:
            pass
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        # clic o arrastre sobre las gráficas = ir a ese instante
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("button_press_event", self._clic_grafica)
        self.canvas.mpl_connect("motion_notify_event", self._clic_grafica)

    # ------------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------------
    def _cambio_tipo(self):
        self._actualizar_estado_controles()
        self.actualizar()

    def _actualizar_estado_controles(self):
        usados = USA[self.tipo.get()]
        for clave in ("w0", "f", "t_ini", "dur"):
            esc, lbl = self.escalas[clave]
            activo = clave in usados
            esc.configure(state="normal" if activo else "disabled",
                          bg=TEMA["acento"] if activo else "#b6bfcc")
            lbl.configure(foreground=TEMA["texto"] if activo else "#b6bfcc")
            self.valores_lbl[clave].configure(
                foreground=TEMA["acento"] if activo else "#b6bfcc")

    def programar(self):
        """Espera 120 ms sin cambios antes de recalcular (evita cálculos de más)."""
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(120, self.actualizar)

    def restablecer(self):
        for clave, var in self.vars.items():
            var.set(DEFAULTS[clave])
        self.tipo.set(TIPO_DEFECTO)
        self.ver_analitica.set(True)
        self.parar_suelo.set(False)
        self._actualizar_estado_controles()
        self.actualizar()

    # ------------------------------------------------------------------
    # Cálculo y gráfica
    # ------------------------------------------------------------------
    def actualizar(self):
        self._after_id = None
        p = {c: v.get() for c, v in self.vars.items()}
        k = coef_k(p["m"], p["A"], p["Cd"], p["rho"])
        g = p["g"]
        vt = velocidad_terminal(g, k)
        tipo = self.tipo.get()
        w, tramos = crear_viento(tipo, p["w0"], p["f"], p["t_ini"], p["dur"])

        # --- integración con el RK4 propio ---
        t, y, v, h_ef = integrar_rk4(g, k, w, p["y0"], p["v0"], p["T"], p["h"])

        # --- impacto con el suelo (y = H) ---
        H = p["H"]
        t_imp = v_imp = None
        if p["y0"] < H and np.any(y >= H):
            i = int(np.argmax(y >= H))                   # primer punto con y >= H
            lam = (H - y[i - 1]) / (y[i] - y[i - 1])     # interpolación lineal
            t_imp = t[i - 1] + lam * (t[i] - t[i - 1])
            v_imp = v[i - 1] + lam * (v[i] - v[i - 1])

            # Si está activada la opción de detener al llegar al suelo,
            # se fuerza la física del reposo tras el impacto (y = H, v = 0)
            if self.parar_suelo.get():
                t_pre = t[:i]
                y_pre = y[:i]
                v_pre = v[:i]

                t_post = t[i:]
                y_post = np.full_like(t_post, H)
                v_post = np.zeros_like(t_post)

                # Se concatena el evento exacto de impacto (caída a v_imp -> detención a 0 m/s)
                t = np.concatenate([t_pre, [t_imp, t_imp + 1e-6], t_post])
                y = np.concatenate([y_pre, [H, H], y_post])
                v = np.concatenate([v_pre, [v_imp, 0.0], v_post])

        t_fin = t[-1]

        # --- datos para la animación ---
        tw = np.linspace(0.0, t_fin, 4001)
        ww = np.array([w(x) for x in tw])
        Wc = np.concatenate(([0.0], np.cumsum(0.5 * (ww[1:] + ww[:-1]) * np.diff(tw))))
        self.datos = dict(t=t, y=y, v=v, w=w, k=k, g=g, vt=vt, H=H, t_imp=t_imp,
                          v_imp=v_imp, tw=tw, Wc=Wc, t_fin=t_fin)
        self.t_anim = min(self.t_anim, t_fin)

        # --- solución analítica ---
        hay_analitica = tramos is not None
        err_txt = "Sin solución analítica para este tipo de viento."
        y_an = v_an = t_an = None
        if hay_analitica:
            t_an = np.linspace(0.0, t_fin, 1200)
            y_an, v_an = solucion_analitica(t_an, p["y0"], p["v0"], g, k, tramos)
            
            # Ajuste de la gráfica analítica tras el impacto
            if self.parar_suelo.get() and t_imp is not None:
                mask_an = t_an >= t_imp
                y_an[mask_an] = H
                v_an[mask_an] = 0.0

            t_cmp = t[:-1] if (self.parar_suelo.get() and t_imp is not None) else t
            y_ex, _ = solucion_analitica(t_cmp, p["y0"], p["v0"], g, k, tramos)
            if self.parar_suelo.get() and t_imp is not None:
                mask_cmp = t_cmp >= t_imp
                y_ex[mask_cmp] = H

            err = float(np.max(np.abs(y_ex - y[:len(t_cmp)])))
            err_txt = "Error máx |y_analítica - y_RK4| = %.3e m" % err
            if tipo == "pulso":
                err_txt += "\n(w(t) discontinua: RK4 baja a orden ~1)"

        # --- gráficas ---
        ax_y, ax_v = self.ax_y, self.ax_v
        ax_y.clear()
        ax_v.clear()

        ax_y.plot(t, y, color="tab:blue", lw=2, label="y(t)  RK4 propio")
        if hay_analitica and self.ver_analitica.get():
            ax_y.plot(t_an, y_an, "k--", lw=1.2, label="y(t)  analítica")
        if self.parar_suelo.get():
            ax_y.axhline(H, color="saddlebrown", lw=1.5, ls="-", label="Suelo (y = H)")
        if t_imp is not None:
            ax_y.plot([t_imp], [H], "rv", ms=9, label="Impacto")
        ax_y.set_ylabel("Posición y (m)\n(positiva hacia abajo)")
        ax_y.invert_yaxis()
        ax_y.set_title("Caída con resistencia cuadrática y viento vertical  "
                       "(h = %.4g s)" % h_ef, fontsize=11)
        ax_y.grid(alpha=0.3)
        ax_y.legend(loc="upper left", fontsize=8)

        ax_v.plot(t, v, color="tab:blue", lw=2, label="v(t)  RK4 propio")
        if hay_analitica and self.ver_analitica.get():
            ax_v.plot(t_an, v_an, "k--", lw=1.2, label="v(t)  analítica")
        t_w = np.linspace(0.0, t_fin, 1500)
        ax_v.plot(t_w, [w(x) for x in t_w], color="tab:orange", lw=1.6,
                  label="w(t)  viento (excitación)")
        ax_v.axhline(vt, color="gray", ls=":", lw=1.4,
                     label="v_t sin viento = %.2f m/s" % vt)
        if tipo == "constante":
            ax_v.axhline(p["w0"] + vt, color="tab:green", ls=":", lw=1.4,
                         label="v_eq = w + v_t = %.2f m/s" % (p["w0"] + vt))
        ax_v.set_xlabel("Tiempo t (s)")
        ax_v.set_ylabel("Velocidad (m/s)\n(positiva hacia abajo)")
        ax_v.grid(alpha=0.3)
        ax_v.legend(loc="best", fontsize=8)
        ax_v.set_xlim(0.0, t_fin)

        # cursores de tiempo
        self.cur_y = ax_y.axvline(0.0, color="crimson", lw=1.3, animated=True)
        self.cur_v = ax_v.axvline(0.0, color="crimson", lw=1.3, animated=True)
        self.pt_y, = ax_y.plot([], [], "o", color="crimson", ms=8, animated=True, zorder=10)
        self.pt_v, = ax_v.plot([], [], "o", color="crimson", ms=8, animated=True, zorder=10)
        self.pt_w, = ax_v.plot([], [], "o", color="tab:orange", ms=7, animated=True, zorder=10)
        self._poner_datos_cursores()
        self._bg = None

        self.canvas.draw_idle()

        self._interno = True
        self.escala_t.configure(to=t_fin)
        self.var_t.set(self.t_anim)
        self._interno = False
        self.dibujar_animacion()

        lineas = [
            "k = c/m = %.5f 1/m,   c = %.4f kg/m" % (k, 0.5 * p["rho"] * p["Cd"] * p["A"]),
            "v_t = sqrt(m g / c) = %.2f m/s" % vt,
            "tau = v_t / g = %.2f s" % (vt / g),
            "h efectivo = %.4g s,   pasos = %d" % (h_ef, len(t) - 1),
            "y(T) = %.2f m,   v(T) = %.2f m/s" % (y[-1], v[-1]),
        ]
        if t_imp is not None:
            lineas.append("Impacto (y = H): t = %.2f s,   v = %.2f m/s" % (t_imp, v_imp))
        else:
            lineas.append("No llega a y = H en el tiempo simulado.")
        lineas.append(err_txt)
        self.texto_info.configure(text="\n".join(lineas))
    # ------------------------------------------------------------------
    # Animación interactiva
    # ------------------------------------------------------------------
    def _estado(self, t):
        d = self.datos
        if d is None:
            return 0.0, 0.0
            
        # Si la casilla detener al llegar al suelo está activada y t >= t_imp, el estado se fija en reposo
        if d["t_imp"] is not None and self.parar_suelo.get() and t >= d["t_imp"]:
            return float(d["H"]), 0.0

        ts, ys, vs = d["t"], d["y"], d["v"]
        t = min(max(t, ts[0]), ts[-1])
        i = int(np.searchsorted(ts, t, side="right")) - 1
        i = min(max(i, 0), len(ts) - 2)
        h = ts[i + 1] - ts[i]

        if abs(h) < 1e-12:
            return float(ys[i]), float(vs[i])

        s_ = (t - ts[i]) / h
        h00 = 2 * s_ ** 3 - 3 * s_ ** 2 + 1
        h10 = s_ ** 3 - 2 * s_ ** 2 + s_
        h01 = -2 * s_ ** 3 + 3 * s_ ** 2
        h11 = s_ ** 3 - s_ ** 2
        y = h00 * ys[i] + h10 * h * vs[i] + h01 * ys[i + 1] + h11 * h * vs[i + 1]
        v = vs[i] * (1.0 - s_) + vs[i + 1] * s_
        return float(y), float(v)

    def _poner_datos_cursores(self):
        """Coloca los cursores de las gráficas en t_anim."""
        if self.datos is None:
            return
        t = self.t_anim
        y, v = self._estado(t)
        w = self.datos["w"](t)
        self.cur_y.set_xdata([t, t])
        self.cur_v.set_xdata([t, t])
        self.pt_y.set_data([t], [y])
        self.pt_v.set_data([t], [v])
        self.pt_w.set_data([t], [w])

    def _artistas_animados(self):
        return (self.cur_y, self.cur_v, self.pt_y, self.pt_v, self.pt_w)

    def _on_draw(self, _evento):
        """Tras cada redibujado completo: guarda el fondo y pinta los cursores."""
        if not hasattr(self, "cur_y"):
            return
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        for art in self._artistas_animados():
            art.axes.draw_artist(art)

    def _actualizar_cursores(self):
        """Mueve los cursores usando blitting (rápido, no redibuja las curvas)."""
        if not hasattr(self, "cur_y"):
            return
        self._poner_datos_cursores()
        if self._bg is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._bg)
        for art in self._artistas_animados():
            art.axes.draw_artist(art)
        self.canvas.blit(self.fig.bbox)

    def _refrescar_animacion(self):
        """Actualiza animación, cursores y deslizador de tiempo."""
        self._interno = True
        self.var_t.set(self.t_anim)
        self._interno = False
        self._actualizar_cursores()
        self.dibujar_animacion()

    def _mover_tiempo(self, t):
        """Deslizador de tiempo movido por el usuario."""
        if self._interno or self.datos is None:
            return
        self.t_anim = min(max(t, 0.0), self.datos["t_fin"])
        self._actualizar_cursores()
        self.dibujar_animacion()

    def _clic_grafica(self, ev):
        """Clic o arrastre con el botón izquierdo sobre las gráficas = ir a ese t."""
        if self.datos is None or ev.inaxes not in (self.ax_y, self.ax_v):
            return
        if ev.xdata is None or ev.button != 1 or self.barra.mode:
            return
        self.t_anim = min(max(float(ev.xdata), 0.0), self.datos["t_fin"])
        self._refrescar_animacion()

    def alternar_reproduccion(self):
        if self.reproduciendo:
            self.reproduciendo = False
            self.btn_play.configure(text="▶ Reproducir")
            return
        if self.datos is None:
            return
        if self.t_anim >= self.datos["t_fin"] - 1e-9:
            self.t_anim = 0.0
        self.reproduciendo = True
        self._t_ultimo = time.perf_counter()
        self.btn_play.configure(text="⏸ Pausar")
        self._tick()

    def reiniciar_animacion(self):
        self.t_anim = 0.0
        self._refrescar_animacion()

    def _tick(self):
        """Avanza el tiempo de la animación según el reloj real y la velocidad elegida."""
        if not self.reproduciendo or self.datos is None:
            return
        ahora = time.perf_counter()
        dt = ahora - self._t_ultimo
        self._t_ultimo = ahora
        self.t_anim += dt * self.var_vel.get()
        t_fin = self.datos["t_fin"]
        if self.t_anim >= t_fin:
            if self.repetir.get():
                self.t_anim = 0.0
            else:
                self.t_anim = t_fin
                self.reproduciendo = False
                self.btn_play.configure(text="▶ Reproducir")
        self._refrescar_animacion()
        if self.reproduciendo:
            self.after(25, self._tick)

    def dibujar_animacion(self):
        """
        Dibuja el fotograma en t_anim. La cámara sigue al objeto:
          * regla de la izquierda: posición y respecto al suelo (fija en el suelo);
          * rayas azules de la derecha: partículas de AIRE (se mueven con el viento w);
            su movimiento relativo al objeto es la velocidad relativa u = v - w;
          * flechas: peso (rojo) y arrastre (verde); el arrastre iguala al peso en v_t;
          * flecha del viento arriba a la derecha.
        (El cielo, las nubes y el pasto son solo decoración.)
        """
        c = self.lienzo
        c.delete("all")
        if self.datos is None:
            return
        W, Hc = c.winfo_width(), c.winfo_height()
        if W < 80 or Hc < 120:
            return
        fu = self.fuente
        d = self.datos
        t = self.t_anim
        y, v = self._estado(t)
        w = d["w"](t)
        H = d["H"]
        impacto = d["t_imp"] is not None and t >= d["t_imp"] - 1e-9
        if impacto:
            y, v = H, 0.0

        vista = self.var_vista.get()
        esc = Hc / vista                       # píxeles por metro
        cx, cy = int(W * 0.40), int(Hc * 0.38)

        def sy(ym):
            return cy + (ym - y) * esc

        # --- cielo (degradé) ---
        nb = 14
        for i in range(nb):
            c.create_rectangle(0, Hc * i / nb, W, Hc * (i + 1) / nb + 1, outline="",
                               fill=_mezcla(TEMA["cielo_arriba"], TEMA["cielo_abajo"], i / (nb - 1)))

        # --- nubes (fijas al suelo, pasan hacia arriba al caer) ---
        sep = 170.0
        n0 = math.floor((y - cy / esc) / sep) - 1
        n1 = math.ceil((y + (Hc - cy) / esc) / sep) + 1
        for n in range(n0, n1 + 1):
            yn = n * sep + 35.0 * ((n * 7) % 5)
            sc = sy(yn)
            if yn >= H - 30 or sc < 100 or sc > Hc:
                continue
            xn = W * (0.14 + 0.62 * (((n * 37) % 11) / 10.0))
            c.create_oval(xn - 24, sc, xn + 32, sc + 9, fill="#eaf4ff", outline="")
            c.create_oval(xn - 22, sc - 8, xn + 4, sc + 8, fill="#ffffff", outline="")
            c.create_oval(xn - 8, sc - 15, xn + 22, sc + 8, fill="#ffffff", outline="")
            c.create_oval(xn + 6, sc - 6, xn + 30, sc + 8, fill="#ffffff", outline="")

        # --- suelo con pasto ---
        y_suelo = sy(H)
        if y_suelo < Hc:
            c.create_rectangle(0, y_suelo, W, Hc, fill=TEMA["suelo"], outline="")
            c.create_rectangle(0, y_suelo, W, y_suelo + 9, fill=TEMA["pasto"], outline="")
            c.create_line(0, y_suelo, W, y_suelo, fill="#2e7d32", width=2)
            c.create_text(W - 8, min(y_suelo + 24, Hc - 8), anchor="e", fill="#ffffff",
                          text="Suelo (y = %g m)" % H, font=(fu, 8, "bold"))
        # --- línea de salida ---
        y_sal = sy(d["y"][0])
        if 0 < y_sal < Hc:
            c.create_line(0, y_sal, W, y_sal, dash=(4, 3), fill="#5b7088")
            c.create_text(W - 8, y_sal - 8, anchor="e", fill="#43566c",
                          text="salida", font=(fu, 8))

        # --- regla (fija en el suelo) ---
        xr = 40
        c.create_line(xr, 0, xr, Hc, fill="#5b7088")
        paso = 5000
        for p in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000):
            if p * esc >= 48:
                paso = p
                break
        m = math.floor((y - cy / esc) / paso) * paso
        while sy(m) < Hc + 20:
            if sy(m) > -20:
                col = "#ffffff" if sy(m) > y_suelo else "#3d5068"
                c.create_line(xr, sy(m), xr + 9, sy(m), fill=col)
                c.create_text(xr - 4, sy(m), anchor="e", text="%g m" % m,
                              fill=col, font=(fu, 8))
            m += paso

        # --- aire: partículas que se mueven con el viento ---
        u = v - w                                # velocidad relativa objeto-aire
        if self.ver_aire.get():
            Wc = float(np.interp(t, d["tw"], d["Wc"]))   # desplazamiento del aire (m)
            sp = vista / 8.0
            n = int(math.ceil(1.3 * vista / sp))
            span = n * sp
            largo = min(max(abs(u) * 0.8, 3.0), 38.0)
            sentido = -1 if u >= 0 else 1        # el aire pasa hacia arriba si u > 0
            for k_, fx in enumerate((0.70, 0.82, 0.94)):
                xx = W * fx
                for j in range(n):
                    pos = j * sp + k_ * sp * 0.37 + Wc
                    rel = (pos - y + 0.45 * vista) % span - 0.45 * vista
                    yy = cy + rel * esc
                    if -40 < yy < Hc + 40:
                        c.create_line(xx, yy, xx, yy + sentido * largo, fill=TEMA["aire"],
                                      width=2, capstyle="round",
                                      arrow=tk.LAST if largo > 6 else None,
                                      arrowshape=(7, 8, 3))

        # --- fuerzas ---
        if self.ver_fuerzas.get() and not impacto:
            Lp = 52.0
            xf = cx + 34
            c.create_line(xf, cy, xf, cy + Lp, fill=TEMA["peso"], width=4, arrow=tk.LAST,
                          arrowshape=(10, 12, 4), capstyle="round")
            frac = d["k"] * abs(u) * u / d["g"]          # arrastre / peso (+ = hacia arriba)
            Ld = max(-130.0, min(130.0, frac * Lp))
            if abs(Ld) > 2:
                c.create_line(cx + 60, cy, cx + 60, cy - Ld, fill=TEMA["arrastre"], width=4,
                              arrow=tk.LAST, arrowshape=(10, 12, 4), capstyle="round")
            c.create_oval(10, Hc - 29, 18, Hc - 21, fill=TEMA["peso"], outline="")
            c.create_text(24, Hc - 25, anchor="w", fill="#1f2937",
                          text="peso (m g)", font=(fu, 8, "bold"))
            c.create_oval(10, Hc - 15, 18, Hc - 7, fill=TEMA["arrastre"], outline="")
            c.create_text(24, Hc - 11, anchor="w", fill="#1f2937",
                          text="arrastre (c |u| u)", font=(fu, 8, "bold"))

        # --- objeto (con líneas de velocidad) ---
        r = 13
        yc = cy - r if impacto else cy
        if abs(v) > 3 and not impacto:
            Lv = min(abs(v) * 0.7, 30.0)
            dirn = -1 if v > 0 else 1
            for dx in (-7, 0, 7):
                c.create_line(cx + dx, cy + dirn * (r + 4),
                              cx + dx, cy + dirn * (r + 4 + Lv - abs(dx) * 1.2),
                              fill="#475569", width=2, capstyle="round")
        c.create_oval(cx - r, yc - r, cx + r, yc + r, fill=TEMA["acento"],
                      outline="#1e3a8a", width=2)
        c.create_oval(cx - 7, yc - 8, cx - 1, yc - 3, fill="#bfdbfe", outline="")
        if impacto:
            tid = c.create_text(cx, yc - 30, text="¡IMPACTO!  v = %.1f m/s" % d["v_imp"],
                                fill=TEMA["peso"], font=(fu, 10, "bold"))
            x1, y1, x2, y2 = c.bbox(tid)
            c.tag_lower(c.create_rectangle(x1 - 6, y1 - 3, x2 + 6, y2 + 3, fill="#ffffff",
                                           outline=TEMA["peso"]), tid)

        # --- tarjeta de datos (arriba) ---
        c.create_rectangle(4, 4, W - 4, 92, fill="#ffffff", outline=TEMA["borde"])
        if abs(w) < 0.05:
            txt_w = "sin viento"
        else:
            txt_w = "baja" if w > 0 else "sube"
        c.create_text(12, 9, anchor="nw", font=("Courier", 9), fill=TEMA["texto"],
                      text="t = %6.2f s\ny = %7.1f m\nv = %6.2f m/s\n"
                           "w = %6.2f m/s (%s)\nv - w = %6.2f m/s"
                           % (t, y, v, w, txt_w, u))
        # flecha del viento
        xw = W - 32
        c.create_text(xw, 82, text="viento", font=(fu, 8, "bold"), fill=TEMA["texto_suave"])
        if abs(w) < 0.3:
            c.create_oval(xw - 3, 46, xw + 3, 52, fill=TEMA["viento"], outline="")
        else:
            largo_w = max(-30.0, min(30.0, w * 1.5))
            c.create_line(xw, 49, xw, 49 + largo_w, fill=TEMA["viento"], width=5,
                          arrow=tk.LAST, arrowshape=(11, 12, 5), capstyle="round")

    # ------------------------------------------------------------------
    # Ventana de validación
    # ------------------------------------------------------------------
    def abrir_validacion(self):
        win = tk.Toplevel(self)
        win.title("Validación: RK4 propio vs solución analítica (sin viento)")
        win.geometry("1000x720")
        win.configure(bg=TEMA["fondo"])
        fig = Figure(figsize=(10, 4.2), dpi=100, facecolor="#ffffff")
        filas = validacion.tabla_convergencia(("ninguno",), 0.0, 0.0)
        validacion.figura_validacion(filas, fig=fig)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw()
        ordenes = [f[4] for f in filas if f[4] is not None]
        texto = ("Caso: w = 0, y0 = 0, v0 = 0, T = %.0f s  "
                 "(m=75 kg, A=0.7 m², Cd=1, ρ=1.225 kg/m³, g=9.81 m/s²)\n\n" % validacion.T_VALID
                 + validacion.texto_tabla(filas)
                 + "\n\nOrden observado promedio: %.2f  (teórico: 4; razón teórica: 16)"
                 % (sum(ordenes) / len(ordenes)))
        caja = tk.Text(win, height=12, font=("Courier", 10), bg=TEMA["tarjeta"],
                       fg=TEMA["texto"], relief="flat", padx=12, pady=8,
                       highlightthickness=1, highlightbackground=TEMA["borde"])
        caja.insert("1.0", texto)
        caja.configure(state="disabled")
        caja.pack(side="bottom", fill="x")


if __name__ == "__main__":
    Simulador().mainloop()