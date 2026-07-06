import sys
import numpy as np
from scipy.integrate import cumulative_trapezoid
import sympy as sp
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLineEdit, QPushButton, QComboBox,
    QVBoxLayout, QLabel
)
from PyQt5.QtCore import Qt
import plotly.graph_objects as go
from PyQt5.QtWidgets import QRadioButton

# =========================
# Символы и функции
# =========================
x_sym, t_sym = sp.symbols('x t')






def parse_function(expr, vars_):
    sym_expr = sp.sympify(expr)
    return sp.lambdify(vars_, sym_expr, modules=['numpy'])

# Начальная функция по умолчанию
def phi0(z):
    return np.exp(-z**2)

# =========================
# Решатели
# =========================
def solve_const_a(a, xgrid, tgrid, phi=phi0):
    X, T = np.meshgrid(xgrid, tgrid)
    return phi(X - a*T)

def solve_ax(expr, xgrid, tgrid, phi=phi0):
    a = parse_function(expr, x_sym)
    a_vals = a(xgrid)
    f_vals = cumulative_trapezoid(1/a_vals, xgrid, initial=0)
    X, T = np.meshgrid(xgrid, tgrid)
    return phi(T - f_vals[np.newaxis, :])

def solve_at(expr, xgrid, tgrid, phi=phi0):
    a = parse_function(expr, t_sym)
    a_vals = a(tgrid)
    s_vals = cumulative_trapezoid(a_vals, tgrid, initial=0)
    X, T = np.meshgrid(xgrid, tgrid)
    return phi(X - s_vals[:, np.newaxis])

def solve_axt_catalog(case, xgrid, tgrid, phi=phi0, alpha=1.0, beta=0.0, d=0.0):
    X, T = np.meshgrid(xgrid, tgrid)
    X_safe = np.maximum(X, 1e-8)

    if case == "a = x - t":
        xi = np.exp(-T) * (X - T - 1)
    elif case == "a = x + t":
        xi = np.exp(-T) * (X + T + 1)
    elif case == "a = x * t":
        xi = 0.5 * T**2 - np.log(X_safe)
    elif case == "a = x * exp(-t)":
        xi = -np.exp(-T) - np.log(X_safe)
    elif case == "a = α x + β t":
        xi = np.exp(-alpha*T)*(alpha*X + beta*T + beta/alpha)
    elif case == "a = α x + β t + d":
        xi = np.exp(-alpha*T)*(alpha**2*X + alpha*beta*T + beta + d*alpha)/(alpha**2)
    else:
        return None

    return phi(xi)

# =========================
# Графики
# =========================
def plot_contour(xgrid, tgrid, U):
    fig = go.Figure(data=go.Contour(x=xgrid, y=tgrid, z=U, colorscale='Viridis'))
    fig.update_layout(title="Контурный график u(x,t)")
    fig.show()

def plot_3d(xgrid, tgrid, U):
    fig = go.Figure(data=go.Surface(x=xgrid, y=tgrid, z=U, colorscale='Viridis'))
    fig.update_layout(title="3D график u(x,t)")
    fig.show()

def animate_solution(xgrid, tgrid, U):
    frames = [
        go.Frame(
            data=[go.Scatter(x=xgrid, y=U[i], mode="lines")],
            name=str(i)
        )
        for i in range(len(tgrid))
    ]

    # Диапазоны осей
    y_min, y_max = np.min(U), np.max(U)

    fig = go.Figure(
        data=[go.Scatter(x=xgrid, y=U[0], mode="lines")],
        layout=go.Layout(
            title="Анимация u(x,t)",
            xaxis=dict(range=[xgrid.min(), xgrid.max()]),
            yaxis=dict(range=[y_min, y_max]),
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Loop",
                     "method": "animate",
                     "args": [None, {"frame": {"duration": 60}, "fromcurrent": True, "loop": True}]},
                    {"label": "⏹ Stop",
                     "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "y": 1.1
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"prefix": "t = "},
                "pad": {"b": 10, "t": 50},
                "steps": [
                    {"method": "animate", "args": [[str(i)], {"mode": "immediate"}],
                     "label": f"{tgrid[i]:.2f}"}
                    for i in range(len(tgrid))
                ]
            }]
        ),
        frames=frames
    )

    fig.show()


# =========================
# GUI
# =========================
class TransportApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Уравнение переноса")
        self.setGeometry(200, 100, 500, 600)

        main_layout = QVBoxLayout()

        # Тип коэффициента
        self.combo = QComboBox()
        self.combo.addItems(["a = const", "a = a(x)", "a = a(t)", "a = a(x,t)"])
        self.combo.setToolTip("Выберите тип коэффициента a")
        main_layout.addWidget(self.combo)

        # Поле для формулы a
        self.a_input = QLineEdit()
        self.a_input.setPlaceholderText("Введите a (например, 1, sin(x), ...)")
        main_layout.addWidget(self.a_input)

        # Модель a(x,t) и α,β,d
        self.combo_axt = QComboBox()
        self.combo_axt.addItems([
            "a = x - t", "a = x + t", "a = x * t",
            "a = x * exp(-t)", "a = α x + β t", "a = α x + β t + d"
        ])
        self.combo_axt.setVisible(False)
        main_layout.addWidget(self.combo_axt)

        self.alpha_input = QLineEdit()
        self.alpha_input.setPlaceholderText("α")
        self.alpha_input.setVisible(False)
        main_layout.addWidget(self.alpha_input)

        self.beta_input = QLineEdit()
        self.beta_input.setPlaceholderText("β")
        self.beta_input.setVisible(False)
        main_layout.addWidget(self.beta_input)

        self.d_input = QLineEdit()
        self.d_input.setPlaceholderText("d")
        self.d_input.setVisible(False)
        main_layout.addWidget(self.d_input)

        # Диапазоны
        self.xmin = QLineEdit()
        self.xmin.setPlaceholderText("x_min")
        main_layout.addWidget(self.xmin)
        self.xmax = QLineEdit()
        self.xmax.setPlaceholderText("x_max")
        main_layout.addWidget(self.xmax)
        self.tmin = QLineEdit()
        self.tmin.setPlaceholderText("t_min")
        main_layout.addWidget(self.tmin)
        self.tmax = QLineEdit()
        self.tmax.setPlaceholderText("t_max")
        main_layout.addWidget(self.tmax)

        # Поле для φ₀(x)
        self.phi_input = QLineEdit()
        self.phi_input.setPlaceholderText("Введите φ₀(x) для анимации (например: exp(-x**2))")
        main_layout.addWidget(self.phi_input)

        # Кнопка построения графиков
        btn = QPushButton("Построить графики")
        btn.setStyleSheet("font-size: 16px; padding: 8px;")
        btn.clicked.connect(self.run)
        main_layout.addWidget(btn)

        self.setLayout(main_layout)
        self.combo.currentTextChanged.connect(self.update_ui)

    def update_ui(self):
        mode = self.combo.currentText()
        is_axt = (mode == "a = a(x,t)")
        self.a_input.setVisible(not is_axt)
        self.combo_axt.setVisible(is_axt)
        self.alpha_input.setVisible(is_axt)
        self.beta_input.setVisible(is_axt)
        self.d_input.setVisible(is_axt)

    def run(self):
        try:
            xgrid = np.linspace(float(self.xmin.text()), float(self.xmax.text()), 200)
            tgrid = np.linspace(float(self.tmin.text()), float(self.tmax.text()), 200)
        except:
            print("Введите корректные диапазоны!")
            return

        mode = self.combo.currentText()

        # Основная функция φ₀ для графиков
        phi_graph = phi0

        if mode == "a = const":
            U = solve_const_a(float(self.a_input.text()), xgrid, tgrid, phi=phi_graph)
        elif mode == "a = a(x)":
            U = solve_ax(self.a_input.text(), xgrid, tgrid, phi=phi_graph)
        elif mode == "a = a(t)":
            U = solve_at(self.a_input.text(), xgrid, tgrid, phi=phi_graph)
        else:
            try:
                alpha = float(self.alpha_input.text())
                beta = float(self.beta_input.text())
                d = float(self.d_input.text())
            except:
                alpha = beta = d = 0.0
            U = solve_axt_catalog(self.combo_axt.currentText(), xgrid, tgrid, phi=phi_graph, alpha=alpha, beta=beta, d=d)
            if U is None:
                print("Ошибка в расчетах a(x,t)")
                return

        # Контур и 3D графики
        plot_contour(xgrid, tgrid, U)
        plot_3d(xgrid, tgrid, U)

        # Анимация
        phi_expr = self.phi_input.text()
        if phi_expr.strip() == "":
            phi_func = phi0
        else:
            try:
                # Поддержка UnitStep / heaviside
                phi_func = sp.lambdify(
                    x_sym, 
                    sp.sympify(phi_expr),
                    modules=[{"heaviside": lambda x: np.heaviside(x, 1)}, "numpy"]
                )
            except Exception as e:
                print("Ошибка в φ₀(x):", e)
                return

        if mode == "a = const":
            U_anim = solve_const_a(float(self.a_input.text()), xgrid, tgrid, phi=phi_func)
        elif mode == "a = a(x)":
            U_anim = solve_ax(self.a_input.text(), xgrid, tgrid, phi=phi_func)
        elif mode == "a = a(t)":
            U_anim = solve_at(self.a_input.text(), xgrid, tgrid, phi=phi_func)
        else:
            U_anim = solve_axt_catalog(
                self.combo_axt.currentText(),
                xgrid, tgrid,
                phi=phi_func,
                alpha=alpha, beta=beta, d=d
            )

        animate_solution(xgrid, tgrid, U_anim)

# =========================
# Запуск приложения
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TransportApp()
    win.show()
    sys.exit(app.exec_())
