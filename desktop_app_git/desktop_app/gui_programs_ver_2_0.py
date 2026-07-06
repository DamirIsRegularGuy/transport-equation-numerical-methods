import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List

if sys.platform.startswith("linux") and not os.environ.get("QT_QPA_PLATFORM"):
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        # On Ubuntu/GNOME, forcing xcb avoids Wayland-specific Qt warnings and plugin issues.
        os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
import sympy as sp
from scipy.integrate import cumulative_trapezoid
import scipy.linalg as sla

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAVE_PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    HAVE_PLOTLY = False

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

x_sym, t_sym = sp.symbols("x t")


def ensure_plotly():
    if not HAVE_PLOTLY:
        raise RuntimeError("Нужен plotly: pip install plotly")


def parse_function(expr: str, vars_):
    return sp.lambdify(
        vars_,
        sp.sympify(expr),
        modules=[{"heaviside": lambda z: np.heaviside(z, 1.0)}, "numpy"],
    )


def phi0(z):
    return np.exp(-(np.asarray(z, dtype=float) ** 2))


def iinterp(xg, yg, q):
    q = np.asarray(q, dtype=float)
    return np.interp(q.ravel(), xg, yg).reshape(q.shape)


def inv_mono_interp(xg, yg, q):
    q = np.asarray(q, dtype=float)
    if yg[0] <= yg[-1]:
        yy, xx = yg, xg
    else:
        yy, xx = yg[::-1], xg[::-1]
    qf = q.ravel()
    out = np.interp(qf, yy, xx)
    if len(yy) >= 2:
        dyl = yy[1] - yy[0]
        dyr = yy[-1] - yy[-2]
        sl = 0.0 if abs(dyl) < 1e-14 else (xx[1] - xx[0]) / dyl
        sr = 0.0 if abs(dyr) < 1e-14 else (xx[-1] - xx[-2]) / dyr
        ml = qf < yy[0]
        mr = qf > yy[-1]
        out[ml] = xx[0] + (qf[ml] - yy[0]) * sl
        out[mr] = xx[-1] + (qf[mr] - yy[-1]) * sr
    return out.reshape(q.shape)


@dataclass
class Ctx:
    x: np.ndarray
    t: np.ndarray
    a: Callable[[np.ndarray, float], np.ndarray]
    u: Callable[[np.ndarray, np.ndarray], np.ndarray]
    const: bool
    aval: float


def make_ctx(mode, aexpr, axt, x, t, phi, alpha=0.0, beta=0.0, d=0.0):
    if mode == "a = const":
        a0 = float(aexpr)
        af = lambda xx, tt: np.full_like(np.asarray(xx, dtype=float), a0, dtype=float)
        uf = lambda xx, tt: phi(np.asarray(xx, dtype=float) - a0 * np.asarray(tt, dtype=float))
        return Ctx(x, t, af, uf, True, a0)
    if mode == "a = a(x)":
        ax = parse_function(aexpr, x_sym)
        av = np.asarray(ax(x), dtype=float)
        if np.any(~np.isfinite(av)):
            raise ValueError("Для a(x) получены нечисловые значения на сетке")
        if np.any(np.isclose(av, 0)):
            raise ValueError("Для a(x) есть точки с a = 0")
        if not (np.all(av > 0) or np.all(av < 0)):
            raise ValueError("Для a(x) коэффициент должен сохранять знак на сетке")
        fv = cumulative_trapezoid(1.0 / av, x, initial=0.0)
        af = lambda xx, tt: np.asarray(ax(np.asarray(xx, dtype=float)), dtype=float)
        def uf(xx, tt):
            xx = np.asarray(xx, dtype=float)
            tt = np.asarray(tt, dtype=float)
            fxx = iinterp(x, fv, xx)
            xi = inv_mono_interp(x, fv, fxx - tt)
            return phi(xi)
        return Ctx(x, t, af, uf, False, 0.0)
    if mode == "a = a(t)":
        at = parse_function(aexpr, t_sym)
        av = np.asarray(at(t), dtype=float)
        sv = cumulative_trapezoid(av, t, initial=0.0)
        def af(xx, tt):
            xx = np.asarray(xx, dtype=float)
            tt = np.asarray(tt, dtype=float)
            return np.ones_like(xx, dtype=float) * np.asarray(at(tt), dtype=float)
        uf = lambda xx, tt: phi(np.asarray(xx, dtype=float) - iinterp(t, sv, tt))
        return Ctx(x, t, af, uf, False, 0.0)

    def af(xx, tt):
        xx = np.asarray(xx, dtype=float)
        tt = np.asarray(tt, dtype=float)
        if xx.ndim == 1 and tt.ndim == 1:
            xx = xx[None, :]
            tt = tt[:, None]
        if axt == "a = x - t":
            return xx - tt
        if axt == "a = x + t":
            return xx + tt
        if axt == "a = x * t":
            return xx * tt
        if axt == "a = x * exp(-t)":
            return xx * np.exp(-tt)
        if axt == "a = alpha*x + beta*t":
            return alpha * xx + beta * tt
        return alpha * xx + beta * tt + d

    def uf(xx, tt):
        xx = np.asarray(xx, dtype=float)
        tt = np.asarray(tt, dtype=float)
        if axt == "a = x - t":
            xi = np.exp(-tt) * (xx - tt)
        elif axt == "a = x + t":
            xi = np.exp(-tt) * (xx + tt)
        elif axt == "a = x * t":
            xi = 0.5 * tt**2 - np.log(xx)
        elif axt == "a = x * exp(-t)":
            xi = -np.exp(-tt) - np.log(xx)
        else:
            c0 = 0.0 if axt == "a = alpha*x + beta*t" else d
            if abs(alpha) < 1e-12:
                xi = xx - 0.5 * beta * tt**2 - c0 * tt
            else:
                ea = np.exp(-alpha * tt)
                shift = beta / alpha + c0 / alpha
                xi = ea * (alpha * xx + beta * tt + shift) - shift
        return phi(xi)

    return Ctx(x, t, af, uf, False, 0.0)


def dxdt(ctx):
    return float(ctx.x[1] - ctx.x[0]), float(ctx.t[1] - ctx.t[0])


def u0(ctx):
    U = np.zeros((len(ctx.t), len(ctx.x)), dtype=float)
    U[0] = np.asarray(ctx.u(ctx.x, ctx.t[0]), dtype=float)
    return U


def us(ctx, xv, tv):
    return float(np.asarray(ctx.u(np.array([xv]), np.array([tv])), dtype=float)[0])


def tdma(l, d, u, r):
    n = len(d)
    l, d, u, r = np.array(l, float), np.array(d, float), np.array(u, float), np.array(r, float)
    for i in range(1, n):
        if abs(d[i - 1]) < 1e-14:
            raise ValueError("Сингулярная матрица в методе прогонки")
        m = l[i - 1] / d[i - 1]
        d[i] -= m * u[i - 1]
        r[i] -= m * r[i - 1]
    x = np.zeros(n)
    x[-1] = r[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (r[i] - u[i] * x[i + 1]) / d[i]
    return x


def bnd(u_next, ctx, tnext):
    u_next[0] = us(ctx, float(ctx.x[0]), tnext)
    u_next[-1] = us(ctx, float(ctx.x[-1]), tnext)


def lim_minmod(r):
    return max(0.0, min(1.0, r))


def lim_mc(r):
    return max(0.0, min(2.0 * r, 1.0), min(r, 2.0))


def s_up(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        c = (dt / dx) * ctx.a(ctx.x, float(ctx.t[n])); p, q = U[n], U[n + 1]
        for i in range(1, nx - 1):
            if c[i] >= 0:
                q[i] = p[i] - c[i] * (p[i] - p[i - 1])
            else:
                q[i] = p[i] - c[i] * (p[i + 1] - p[i])
        bnd(q, ctx, float(ctx.t[n + 1]))
    return U


def s_lf(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        c = (dt / dx) * ctx.a(ctx.x, float(ctx.t[n])); p, q = U[n], U[n + 1]
        for i in range(1, nx - 1):
            q[i] = 0.5 * (p[i + 1] + p[i - 1]) - 0.5 * c[i] * (p[i + 1] - p[i - 1])
        bnd(q, ctx, float(ctx.t[n + 1]))
    return U


def s_lw(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        c = (dt / dx) * ctx.a(ctx.x, float(ctx.t[n])); p, q = U[n], U[n + 1]
        for i in range(1, nx - 1):
            q[i] = p[i] - 0.5 * c[i] * (p[i + 1] - p[i - 1]) + 0.5 * c[i] ** 2 * (p[i + 1] - 2 * p[i] + p[i - 1])
        bnd(q, ctx, float(ctx.t[n + 1]))
    return U


def s_mc(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        c = (dt / dx) * ctx.a(ctx.x, float(ctx.t[n])); p, q = U[n], U[n + 1]; s = p.copy()
        for i in range(1, nx - 1):
            s[i] = p[i] - c[i] * (p[i + 1] - p[i])
        for i in range(1, nx - 1):
            q[i] = 0.5 * (p[i] + s[i] - c[i] * (s[i] - s[i - 1]))
        bnd(q, ctx, float(ctx.t[n + 1]))
    return U


def s_visc(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        p, q = U[n], U[n + 1]
        a_vals = ctx.a(ctx.x, float(ctx.t[n]))
        for i in range(1, nx - 1):
            central = -0.5 * a_vals[i] * (dt / dx) * (p[i + 1] - p[i - 1])
            diff = p[i + 1] - 2.0 * p[i] + p[i - 1]
            if ctx.const:
                visc = 0.5 * diff
                q[i] = p[i] + central + dt * visc
            else:
                visc = (dx ** 2 / (2 * dt)) * diff
                q[i] = p[i] + central + visc
        bnd(q, ctx, float(ctx.t[n + 1]))
    return U


def s_rusanov(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        p, q = U[n], U[n + 1]
        a_vals = ctx.a(ctx.x, float(ctx.t[n]))
        for i in range(1, nx - 1):
            if a_vals[i] >= 0:
                flux_r = a_vals[i] * p[i]
                flux_l = a_vals[i] * p[i - 1]
            else:
                flux_r = a_vals[i] * p[i + 1]
                flux_l = a_vals[i] * p[i]
            q[i] = p[i] - (dt / dx) * (flux_r - flux_l)
        bnd(q, ctx, float(ctx.t[n + 1]))
    return U


def s_godunov(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        p, q = U[n], U[n + 1]
        a_vals = ctx.a(ctx.x, float(ctx.t[n]))
        for i in range(1, nx - 1):
            if a_vals[i] >= 0:
                flux_r = a_vals[i] * p[i]
                flux_l = a_vals[i] * p[i - 1]
            else:
                flux_r = a_vals[i] * p[i + 1]
                flux_l = a_vals[i] * p[i]
            q[i] = p[i] - (dt / dx) * (flux_r - flux_l)
        bnd(q, ctx, float(ctx.t[n + 1]))
    return U


def _s_tvd_like(ctx, limiter):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        p, q = U[n], U[n + 1]
        c = (dt / dx) * ctx.a(ctx.x, float(ctx.t[n]))
        for i in range(2, nx - 2):
            ci = c[i]
            if ci >= 0:
                delm = p[i] - p[i - 1]
                dp = p[i + 1] - p[i]
                dml = p[i - 1] - p[i - 2]
                dpl = delm
            else:
                delm = p[i + 1] - p[i]
                dp = p[i] - p[i - 1]
                dml = p[i] - p[i - 1]
                dpl = delm
            r = 0.0 if abs(dp) < 1e-14 else delm / dp
            rl = 0.0 if abs(dpl) < 1e-14 else dml / dpl
            ph, phl = limiter(r), limiter(rl)
            corr = ph * dp - phl * delm
            coeff = 0.5 * abs(ci) * max(0.0, 1.0 - abs(ci))
            q[i] = p[i] - ci * delm - coeff * corr
        bnd(q, ctx, float(ctx.t[n + 1]))
        q[1] = q[2]
        q[-2] = q[-3]
    return U


def s_tvd(ctx):
    return _s_tvd_like(ctx, lim_minmod)


def s_muscl(ctx):
    return _s_tvd_like(ctx, lim_mc)


def simpl(ctx, mode):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx)
    for n in range(len(ctx.t) - 1):
        tn = float(ctx.t[n + 1]); c = (dt / dx) * ctx.a(ctx.x, tn); av = ctx.a(ctx.x, tn)
        l, d, u, r = np.zeros(nx - 1), np.ones(nx), np.zeros(nx - 1), U[n].copy()
        for i in range(1, nx - 1):
            if mode == "314":
                if c[i] >= 0: d[i], l[i - 1] = 1 + c[i], -c[i]
                else: d[i], u[i] = 1 - c[i], c[i]
            elif mode == "lw":
                d[i], l[i - 1], u[i] = 1 + c[i] ** 2, -0.5 * c[i] - 0.5 * c[i] ** 2, 0.5 * c[i] - 0.5 * c[i] ** 2
            elif mode == "cn":
                if c[i] >= 0:
                    d[i], l[i - 1], r[i] = 1 + 0.5 * c[i], -0.5 * c[i], (1 - 0.5 * c[i]) * U[n, i] + 0.5 * c[i] * U[n, i - 1]
                else:
                    d[i], u[i], r[i] = 1 - 0.5 * c[i], 0.5 * c[i], (1 + 0.5 * c[i]) * U[n, i] - 0.5 * c[i] * U[n, i + 1]
            elif mode == "317":
                a, b = (1.0 if c[i] < 0 else 0.0), (1.0 if c[i] >= 0 else 0.0)
                d[i], l[i - 1], u[i] = 1 + abs(c[i]), -b * c[i], a * c[i]
            elif mode == "320":
                w = 0.2; d[i], l[i - 1], u[i] = 1 + 2 * w, -0.5 * c[i] - w, 0.5 * c[i] - w
            elif mode == "319r":
                k = 0.5 * dt * av[i] ** 2 / (dx ** 2); d[i], l[i - 1], u[i] = 1 + 2 * k, -0.5 * c[i] - k, 0.5 * c[i] - k
            elif mode == "gen":
                d[i], l[i - 1], u[i] = 1.0, -0.5 * c[i], 0.5 * c[i]
        d[0] = d[-1] = 1.0; u[0] = 0.0; l[-1] = 0.0; r[0], r[-1] = us(ctx, float(ctx.x[0]), tn), us(ctx, float(ctx.x[-1]), tn)
        U[n + 1] = tdma(l, d, u, r)
    return U


def s_318(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx); w = 0.3
    for n in range(len(ctx.t) - 1):
        tn = float(ctx.t[n + 1]); c = (dt / dx) * ctx.a(ctx.x, tn); A = np.zeros((nx, nx)); r = U[n].copy()
        for i in range(nx):
            A[i, i] = 1 + 6 * w * dt / 8
            if i + 1 < nx: A[i, i + 1] = 0.5 * c[i] - 4 * w * dt / 8
            if i - 1 >= 0: A[i, i - 1] = -0.5 * c[i] - 4 * w * dt / 8
            if i + 2 < nx: A[i, i + 2] = w * dt / 8
            if i - 2 >= 0: A[i, i - 2] = w * dt / 8
        A[0, :], A[-1, :] = 0.0, 0.0; A[0, 0], A[-1, -1] = 1.0, 1.0
        r[0], r[-1] = us(ctx, float(ctx.x[0]), tn), us(ctx, float(ctx.x[-1]), tn)
        U[n + 1] = np.linalg.solve(A, r)
    return U


def s_321(ctx):
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx); w = 0.5
    for n in range(len(ctx.t) - 1):
        tn = float(ctx.t[n + 1]); c = (dt / dx) * ctx.a(ctx.x, tn); a = ctx.a(ctx.x, tn); p = U[n]
        ux = (np.roll(p, -1) - np.roll(p, 1)) / (2 * dx); k = w * dt * (a ** 2) * np.abs(ux) / (dx ** 2)
        l, d, u, r = np.zeros(nx - 1), np.ones(nx), np.zeros(nx - 1), p.copy()
        for i in range(1, nx - 1):
            d[i], l[i - 1], u[i] = 1 + 2 * k[i], -0.5 * c[i] - k[i], 0.5 * c[i] - k[i]
        d[0] = d[-1] = 1.0; r[0], r[-1] = us(ctx, float(ctx.x[0]), tn), us(ctx, float(ctx.x[-1]), tn)
        U[n + 1] = tdma(l, d, u, r)
    return U


def s_315(ctx):
    if not ctx.const: raise ValueError("Доступно только для a = const")
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx); c = ctx.aval * dt / dx
    l0, d0, u0_ = np.full(nx - 1, -3 * c / 4), np.full(nx, 1 + c / 2), np.full(nx - 1, c / 4)
    for n in range(len(ctx.t) - 1):
        l, d, u, r = l0.copy(), d0.copy(), u0_.copy(), U[n].copy()
        d[0] = d[-1] = 1.0; u[0] = 0.0; l[-1] = 0.0
        r[0], r[-1] = us(ctx, float(ctx.x[0]), float(ctx.t[n + 1])), us(ctx, float(ctx.x[-1]), float(ctx.t[n + 1]))
        U[n + 1] = tdma(l, d, u, r)
    return U


def s_316(ctx):
    if not ctx.const: raise ValueError("Доступно только для a = const")
    U = u0(ctx); nx = len(ctx.x); dx, dt = dxdt(ctx); c = ctx.aval * dt / dx
    l0, d0 = np.full(nx - 1, -0.5 * c), np.full(nx, 1 + 0.5 * c)
    for n in range(len(ctx.t) - 1):
        r = np.zeros(nx); r[1:-1] = (1 + 0.5 * c) * U[n, 1:-1] - 0.5 * c * U[n, 2:]
        r[0], r[-1] = us(ctx, float(ctx.x[0]), float(ctx.t[n + 1])), us(ctx, float(ctx.x[-1]), float(ctx.t[n + 1]))
        l, d, u = l0.copy(), d0.copy(), np.zeros(nx - 1); d[0] = d[-1] = 1.0; l[-1] = 0.0
        U[n + 1] = tdma(l, d, u, r)
    return U


EXACT_NAME = "Аналитическое решение"

EXP = OrderedDict({
    "Явная против потока": s_up,
    "Явная Лакс-Фридрихс": s_lf,
    "Явная Лакс-Вендрофф": s_lw,
    "Явная вязкость": s_visc,
    "Явная Мак-Кормак": s_mc,
    "Явная TVD (minmod)": s_tvd,
    "Явная MUSCL (MC)": s_muscl,
    "Явная Русанов": s_rusanov,
    "Явная Годунов": s_godunov,
})
IMP = OrderedDict({
    "Неявная против потока (3.14)": lambda c: simpl(c, "314"),
    "Неявная центральная gen": lambda c: simpl(c, "gen"),
    "Неявная Лакс-Вендрофф (3.19)": lambda c: simpl(c, "lw"),
    "Кранк-Николсон (3.15)": lambda c: simpl(c, "cn"),
    "Неявная апвинд (3.17)": lambda c: simpl(c, "317"),
    "Линейная вязкость (3.20)": lambda c: simpl(c, "320"),
    "Вязкость 4-го порядка (3.18)": s_318,
    "Нелинейная вязкость (3.21)": s_321,
    "Регуляриз. Лакс-Вендрофф (3.19)": lambda c: simpl(c, "319r"),
})
COL = {
    EXACT_NAME: "#ffd700",
    "Явная против потока": "#d62728",
    "Явная Лакс-Фридрихс": "#1f77b4",
    "Явная Лакс-Вендрофф": "#9467bd",
    "Явная вязкость": "#8c564b",
    "Явная Мак-Кормак": "#17becf",
    "Явная TVD (minmod)": "#7f7f7f",
    "Явная MUSCL (MC)": "#ff7f0e",
    "Явная Русанов": "#2ca02c",
    "Явная Годунов": "#111111",
    "Неявная против потока (3.14)": "#ff4d4d",
    "Неявная центральная gen": "#4f9dff",
    "Неявная Лакс-Вендрофф (3.19)": "#cc00ff",
    "Кранк-Николсон (3.15)": "#ff9900",
    "Неявная апвинд (3.17)": "#2ca02c",
    "Линейная вязкость (3.20)": "#8c564b",
    "Вязкость 4-го порядка (3.18)": "#00c8ff",
    "Нелинейная вязкость (3.21)": "#7f3fbf",
    "Регуляриз. Лакс-Вендрофф (3.19)": "#8b4513",
}

USER_SCHEMES = {}  # name: (code, type)

SCHEME_FORMULAS = {
    "Явная против потока": "u_i^{n+1} = u_i^n - λ (u_i^n - u_{i-1}^n)",
    "Явная Лакс-Фридрихс": "u_i^{n+1} = 0.5 (u_{i+1}^n + u_{i-1}^n) - 0.5 λ (u_{i+1}^n - u_{i-1}^n)",
    "Явная Лакс-Вендрофф": "u_i^{n+1} = u_i^n - 0.5 λ (u_{i+1}^n - u_{i-1}^n) + 0.5 λ² (u_{i+1}^n - 2u_i^n + u_{i-1}^n)",
    "Явная вязкость": "u_i^{n+1} = u_i^n - 0.5 λ (u_{i+1}^n - u_{i-1}^n) + τ * 0.5 (u_{i+1}^n - 2u_i^n + u_{i-1}^n)",
    "Явная Мак-Кормак": "Predictor: u_i^* = u_i^n - λ (u_{i+1}^n - u_i^n)\nCorrector: u_i^{n+1} = 0.5 (u_i^n + u_i^* - λ (u_i^* - u_{i-1}^*))",
    "Явная TVD (minmod)": "u_i^{n+1} = u_i^n - λ Δu_i - (λ(1-λ)/2) * (φ(r) Δu_{i+1} - φ(r_l) Δu_i)",
    "Явная MUSCL (MC)": "Аналогично TVD с limiter MC: φ(r) = max(0, min(2r, 1), min(r, 2))",
    "Явная Русанов": "u_i^{n+1} = u_i^n - (τ/h) (a u_i^n - a u_{i-1}^n)",
    "Явная Годунов": "Аналогично Русанов для a > 0",
    "Неявная против потока (3.14)": "Матрица: диагональ 1+λ, поддиагональ -λ",
    "Неявная центральная gen": "Матрица: диагональ 1, поддиагональ -λ/2, наддиагональ λ/2",
    "Неявная Лакс-Вендрофф (3.19)": "Диагональ 1+λ², поддиагональ -λ/2 - λ²/2, наддиагональ λ/2 - λ²/2",
    "Кранк-Николсон (3.15)": "Диагональ 1+λ/2, поддиагональ -λ/2",
    "Неявная апвинд (3.17)": "Диагональ 1+|λ|, поддиагональ -λ H(-λ), наддиагональ λ H(λ)",
    "Линейная вязкость (3.20)": "Диагональ 1+2ω, поддиагональ -λ/2 - ω, наддиагональ λ/2 - ω",
    "Вязкость 4-го порядка (3.18)": "Пентодиагональная матрица с искусственной вязкостью",
    "Нелинейная вязкость (3.21)": "Аналогично линейной, но ω зависит от градиента",
    "Регуляриз. Лакс-Вендрофф (3.19)": "Диагональ 1+2k, поддиагональ -λ/2 - k, наддиагональ λ/2 - k, где k = τ a²/(2 h²)",
}


def normalize_user_scheme_entry(data, type_):
    if type_ == "Неявная":
        if isinstance(data, dict) and "diagonals" in data and "rhs" in data:
            return data, "Неявная (Матричная)"
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return tuple(data), "Неявная (Режимы)"
    return data, type_


def format_user_scheme_formula(name, data, type_):
    data, type_ = normalize_user_scheme_entry(data, type_)

    if type_ == "Явная":
        return f"Пользовательская схема: {name}\n\nТип: Явная\n\nФормула:\n{data}"

    if type_ == "Неявная (Режимы)":
        if isinstance(data, (list, tuple)) and len(data) == 2:
            mode, rhs = data
            return (
                f"Пользовательская схема: {name}\n\n"
                f"Тип: Неявная\n\n"
                f"Шаблон матрицы: {mode}\n\n"
                f"Правая часть (RHS):\n{rhs}"
            )
        return f"Пользовательская схема: {name}\n\nТип: Неявная\n\nОписание:\n{data}"

    if type_ == "Неявная (Матричная)":
        diagonals = data.get("diagonals", {}) if isinstance(data, dict) else {}
        rhs = data.get("rhs", "") if isinstance(data, dict) else ""
        diag_lines = []
        for offset in sorted(diagonals, key=lambda value: int(value)):
            diag_lines.append(f"Смещение {offset}: {diagonals[offset]}")
        diag_text = "\n".join(diag_lines) if diag_lines else "Диагонали не заданы"
        return (
            f"Пользовательская схема: {name}\n\n"
            f"Тип: Неявная матричная\n\n"
            f"Диагонали матрицы:\n{diag_text}\n\n"
            f"Правая часть (RHS):\n{rhs}"
        )

    return f"Пользовательская схема: {name}\n\nТип: {type_}\n\nОписание:\n{data}"


def norms(UN, UE, dx):
    # UN - численное (Nt, Nx), UE - аналитическое (Nt, Nx)
    e = UN - UE
    return {
        "L1": dx * np.sum(np.abs(e), axis=1), 
        "L2": np.sqrt(dx * np.sum(e * e, axis=1)), 
        "Linf": np.max(np.abs(e), axis=1)
    }


def fmt_norm(nn, tg, title):
    ii = sorted(set([0, len(tg) // 10, len(tg) // 4, len(tg) // 2, 3 * len(tg) // 4, len(tg) - 1]))
    out = [title, "-" * len(title)]
    for i in ii:
        out.append(f"t={tg[i]:.4g} | L1={nn['L1'][i]:.3e} | L2={nn['L2'][i]:.3e} | Linf={nn['Linf'][i]:.3e}")
    return "\n".join(out)


def norm_summary_rows(nd):
    rows = []
    for name, values in sorted(nd.items(), key=lambda kv: kv[1]["L2"][-1]):
        rows.append(
            (
                name,
                values["L1"][-1],
                values["L2"][-1],
                values["Linf"][-1],
                float(np.max(values["L2"])),
            )
        )
    return rows


def norm_formulas_text(dx, t_end):
    return (
        "Нормы считаются корректно для равномерной пространственной сетки:\n"
        "L1 = h * sum_i |u_num - u_exact|\n"
        "L2 = sqrt(h * sum_i (u_num - u_exact)^2)\n"
        "Linf = max_i |u_num - u_exact|\n"
        f"Здесь h = {dx:.4g}, итоговая таблица отсортирована по L2 в момент t = {t_end:.4g}."
    )


class NormsTableDialog(QDialog):
    def __init__(self, rows, dx, t_end, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Таблица норм ошибок")
        self.resize(780, 420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(norm_formulas_text(dx, t_end)))

        table = QTableWidget(len(rows), 5)
        table.setHorizontalHeaderLabels([
            "Схема",
            "L1(t_end)",
            "L2(t_end)",
            "Linf(t_end)",
            "max L2 по времени",
        ])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setAlternatingRowColors(True)

        for row_idx, row in enumerate(rows):
            scheme, l1, l2, linf, max_l2 = row
            values = [scheme, f"{l1:.3e}", f"{l2:.3e}", f"{linf:.3e}", f"{max_l2:.3e}"]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_idx > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, col_idx, item)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col_idx in range(1, 5):
            header.setSectionResizeMode(col_idx, QHeaderView.ResizeToContents)
        table.selectRow(0) if rows else None

        layout.addWidget(table)

        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


def plot_anim(x, t, sols, x0, sx, y0, sy):
    ensure_plotly()
    names = list(sols.keys())
    d0 = [go.Scatter(x=x, y=sols[n][0], mode="lines", name=n, line={"color": COL.get(n, None)}) for n in names]
    fr = [go.Frame(data=[go.Scatter(x=x, y=sols[n][k], mode="lines", name=n, line={"color": COL.get(n, None)}) for n in names], name=str(k)) for k in range(len(t))]
    speed_buttons = [
        {
            "label": "0.5x",
            "method": "animate",
            "args": [None, {"frame": {"duration": 120, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        },
        {
            "label": "1x",
            "method": "animate",
            "args": [None, {"frame": {"duration": 60, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        },
        {
            "label": "2x",
            "method": "animate",
            "args": [None, {"frame": {"duration": 30, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        },
        {
            "label": "4x",
            "method": "animate",
            "args": [None, {"frame": {"duration": 15, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        },
    ]
    fig = go.Figure(
        data=d0,
        layout=go.Layout(
            title="Анимация схем (аналог Manipulate)",
            xaxis={"title": "x", "range": [x0 - sx, x0 + sx]},
            yaxis={"title": "u(x,t)", "range": [y0 - sy, y0 + sy]},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Пуск",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 60, "redraw": False}, "fromcurrent": True}],
                        },
                        {
                            "label": "Стоп",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "x": 0.1,
                    "y": 1.15,
                    "xanchor": "left",
                    "yanchor": "top",
                },
                {
                    "type": "buttons",
                    "buttons": speed_buttons,
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "x": 0.1,
                    "y": 1.0,
                    "xanchor": "left",
                    "yanchor": "top",
                },
            ],
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "t = "},
                    "pad": {"b": 10, "t": 60},
                    "steps": [
                        {"method": "animate", "args": [[str(k)], {"mode": "immediate"}], "label": f"{t[k]:.3g}"}
                        for k in range(len(t))
                    ],
                }
            ],
        ),
        frames=fr,
    )
    fig.show()


def export_anim(x, t, sols, x0, sx, y0, sy, file_path):
    return export_anim_with_progress(x, t, sols, x0, sx, y0, sy, file_path)


class ExportCancelledError(RuntimeError):
    pass


class ExportProgressDialog(QDialog):
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Экспорт анимации")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        self.label = QLabel("Подготовка экспорта...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.cancel_button)

    def update_progress(self, value, current_frame, total_frames):
        self.label.setText(f"Экспорт анимации: кадр {current_frame} из {total_frames}")
        self.progress_bar.setValue(value)

    def mark_finished(self):
        self.label.setText("Экспорт анимации завершён")
        self.progress_bar.setValue(100)


def export_anim_with_progress(x, t, sols, x0, sx, y0, sy, file_path, progress_callback=None, is_cancelled=None):
    try:
        import imageio.v2 as imageio
    except ImportError as ex:
        raise RuntimeError("Для экспорта анимации установите зависимости imageio и imageio-ffmpeg из requirements.txt") from ex
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in {".gif", ".mp4"}:
        raise ValueError("Поддерживается экспорт только в GIF или MP4")

    ext_settings = {
        ".gif": {"fps": 12, "dpi": 90},
        ".mp4": {"fps": 15, "dpi": 110},
    }
    settings = ext_settings[ext]

    fig = Figure(figsize=(10, 6), dpi=settings["dpi"])
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(x0 - sx, x0 + sx)
    ax.set_ylim(y0 - sy, y0 + sy)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")

    lines = []
    for name, data in sols.items():
        (line,) = ax.plot(x, data[0], label=name, color=COL.get(name, None))
        lines.append((name, line, data))

    if lines:
        ax.legend(loc="best")

    if ext == ".gif":
        writer = imageio.get_writer(file_path, mode="I", fps=settings["fps"], loop=0)
    else:
        writer = imageio.get_writer(
            file_path,
            fps=settings["fps"],
            codec="libx264",
            quality=7,
            macro_block_size=2,
        )

    total_frames = len(t)
    if total_frames == 0:
        raise ValueError("Нет кадров для экспорта")

    try:
        for frame_idx in range(total_frames):
            if is_cancelled and is_cancelled():
                raise ExportCancelledError("Экспорт отменён пользователем")

            for _, line, data in lines:
                line.set_ydata(data[frame_idx])
            ax.set_title(f"Сравнение схем, t={t[frame_idx]:.4g}")

            canvas.draw()
            frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            writer.append_data(frame)

            if progress_callback:
                progress = int(round(((frame_idx + 1) / total_frames) * 100))
                progress_callback(progress, frame_idx + 1, total_frames)
    except Exception:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise
    finally:
        writer.close()


class ExportAnimationWorker(QObject):
    progress = pyqtSignal(int, int, int)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, x, t, sols, x0, sx, y0, sy, file_path):
        super().__init__()
        self.x = x
        self.t = t
        self.sols = sols
        self.x0 = x0
        self.sx = sx
        self.y0 = y0
        self.sy = sy
        self.file_path = file_path
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            export_anim_with_progress(
                self.x,
                self.t,
                self.sols,
                self.x0,
                self.sx,
                self.y0,
                self.sy,
                self.file_path,
                progress_callback=self.progress.emit,
                is_cancelled=lambda: self._cancel_requested,
            )
        except ExportCancelledError:
            self.cancelled.emit()
        except Exception as ex:
            self.failed.emit(str(ex))
        else:
            self.finished.emit(self.file_path)


def plot_norms(t, nd):
    ensure_plotly()
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Норма L1 (логарифмическая шкала)",
            "Норма L2 (логарифмическая шкала)",
            "Норма Linf (логарифмическая шкала)",
        ),
    )

    def sanitize_log_values(values):
        arr = np.asarray(values, dtype=float)
        return np.where(arr > 0, arr, np.nan)

    positive_values = []
    for n, v in nd.items():
        c = COL.get(n, None)
        l1_vals = sanitize_log_values(v["L1"])
        l2_vals = sanitize_log_values(v["L2"])
        linf_vals = sanitize_log_values(v["Linf"])
        positive_values.extend(l1_vals[np.isfinite(l1_vals)].tolist())
        positive_values.extend(l2_vals[np.isfinite(l2_vals)].tolist())
        positive_values.extend(linf_vals[np.isfinite(linf_vals)].tolist())
        trace_style = {
            "mode": "lines+markers",
            "name": n,
            "line": {"color": c, "width": 2},
            "marker": {"size": 7, "color": c},
        }
        fig.add_trace(go.Scatter(x=t, y=l1_vals, showlegend=True, **trace_style), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=l2_vals, showlegend=False, **trace_style), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=linf_vals, showlegend=False, **trace_style), row=1, col=3)

    if positive_values:
        min_positive = min(positive_values)
        max_positive = max(positive_values)
        lower_power = int(np.floor(np.log10(min_positive))) - 1
        upper_power = int(np.ceil(np.log10(max_positive)))
        tick_vals = []
        for power in range(lower_power, upper_power + 1):
            for multiplier in (1, 2, 5):
                tick = multiplier * (10 ** power)
                if tick >= 0.1 and tick <= max_positive * 1.25:
                    tick_vals.append(tick)
        tick_vals = sorted(set(tick_vals)) or [0.1, 0.2, 0.5, 1, 2]
    else:
        tick_vals = [0.1, 0.2, 0.5, 1, 2]

    tick_text = [f"{tick:g}" for tick in tick_vals]
    fig.update_yaxes(type="log", title_text="log(L1)", tickmode="array", tickvals=tick_vals, ticktext=tick_text, row=1, col=1)
    fig.update_yaxes(type="log", title_text="log(L2)", tickmode="array", tickvals=tick_vals, ticktext=tick_text, row=1, col=2)
    fig.update_yaxes(type="log", title_text="log(Linf)", tickmode="array", tickvals=tick_vals, ticktext=tick_text, row=1, col=3)
    fig.update_xaxes(title_text="t", row=1, col=1)
    fig.update_xaxes(title_text="t", row=1, col=2)
    fig.update_xaxes(title_text="t", row=1, col=3)
    fig.update_layout(title="Логарифмические графики норм ошибок", legend_title="Схемы")
    fig.show()


def plot_snapshot(x, t, sols, k, x0, sx, y0, sy):
    ensure_plotly()
    names = list(sols.keys())
    data = [go.Scatter(x=x, y=sols[n][k], mode="lines", name=n, line={"color": COL.get(n, None)}) for n in names]
    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"Сравнение схем в срезе, t={t[k]:.4g}",
        xaxis={"title": "x", "range": [x0 - sx, x0 + sx]},
        yaxis={"title": "u(x,t)", "range": [y0 - sy, y0 + sy]},
    )
    fig.show()


def plot_final_errors(errs):
    ensure_plotly()
    names = list(errs.keys())
    l1 = [errs[n]["L1"][-1] for n in names]
    l2 = [errs[n]["L2"][-1] for n in names]
    li = [errs[n]["Linf"][-1] for n in names]
    fig = go.Figure(
        data=[
            go.Bar(name="L1(конец)", x=names, y=l1),
            go.Bar(name="L2(конец)", x=names, y=l2),
            go.Bar(name="Linf(конец)", x=names, y=li),
        ]
    )
    fig.update_layout(title="Итоговые ошибки на последнем шаге", barmode="group")
    fig.show()


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

    y_min, y_max = np.min(U), np.max(U)

    speed_buttons = [
        {
            "label": "0.5x",
            "method": "animate",
            "args": [None, {"frame": {"duration": 120, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        },
        {
            "label": "1x",
            "method": "animate",
            "args": [None, {"frame": {"duration": 60, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        },
        {
            "label": "2x",
            "method": "animate",
            "args": [None, {"frame": {"duration": 30, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        },
        {
            "label": "4x",
            "method": "animate",
            "args": [None, {"frame": {"duration": 15, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        },
    ]
    fig = go.Figure(
        data=[go.Scatter(x=xgrid, y=U[0], mode="lines")],
        layout=go.Layout(
            title="Анимация u(x,t)",
            xaxis=dict(range=[xgrid.min(), xgrid.max()]),
            yaxis=dict(range=[y_min, y_max]),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "▶ Loop",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 60, "redraw": False}, "fromcurrent": True, "loop": True}],
                        },
                        {
                            "label": "⏹ Stop",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "x": 0.1,
                    "y": 1.15,
                    "xanchor": "left",
                    "yanchor": "top",
                },
                {
                    "type": "buttons",
                    "buttons": speed_buttons,
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "x": 0.1,
                    "y": 1.0,
                    "xanchor": "left",
                    "yanchor": "top",
                },
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"prefix": "t = "},
                "pad": {"b": 10, "t": 60},
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


def _a_scalar(ctx, xv, tv):
    return float(np.asarray(ctx.a(np.array([xv], dtype=float), float(tv)), dtype=float)[0])


def _characteristics(ctx, nchar=21):
    x0 = np.linspace(float(ctx.x[0]), float(ctx.x[-1]), int(nchar))
    t = ctx.t
    paths = np.zeros((len(t), len(x0)), dtype=float)
    vals = np.zeros_like(paths)
    paths[0, :] = x0
    vals[0, :] = np.asarray(ctx.u(x0, np.full_like(x0, t[0])), dtype=float)
    for j in range(len(x0)):
        xk = float(x0[j])
        for n in range(len(t) - 1):
            tn = float(t[n]); dt = float(t[n + 1] - t[n])
            k1 = _a_scalar(ctx, xk, tn)
            k2 = _a_scalar(ctx, xk + 0.5 * dt * k1, tn + 0.5 * dt)
            k3 = _a_scalar(ctx, xk + 0.5 * dt * k2, tn + 0.5 * dt)
            k4 = _a_scalar(ctx, xk + dt * k3, tn + dt)
            xk = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            paths[n + 1, j] = xk
            vals[n + 1, j] = float(np.asarray(ctx.u(np.array([xk]), np.array([t[n + 1]])), dtype=float)[0])
    return t, paths, vals


def plot_characteristics(ctx, nchar=21, highlight_x0=None):
    ensure_plotly()
    t, paths, vals = _characteristics(ctx, nchar=nchar)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Характеристики x(t)", "u вдоль характеристик"))
    hi = None
    if highlight_x0 is not None:
        hi = int(np.argmin(np.abs(paths[0, :] - float(highlight_x0))))
    for j in range(paths.shape[1]):
        lbl = f"x0={paths[0, j]:.3g}"
        lw = 3 if hi is not None and j == hi else 1.2
        col = "#d62728" if hi is not None and j == hi else None
        fig.add_trace(go.Scatter(x=t, y=paths[:, j], mode="lines", name=lbl, line={"width": lw, "color": col}), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=vals[:, j], mode="lines", showlegend=False, line={"width": lw, "color": col}), row=1, col=2)
    fig.update_xaxes(title_text="t", row=1, col=1)
    fig.update_xaxes(title_text="t", row=1, col=2)
    fig.update_yaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="u", row=1, col=2)
    fig.update_layout(title="Характеристическое представление")
    fig.show()


def plot_analytic_3d(ctx):
    ensure_plotly()
    X, T = np.meshgrid(ctx.x, ctx.t)
    U = np.asarray(ctx.u(X, T), dtype=float)
    fig = go.Figure(data=[go.Surface(x=X, y=T, z=U, colorscale="Viridis")])
    fig.update_layout(
        title="3D аналитическое решение",
        scene={"xaxis_title": "x", "yaxis_title": "t", "zaxis_title": "u(x,t)"},
    )
    fig.show()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Решатель уравнения переноса")
        self.setGeometry(110, 80, 920, 760)
        self.export_thread = None
        self.export_worker = None
        self.export_progress = None
        self.export_in_progress = False
        self.export_skip_messages = []
        self.ui()

    def ui(self):
        root = QVBoxLayout()
        head = QLabel("Уравнение переноса: лаборатория явных и неявных схем")
        head.setAlignment(Qt.AlignCenter)
        head.setStyleSheet("font-size:17px;font-weight:600;")
        root.addWidget(head)

        g1 = QGroupBox("1) Коэффициент переноса и начальное условие")
        f1 = QFormLayout()
        self.mode = QComboBox(); self.mode.addItems(["a = const", "a = a(x)", "a = a(t)", "a = a(x,t)"])
        self.ae = QLineEdit("1"); self.axt = QComboBox(); self.axt.addItems(["a = x - t", "a = x + t", "a = x * t", "a = x * exp(-t)", "a = alpha*x + beta*t", "a = alpha*x + beta*t + d"])
        self.al = QLineEdit("1"); self.be = QLineEdit("0"); self.de = QLineEdit("0"); self.phi = QLineEdit("exp(-x**2)")
        f1.addRow("Вид a:", self.mode); f1.addRow("Выражение a:", self.ae); f1.addRow("Случай a(x,t):", self.axt)
        r = QHBoxLayout(); r.addWidget(QLabel("alpha")); r.addWidget(self.al); r.addWidget(QLabel("beta")); r.addWidget(self.be); r.addWidget(QLabel("d")); r.addWidget(self.de)
        f1.addRow("Параметры:", r); f1.addRow("phi0(x):", self.phi); g1.setLayout(f1); root.addWidget(g1)

        g2 = QGroupBox("2) Сетка")
        f2 = QGridLayout()
        self.xmin = QLineEdit("-10"); self.xmax = QLineEdit("10"); self.tmin = QLineEdit("0"); self.tmax = QLineEdit("10")
        self.nx = QSpinBox(); self.nx.setRange(11, 10000); self.nx.setValue(201); self.nx.setFixedWidth(90)
        self.nt = QSpinBox(); self.nt.setRange(11, 10000); self.nt.setValue(201); self.nt.setFixedWidth(90)
        f2.addWidget(QLabel("x_min"), 0, 0); f2.addWidget(self.xmin, 0, 1); f2.addWidget(QLabel("x_max"), 0, 2); f2.addWidget(self.xmax, 0, 3)
        f2.addWidget(QLabel("t_min"), 1, 0); f2.addWidget(self.tmin, 1, 1); f2.addWidget(QLabel("t_max"), 1, 2); f2.addWidget(self.tmax, 1, 3)
        f2.addWidget(QLabel("Nx"), 2, 0); f2.addWidget(self.nx, 2, 1); f2.addWidget(QLabel("Nt"), 2, 2); f2.addWidget(self.nt, 2, 3); g2.setLayout(f2); root.addWidget(g2)

        g3 = QGroupBox("3) Окно графика")
        f3 = QGridLayout()
        self.x0 = QLineEdit("0"); self.sx = QLineEdit("10"); self.y0 = QLineEdit("0.5"); self.sy = QLineEdit("1.2")
        self.kidx = QSpinBox(); self.kidx.setRange(0, int(self.nt.value()) - 1); self.kidx.setValue(int(self.nt.value()) - 1)
        f3.addWidget(QLabel("Центр X"), 0, 0); f3.addWidget(self.x0, 0, 1); f3.addWidget(QLabel("Масштаб X"), 0, 2); f3.addWidget(self.sx, 0, 3)
        f3.addWidget(QLabel("Центр Y"), 1, 0); f3.addWidget(self.y0, 1, 1); f3.addWidget(QLabel("Масштаб Y"), 1, 2); f3.addWidget(self.sy, 1, 3)
        f3.addWidget(QLabel("Срез k"), 2, 0); f3.addWidget(self.kidx, 2, 1)
        g3.setLayout(f3); root.addWidget(g3)

        g_cfl = QGroupBox("3.5) CFL число")
        f_cfl = QHBoxLayout()
        self.cfl_label = QLabel("CFL: не рассчитано")
        self.calc_cfl_btn = QPushButton("Рассчитать CFL")
        self.calc_cfl_btn.clicked.connect(self.calc_cfl)
        f_cfl.addWidget(self.cfl_label)
        f_cfl.addWidget(self.calc_cfl_btn)
        g_cfl.setLayout(f_cfl); root.addWidget(g_cfl)

        g4 = QGroupBox("4) Схемы")
        h = QHBoxLayout(); self.lex = QListWidget(); self.lim = QListWidget(); self.lex.setMinimumHeight(200); self.lim.setMinimumHeight(200)
        for i, n in enumerate(EXP.keys()):
            it = QListWidgetItem(n); it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Checked if i == 0 else Qt.Unchecked); self.lex.addItem(it)
        for i, n in enumerate(IMP.keys()):
            it = QListWidgetItem(n); it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Checked if i == 0 else Qt.Unchecked); self.lim.addItem(it)
        le = QVBoxLayout(); le.addWidget(QLabel("Явные")); le.addWidget(self.lex); ri = QVBoxLayout(); ri.addWidget(QLabel("Неявные")); ri.addWidget(self.lim); h.addLayout(le); h.addLayout(ri); g4.setLayout(h); root.addWidget(g4)
        self.exact = QComboBox(); self.exact.addItems(["Показывать аналитическое решение", "Скрыть аналитическое решение"]); root.addWidget(self.exact)

        g5 = QGroupBox("5) Пользовательские схемы")
        h5 = QVBoxLayout()
        self.user_schemes = QListWidget()
        self.user_schemes.setMinimumHeight(100)
        self.user_schemes.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        h5.addWidget(QLabel("Пользовательские схемы:"))
        h5.addWidget(self.user_schemes)
        btns = QHBoxLayout()
        self.add_scheme_btn = QPushButton("Добавить схему")
        self.del_scheme_btn = QPushButton("Удалить схему")
        self.add_scheme_btn.clicked.connect(self.add_user_scheme)
        self.del_scheme_btn.clicked.connect(self.del_user_scheme)
        btns.addWidget(self.add_scheme_btn)
        btns.addWidget(self.del_scheme_btn)
        h5.addLayout(btns)
        g5.setLayout(h5); root.addWidget(g5)

        b = QHBoxLayout()
        self.banim = QPushButton("Анимация")
        self.bsnap = QPushButton("Срез и сравнение")
        self.bnorm = QPushButton("Нормы ошибок")
        self.bbar = QPushButton("Итоговые ошибки")
        self.bconv = QPushButton("Анализ сходимости")
        self.bexport = QPushButton("Экспорт анимации")
        b.addWidget(self.banim); b.addWidget(self.bsnap); b.addWidget(self.bnorm); b.addWidget(self.bbar); b.addWidget(self.bconv); b.addWidget(self.bexport); root.addLayout(b)
        b2 = QHBoxLayout()
        self.bsimple = QPushButton("Построить графики")
        self.bchar = QPushButton("Характеристики (exp(-x**2))")
        self.b3d = QPushButton("3D аналитика")
        self.bref = QPushButton("Справочник схем")
        b2.addWidget(self.bsimple); b2.addWidget(self.bchar); b2.addWidget(self.b3d); b2.addWidget(self.bref); root.addLayout(b2)
        self.out = QPlainTextEdit(); self.out.setReadOnly(True); self.out.setPlaceholderText("Здесь выводятся ошибки, предупреждения и отчёты."); root.addWidget(self.out)

        self.setStyleSheet("QWidget{font-size:13px;} QGroupBox{border:1px solid #bac5d1;border-radius:8px;margin-top:6px;font-weight:600;padding:7px;} QPushButton{background:#1f6feb;color:#fff;border-radius:6px;padding:8px;font-weight:600;} QPushButton:hover{background:#2f81f7;}")

        scroll = QScrollArea()
        widget = QWidget()
        widget.setLayout(root)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self.mode.currentTextChanged.connect(self.upd)
        self.nt.valueChanged.connect(self.upd_kmax)
        self.banim.clicked.connect(self.run_anim)
        self.bsnap.clicked.connect(self.run_snapshot)
        self.bnorm.clicked.connect(self.run_norm)
        self.bbar.clicked.connect(self.run_final_errors)
        self.bconv.clicked.connect(self.run_convergence)
        self.bexport.clicked.connect(self.run_export)
        self.bsimple.clicked.connect(self.run_simple)
        self.bchar.clicked.connect(self.run_chars)
        self.b3d.clicked.connect(self.run_analytic_3d)
        self.bref.clicked.connect(self.show_ref)
        self.load_user_schemes()
        self.upd()

    def upd(self):
        axt = self.mode.currentText() == "a = a(x,t)"
        self.ae.setVisible(not axt); self.axt.setVisible(axt); self.al.setVisible(axt); self.be.setVisible(axt); self.de.setVisible(axt)
        self.upd_kmax()

    def upd_kmax(self, _=None):
        kmax = max(0, int(self.nt.value()) - 1)
        self.kidx.setMaximum(kmax)
        if self.kidx.value() > kmax:
            self.kidx.setValue(kmax)

    def err(self, s): QMessageBox.critical(self, "Ошибка", s)
    def f(self, w, n):
        try: return float(w.text())
        except ValueError: raise ValueError(f"Поле '{n}' должно быть числом")

    def pfunc(self):
        e = self.phi.text().strip()
        if not e: return phi0
        try:
            ff = sp.lambdify(x_sym, sp.sympify(e), modules=[{"heaviside": lambda z: np.heaviside(z, 1.0)}, "numpy"])
            return lambda z: np.asarray(ff(z), dtype=float)
        except Exception as ex:
            raise ValueError(f"Ошибка разбора phi0(x): {ex}")

    def ctx(self, phi_override=None):
        xmin, xmax, tmin, tmax = self.f(self.xmin, "x_min"), self.f(self.xmax, "x_max"), self.f(self.tmin, "t_min"), self.f(self.tmax, "t_max")
        if xmax <= xmin: raise ValueError("Нужно x_max > x_min")
        if tmax <= tmin: raise ValueError("Нужно t_max > t_min")
        x, t = np.linspace(xmin, xmax, int(self.nx.value())), np.linspace(tmin, tmax, int(self.nt.value()))
        p = phi_override if phi_override is not None else self.pfunc()
        if self.mode.currentText() == "a = a(x,t)":
            return make_ctx(self.mode.currentText(), "", self.axt.currentText(), x, t, p, self.f(self.al, "alpha"), self.f(self.be, "beta"), self.f(self.de, "d"))
        return make_ctx(self.mode.currentText(), self.ae.text().strip(), "", x, t, p)

    def sel(self):
        r = []
        for i in range(self.lex.count()):
            if self.lex.item(i).checkState() == Qt.Checked: r.append(self.lex.item(i).text())
        for i in range(self.lim.count()):
            if self.lim.item(i).checkState() == Qt.Checked: r.append(self.lim.item(i).text())
        for i in range(self.user_schemes.count()):
            if self.user_schemes.item(i).checkState() == Qt.Checked: r.append(self.user_schemes.item(i).text())
        return r

    def build_user_scheme(self, name):
        scheme_data = USER_SCHEMES.get(name)
        if not scheme_data:
            return None

        data, type_ = normalize_user_scheme_entry(*scheme_data)

        if type_ == "Явная":
            def user_scheme(c):
                U = u0(c)
                nx = len(c.x)
                dx, dt = dxdt(c)
                for nn in range(len(c.t) - 1):
                    p, q = U[nn], U[nn + 1]
                    a_vals = c.a(c.x, float(c.t[nn]))
                    for i in range(1, nx - 1):
                        context = {
                            'u': p[i],
                            'u_left': p[i - 1],
                            'u_right': p[i + 1],
                            'a': a_vals[i],
                            'dt': dt,
                            'dx': dx,
                            'np': np,
                        }
                        q[i] = eval(data, {"__builtins__": None}, context)
                    bnd(q, c, float(c.t[nn + 1]))
                return U

            return user_scheme

        if type_ == "Неявная (Режимы)" and isinstance(data, (list, tuple)) and len(data) == 2:
            mode, rhs_formula = data

            def user_scheme(c):
                U = u0(c)
                nx = len(c.x)
                dx, dt = dxdt(c)
                for nn in range(len(c.t) - 1):
                    tn = float(c.t[nn + 1])
                    av = c.a(c.x, tn)
                    cfl = (dt / dx) * av
                    l, d, u, r = np.zeros(nx - 1), np.ones(nx), np.zeros(nx - 1), U[nn].copy()
                    for i in range(1, nx - 1):
                        context = {
                            'u': U[nn, i],
                            'u_left': U[nn, i - 1],
                            'u_right': U[nn, i + 1],
                            'a': av[i],
                            'dt': dt,
                            'dx': dx,
                            'np': np,
                        }
                        r[i] = eval(rhs_formula, {"__builtins__": None}, context)
                        if mode == "cn":
                            d[i], l[i - 1] = 1 + 0.5 * cfl[i], -0.5 * cfl[i]
                        elif mode == "gen":
                            d[i], l[i - 1], u[i] = 1.0, -0.5 * cfl[i], 0.5 * cfl[i]
                        else:
                            if cfl[i] >= 0:
                                d[i], l[i - 1] = 1 + cfl[i], -cfl[i]
                            else:
                                d[i], u[i] = 1 - cfl[i], cfl[i]
                    d[0] = d[-1] = 1.0
                    u[0] = 0.0
                    l[-1] = 0.0
                    r[0], r[-1] = us(c, c.x[0], tn), us(c, c.x[-1], tn)
                    U[nn + 1] = tdma(l, d, u, r)
                return U

            return user_scheme

        if type_ == "Неявная (Матричная)" and isinstance(data, dict):
            def user_scheme(c):
                import scipy.linalg as sla

                U = u0(c)
                nx = len(c.x)
                dx, dt = dxdt(c)
                diag_dict = data['diagonals']
                rhs_formula = data['rhs']
                offsets = [int(k) for k in diag_dict.keys()]
                upper_bw = max(0, max(offsets)) if offsets else 0
                lower_bw = abs(min(0, min(offsets))) if offsets else 0

                for nn in range(len(c.t) - 1):
                    tn = float(c.t[nn + 1])
                    av = c.a(c.x, tn)
                    ab = np.zeros((upper_bw + lower_bw + 1, nx))
                    b = np.zeros(nx)

                    for i in range(nx):
                        ctx_vars = {
                            'u': U[nn, i],
                            'u_left': U[nn, max(0, i - 1)],
                            'u_right': U[nn, min(nx - 1, i + 1)],
                            'a': av[i],
                            'dt': dt,
                            'dx': dx,
                            'np': np,
                            'i': i,
                        }

                        if i == 0 or i == nx - 1:
                            ab[upper_bw, i] = 1.0
                            b[i] = us(c, c.x[i], tn)
                        else:
                            b[i] = eval(rhs_formula, {"__builtins__": None}, ctx_vars)
                            for off, expr in diag_dict.items():
                                val = eval(expr, {"__builtins__": None}, ctx_vars)
                                col_idx = i + int(off)
                                if 0 <= col_idx < nx:
                                    row_idx = upper_bw - int(off)
                                    ab[row_idx, col_idx] = val

                    U[nn + 1] = sla.solve_banded((lower_bw, upper_bw), ab, b)
                return U

            return user_scheme

        raise ValueError(f"Пользовательская схема '{name}' имеет неподдерживаемый формат")

    def resolve_scheme(self, name):
        return EXP.get(name) or IMP.get(name) or self.build_user_scheme(name)

    def solve(self, ctx, exact):
        # Сетка и аналитика (uex имеет форму (Nt, Nx))
        names, sol, skip = self.sel(), OrderedDict(), []
        X_grid, T_grid = np.meshgrid(ctx.x, ctx.t)
        uex = np.asarray(ctx.u(X_grid, T_grid), dtype=float)
        
        if not names and not exact: 
            raise ValueError("Выберите хотя бы одну схему")
        
        for n in names:
            s = self.resolve_scheme(n)

            if s:
                try: 
                    sol[n] = s(ctx)
                except Exception as ex: 
                    skip.append(f"{n}: {ex}")

        if exact: 
            sol[EXACT_NAME] = uex
        if not sol: 
            raise ValueError("Не удалось рассчитать выбранные схемы")
        
        return sol, uex, skip

    def run_anim(self):
        try:
            c = self.ctx(); sol, _, skip = self.solve(c, True)
            x0, sx, y0, sy = self.f(self.x0, "Центр X"), self.f(self.sx, "Масштаб X"), self.f(self.y0, "Центр Y"), self.f(self.sy, "Масштаб Y")
            if sx <= 0 or sy <= 0: raise ValueError("Масштабы X/Y должны быть > 0")
            plot_anim(c.x, c.t, sol, x0, sx, y0, sy)
            t = [f"Построено кривых: {len(sol)}", "Выбраны: " + ", ".join(sol.keys())]
            if skip: t += ["", "Пропущенные схемы:"] + skip
            self.out.setPlainText("\n".join(t))
        except Exception as ex: self.err(str(ex))

    def run_norm(self):
            try:
                c = self.ctx()
                # Получаем решения и аналитику (uex)
                sol, uex, skip = self.solve(c, False)
                dx, _ = dxdt(c)
                nd = OrderedDict()
                text = []

                for n, U_num in sol.items():
                    # Вызываем глобальную функцию нормы
                    nn = norms(U_num, uex, dx)
                    nd[n] = nn
                    # Форматируем текст для вывода в текстовое поле (fmt_norm из вашего оригинала)
                    text.append(fmt_norm(nn, c.t, f"Ошибки для схемы: {n}"))

                # Рисуем графики норм
                plot_norms(c.t, nd)
                NormsTableDialog(norm_summary_rows(nd), dx, c.t[-1], self).exec_()
                
                if skip:
                    text.append("\nПропущенные схемы (ошибки):")
                    text.extend(skip)
                text.append("")
                text.append(norm_formulas_text(dx, c.t[-1]))
                self.out.setPlainText("\n\n".join(text))
            except Exception as ex: 
                self.err(str(ex))

    def run_snapshot(self):
        try:
            c = self.ctx(); sol, _, skip = self.solve(c, self.exact.currentIndex() == 0)
            x0, sx, y0, sy = self.f(self.x0, "Центр X"), self.f(self.sx, "Масштаб X"), self.f(self.y0, "Центр Y"), self.f(self.sy, "Масштаб Y")
            if sx <= 0 or sy <= 0: raise ValueError("Масштабы X/Y должны быть > 0")
            k = max(0, min(int(self.kidx.value()), len(c.t) - 1))
            plot_snapshot(c.x, c.t, sol, k, x0, sx, y0, sy)
            txt = [f"Срез k={k}, t={c.t[k]:.5g}", "Кривые: " + ", ".join(sol.keys())]
            if skip: txt += ["", "Пропущенные схемы:"] + skip
            self.out.setPlainText("\n".join(txt))
        except Exception as ex: self.err(str(ex))

    def run_final_errors(self):
        try:
            c = self.ctx(); sol, ex, skip = self.solve(c, False); dx, _ = dxdt(c)
            nd = OrderedDict((n, norms(U, ex, dx)) for n, U in sol.items())
            plot_final_errors(nd)
            rows_for_table = norm_summary_rows(nd)
            NormsTableDialog(rows_for_table, dx, c.t[-1], self).exec_()
            rows = []
            for n, v in sorted(nd.items(), key=lambda kv: kv[1]["L2"][-1]):
                rows.append(f"{n}: L1={v['L1'][-1]:.3e}, L2={v['L2'][-1]:.3e}, Linf={v['Linf'][-1]:.3e}")
            rows += ["", norm_formulas_text(dx, c.t[-1])]
            if skip: rows += ["", "Пропущенные схемы:"] + skip
            self.out.setPlainText("\n".join(rows))
        except Exception as ex: self.err(str(ex))

    def run_simple(self):
        try:
            c = self.ctx()
            U = np.asarray(c.u(*np.meshgrid(c.x, c.t)), dtype=float)
            plot_contour(c.x, c.t, U)
            plot_3d(c.x, c.t, U)
            animate_solution(c.x, c.t, U)
            self.out.setPlainText("Построены контурный график, 3D и анимация аналитического решения.")
        except Exception as ex: self.err(str(ex))

    def run_chars(self):
        try:
            c = self.ctx(phi_override=phi0)
            nchar = int(max(9, min(33, self.nx.value() // 8)))
            plot_characteristics(c, nchar=nchar, highlight_x0=0.0)
            self.out.setPlainText(f"Построено характеристик: {nchar}\nДля наглядности использовано φ0(x)=exp(-x**2).")
        except Exception as ex: self.err(str(ex))

    def run_analytic_3d(self):
        try:
            c = self.ctx()
            plot_analytic_3d(c)
            self.out.setPlainText("Построена 3D-поверхность аналитического решения.")
        except Exception as ex: self.err(str(ex))

    def run_convergence(self):
        try:
            schemes = self.sel()
            if len(schemes) != 1:
                raise ValueError("Выберите ровно одну схему для анализа сходимости")
            scheme = schemes[0]
            s = self.resolve_scheme(scheme)
            if s is None:
                raise ValueError("Схема не найдена")
            base_nx = self.nx.value()
            hs = []
            errors = []
            for nx in [base_nx // 4, base_nx // 2, base_nx, base_nx * 2]:
                if nx < 10: continue
                self.nx.setValue(nx)
                c = self.ctx()
                U = s(c)
                UE = np.asarray(c.u(*np.meshgrid(c.x, c.t)), dtype=float)
                dx = (c.x[-1] - c.x[0]) / (len(c.x) - 1)
                err = norms(U, UE, dx)["L2"][-1]
                hs.append(dx)
                errors.append(err)
            self.nx.setValue(base_nx)
            ensure_plotly()
            fig = go.Figure(data=go.Scatter(x=hs, y=errors, mode="lines+markers"))
            fig.update_layout(title=f"Сходимость схемы {scheme}", xaxis_title="h", yaxis_title="L2 ошибка", xaxis_type="log", yaxis_type="log")
            fig.show()
            self.out.setPlainText(f"Анализ сходимости для {scheme} завершен.")
        except Exception as ex: self.err(str(ex))

    def run_export(self):
        try:
            if self.export_thread is not None:
                self.err("Экспорт уже выполняется")
                return
            c = self.ctx()
            sols, _, skip = self.solve(c, self.exact.currentIndex() == 0)
            x0 = self.f(self.x0, "Центр X")
            sx = self.f(self.sx, "Масштаб X")
            y0 = self.f(self.y0, "Центр Y")
            sy = self.f(self.sy, "Масштаб Y")
            if sx <= 0 or sy <= 0:
                raise ValueError("Масштабы X/Y должны быть > 0")
            fname, _ = QFileDialog.getSaveFileName(self, "Экспорт анимации", "animation.gif", "GIF animation (*.gif);;MP4 video (*.mp4)")
            if fname:
                self.export_skip_messages = skip
                self.export_progress = ExportProgressDialog(self)
                self.export_in_progress = True

                self.export_thread = QThread(self)
                self.export_worker = ExportAnimationWorker(c.x, c.t, sols, x0, sx, y0, sy, fname)
                self.export_worker.moveToThread(self.export_thread)

                self.export_thread.started.connect(self.export_worker.run)
                self.export_worker.progress.connect(self.on_export_progress)
                self.export_worker.finished.connect(self.on_export_finished)
                self.export_worker.failed.connect(self.on_export_failed)
                self.export_worker.cancelled.connect(self.on_export_cancelled)
                self.export_progress.cancel_requested.connect(self.export_worker.cancel)
                self.export_worker.finished.connect(self.export_thread.quit)
                self.export_worker.failed.connect(self.export_thread.quit)
                self.export_worker.cancelled.connect(self.export_thread.quit)
                self.export_thread.finished.connect(self.cleanup_export)

                self.export_progress.show()
                self.export_thread.start()
        except Exception as ex: self.err(str(ex))

    def on_export_progress(self, value, current_frame, total_frames):
        if self.export_progress is None:
            return
        self.export_progress.update_progress(value, current_frame, total_frames)

    def on_export_finished(self, file_path):
        if self.export_progress is not None:
            self.export_progress.mark_finished()
            self.export_progress.close()
        self.export_in_progress = False
        text = [f"Анимация экспортирована в {file_path}"]
        if self.export_skip_messages:
            text += ["", "Пропущенные схемы:"] + self.export_skip_messages
        self.out.setPlainText("\n".join(text))

    def on_export_failed(self, message):
        if self.export_progress is not None:
            self.export_progress.close()
        self.export_in_progress = False
        self.err(message)

    def on_export_cancelled(self):
        if self.export_progress is not None:
            self.export_progress.close()
        self.export_in_progress = False
        self.out.setPlainText("Экспорт анимации отменён.")

    def cleanup_export(self):
        if self.export_worker is not None:
            self.export_worker.deleteLater()
        if self.export_thread is not None:
            self.export_thread.deleteLater()
        self.export_worker = None
        self.export_thread = None
        self.export_progress = None
        self.export_skip_messages = []

    def calc_cfl(self):
        try:
            c = self.ctx()
            dx, dt = dxdt(c)
            xg, tg = np.meshgrid(c.x, c.t, indexing='xy')
            a_max = np.max(np.abs(c.a(xg, tg)))
            cfl = a_max * dt / dx
            self.cfl_label.setText(f"CFL: {cfl:.3f} {'(стабильно)' if cfl <= 1 else '(нестабильно)'}")
        except Exception as ex: self.err(str(ex))

    def show_ref(self):
        dialog = FormulaDialog(self)
        dialog.exec_()

    def add_user_scheme(self):
            dialog = AddSchemeDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                name, data, type_ = dialog.get_data() # Получаем структурированные данные
                
                if not name:
                    self.err("Введите имя схемы")
                    return
                if name in EXP or name in IMP or name in USER_SCHEMES:
                    self.err("Схема с таким именем уже существует")
                    return

                # Сохраняем (теперь data может быть строкой, кортежем или словарем)
                USER_SCHEMES[name] = (data, type_)
                
                it = QListWidgetItem(name)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked)
                self.user_schemes.addItem(it)
                self.save_user_schemes()
                self.out.setPlainText(f"Схема '{name}' добавлена.")

    def del_user_scheme(self):
        current = self.user_schemes.currentItem()
        if current:
            name = current.text()
            del USER_SCHEMES[name]
            self.user_schemes.takeItem(self.user_schemes.row(current))
            self.save_user_schemes()
            self.out.setPlainText(f"Схема '{name}' удалена.")
        else:
            self.err("Выберите схему для удаления")

    def save_user_schemes(self):
        try:
            with open("user_schemes.json", "w", encoding="utf-8") as f:
                json.dump(USER_SCHEMES, f, ensure_ascii=False, indent=4)
        except Exception as ex:
            self.err(f"Ошибка сохранения: {str(ex)}")

    def load_user_schemes(self):
        try:
            with open("user_schemes.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, (code, type_) in data.items():
                USER_SCHEMES[name] = normalize_user_scheme_entry(code, type_)
                it = QListWidgetItem(name)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked)
                self.user_schemes.addItem(it)
        except FileNotFoundError:
            pass  # Файл не существует, нормально
        except Exception as ex:
            self.err(f"Ошибка загрузки: {str(ex)}")


class FormulaDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справочник формул схем")
        self.setGeometry(300, 200, 600, 400)
        layout = QVBoxLayout()
        self.scheme_list = QListWidget()
        for name in SCHEME_FORMULAS.keys():
            self.scheme_list.addItem(name)
        for name in USER_SCHEMES.keys():
            self.scheme_list.addItem(f"Пользовательская: {name}")
        self.scheme_list.itemClicked.connect(self.show_formula)
        layout.addWidget(QLabel("Выберите схему:"))
        layout.addWidget(self.scheme_list)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        btn = QPushButton("Закрыть")
        btn.clicked.connect(self.close)
        layout.addWidget(btn)
        self.setLayout(layout)

    def show_formula(self, item):
        name = item.text()
        if name.startswith("Пользовательская: "):
            user_name = name[len("Пользовательская: "):]
            scheme_data = USER_SCHEMES.get(user_name)
            formula = format_user_scheme_formula(user_name, *scheme_data) if scheme_data else "Формула не найдена"
        else:
            formula = SCHEME_FORMULAS.get(name, "Формула не найдена")
        self.text.setPlainText(f"Схема: {name}\n\nФормула:\n{formula}")


class AddSchemeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Добавить пользовательскую схему")
        self.setMinimumWidth(550)
        self.setMinimumHeight(600)
        layout = QVBoxLayout()

        # 1. Имя схемы
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Назовите вашу схему")
        layout.addWidget(QLabel("Имя схемы:"))
        layout.addWidget(self.name_edit)

        # 2. Тип схемы
        self.type_combo = QComboBox()
        # Важно: имена типов должны совпадать с теми, что мы прописали в solve()
        self.type_combo.addItems(["Явная", "Неявная (Режимы)", "Неявная (Матричная)"])
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        layout.addWidget(QLabel("Тип реализации:"))
        layout.addWidget(self.type_combo)

        # 3. Виджет для выбора режима (для Неявная (Режимы))
        self.mode_group = QWidget()
        mode_lay = QVBoxLayout(self.mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["upwind", "gen", "cn", "lw", "317", "320", "319r", "318", "321", "315", "316"])
        mode_lay.addWidget(QLabel("Готовый шаблон матрицы:"))
        mode_lay.addWidget(self.mode_combo)
        mode_lay.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mode_group)

        # 4. Виджет конструктора диагоналей (для Неявная (Матричная))
        self.matrix_group = QGroupBox("Конструктор диагоналей матрицы")
        m_lay = QVBoxLayout()
        self.diag_list = QListWidget()
        
        diag_btns = QHBoxLayout()
        self.add_diag_btn = QPushButton("+ Добавить диагональ")
        self.add_diag_btn.clicked.connect(self.add_diagonal_item)
        self.rem_diag_btn = QPushButton("- Удалить выбранную")
        self.rem_diag_btn.clicked.connect(lambda: self.diag_list.takeItem(self.diag_list.currentRow()))
        
        diag_btns.addWidget(self.add_diag_btn)
        diag_btns.addWidget(self.rem_diag_btn)
        
        m_lay.addWidget(QLabel("0 - главная, -1 - левая, +1 - правая и т.д."))
        m_lay.addWidget(self.diag_list)
        m_lay.addLayout(diag_btns)
        self.matrix_group.setLayout(m_lay)
        layout.addWidget(self.matrix_group)

        # 5. Поле ввода основной формулы (RHS или New U)
        self.formula_label = QLabel("Формула схемы:")
        self.code_edit = QPlainTextEdit()
        self.code_edit.setMinimumHeight(100)
        layout.addWidget(self.formula_label)
        layout.addWidget(self.code_edit)

        # Подсказки
        self.help_label = QLabel()
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.help_label)

        # Кнопки
        btns = QHBoxLayout()
        ok_btn = QPushButton("Сохранить")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        self.setLayout(layout)
        # Инициализируем UI
        self.on_type_changed("Явная")

    def on_type_changed(self, text):
        # Управляем видимостью групп
        self.mode_group.setVisible(text == "Неявная (Режимы)")
        self.matrix_group.setVisible(text == "Неявная (Матричная)")

        if text == "Явная":
            self.formula_label.setText("Формула для нового слоя (U_new):")
            self.code_edit.setPlaceholderText("Пример: u - a*dt/dx*(u - u_left)")
            self.help_label.setText("Доступно: u, u_left, u_right, a, dt, dx, np")
        else:
            self.formula_label.setText("Правая часть уравнения (RHS):")
            self.code_edit.setPlaceholderText("Для большинства схем это просто: u")
            self.help_label.setText("Доступно: u, u_left, u_right, a, dt, dx, np, i (индекс)")

    def add_diagonal_item(self):
        """Добавляет строку в список диагоналей"""
        item = QListWidgetItem(self.diag_list)
        row_widget = QWidget()
        row_lay = QHBoxLayout(row_widget)
        
        spin = QSpinBox()
        spin.setRange(-50, 50)
        spin.setValue(0)
        spin.setToolTip("Смещение диагонали (0 - главная)")
        
        edit = QLineEdit()
        edit.setPlaceholderText("Коэффициент, напр: 1 + a*dt/dx")
        
        row_lay.addWidget(QLabel("Смещение:"))
        row_lay.addWidget(spin)
        row_lay.addWidget(QLabel("Коэфф:"))
        row_lay.addWidget(edit)
        row_lay.setContentsMargins(5, 2, 5, 2)
        
        item.setSizeHint(row_widget.sizeHint())
        self.diag_list.addItem(item)
        self.diag_list.setItemWidget(item, row_widget)

    def get_data(self):
        """Возвращает данные в формате, который понимает новый solve()"""
        name = self.name_edit.text().strip()
        t = self.type_combo.currentText()
        formula = self.code_edit.toPlainText().strip()

        if t == "Явная":
            return name, formula, t
        
        if t == "Неявная (Режимы)":
            mode = self.mode_combo.currentText()
            return name, (mode, formula), t

        if t == "Неявная (Матричная)":
            diagonals = {}
            for i in range(self.diag_list.count()):
                item = self.diag_list.item(i)
                widget = self.diag_list.itemWidget(item)
                offset = widget.findChild(QSpinBox).value()
                expr = widget.findChild(QLineEdit).text().strip()
                if expr:
                    diagonals[offset] = expr
            
            data = {
                "diagonals": diagonals,
                "rhs": formula
            }
            return name, data, t


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec_())
