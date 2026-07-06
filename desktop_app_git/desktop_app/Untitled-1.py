# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # --- Параметры ---
# x = np.linspace(-10, 10, 1000)
# t_vals = np.linspace(0, 10, 100)
# a = lambda x: 1 + 0.5*np.sin(x)  # скорость переноса

# # --- Численное интегрирование для характеристики ---
# # cumulative integral 1/a(x) dx
# dx = x[1] - x[0]
# integral = np.cumsum(1/a(x)) * dx  # интеграл от 0 до x

# # --- Начальные условия ---
# def f_step(s): return np.heaviside(s, 1)
# def f_smooth(s): return np.tanh(s)
# def f_sign_change(s): return np.sin(s)
# def f_two_fronts(s): return np.heaviside(s+2,1) + np.heaviside(s-3,1)

# # --- Создание фигуры ---
# fig, ax = plt.subplots(figsize=(8,4))
# line1, = ax.plot([], [], color='orange', label='Step')
# line2, = ax.plot([], [], color='blue', label='Smooth')
# line3, = ax.plot([], [], color='green', label='Sign Change')
# line4, = ax.plot([], [], color='purple', label='Two Fronts')
# ax.set_xlim(-10, 10)
# ax.set_ylim(-1, 3)
# ax.set_xlabel('x')
# ax.set_ylabel('u(x,t)')
# ax.legend()

# # --- Функция обновления кадров ---
# def update(frame):
#     t = t_vals[frame]
#     char_arg = t - integral
#     line1.set_data(x, f_step(char_arg))
#     line2.set_data(x, f_smooth(char_arg))
#     line3.set_data(x, f_sign_change(char_arg))
#     line4.set_data(x, f_two_fronts(char_arg))
#     return line1, line2, line3, line4

# # --- Анимация ---
# ani = FuncAnimation(fig, update, frames=len(t_vals), blit=True)

# # --- Сохраняем GIF ---
# ani.save('transport_animation.gif', writer='pillow', fps=10)

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # --- Параметры ---
# x = np.linspace(-10, 10, 1000)
# t_vals = np.linspace(0, 10, 100)
# a = lambda t: 1 + 0.3*np.sin(t)

# # --- Интеграл скорости по времени ---
# dt = t_vals[1] - t_vals[0]
# integral = np.cumsum(a(t_vals)) * dt  # интеграл от 0 до t

# # --- Начальные условия ---
# def f_step(s): return np.heaviside(s, 1)
# def f_smooth(s): return np.tanh(s)
# def f_sign_change(s): return np.sin(s)
# def f_two_fronts(s): return np.heaviside(s+2,1) + np.heaviside(s-3,1)

# # --- Создание фигуры ---
# fig, ax = plt.subplots(figsize=(8,4))
# line1, = ax.plot([], [], color='orange', label='Step')
# line2, = ax.plot([], [], color='blue', label='Smooth')
# line3, = ax.plot([], [], color='green', label='Sign Change')
# line4, = ax.plot([], [], color='purple', label='Two Fronts')
# ax.set_xlim(-10, 10)
# ax.set_ylim(-1, 3)
# ax.set_xlabel('x')
# ax.set_ylabel('u(x,t)')
# ax.legend()

# # --- Функция обновления кадров ---
# def update(frame):
#     t_int = integral[frame]
#     line1.set_data(x, f_step(-x + t_int))
#     line2.set_data(x, f_smooth(-x + t_int))
#     line3.set_data(x, f_sign_change(-x + t_int))
#     line4.set_data(x, f_two_fronts(-x + t_int))
#     return line1, line2, line3, line4

# # --- Анимация ---
# ani = FuncAnimation(fig, update, frames=len(t_vals), blit=True)

# # --- Сохраняем GIF ---
# ani.save('transport_animation_at.gif', writer='pillow', fps=10)

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # --- Параметры ---
# x = np.linspace(-10, 10, 1000)
# t_vals = np.linspace(0, 5, 100)

# # --- Функция аргумента характеристики ---
# def char_arg3(x, t):
#     return np.exp(-t) * (-1 - t + x)

# # --- Начальные условия ---
# def f_step(s): return np.heaviside(s, 1)
# def f_smooth(s): return np.tanh(s)
# def f_sign_change(s): return np.sin(s)
# def f_two_fronts(s): return np.heaviside(s+2,1) + np.heaviside(s-3,1)

# # --- Создание фигуры ---
# fig, ax = plt.subplots(figsize=(8,4))
# line1, = ax.plot([], [], color='orange', label='Step')
# line2, = ax.plot([], [], color='blue', label='Smooth')
# line3, = ax.plot([], [], color='green', label='Sign Change')
# line4, = ax.plot([], [], color='purple', label='Two Fronts')
# ax.set_xlim(-10, 10)
# ax.set_ylim(-1, 3)
# ax.set_xlabel('x')
# ax.set_ylabel('u(x,t)')
# ax.legend()

# # --- Функция обновления кадров ---
# def update(frame):
#     t = t_vals[frame]
#     arg = char_arg3(x, t)
#     line1.set_data(x, f_step(arg))
#     line2.set_data(x, f_smooth(arg))
#     line3.set_data(x, f_sign_change(arg))
#     line4.set_data(x, f_two_fronts(arg))
#     return line1, line2, line3, line4

# # --- Анимация ---
# ani = FuncAnimation(fig, update, frames=len(t_vals), blit=True)

# # --- Сохраняем GIF ---
# ani.save('transport_animation_x_minus_t.gif', writer='pillow', fps=10)

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from scipy.interpolate import interp1d

# # --- Параметры ---
# x = np.linspace(-10, 10, 1000)
# t_vals = np.linspace(0, 5, 100)  # меньшее t, чтобы значения не взлетали

# # --- Функция a(x-t) ---
# def a(s): 
#     return 1 + 0.5*np.sin(s)

# # --- Сетка для интеграла (узкая, чтобы значения были разумные) ---
# s_grid = np.linspace(-15, 15, 3000)
# inv_grid = 1/(a(s_grid) - 1)
# cumsum = np.cumsum(inv_grid)*(s_grid[1]-s_grid[0])
# integral_interp = interp1d(s_grid, cumsum, kind='linear', fill_value="extrapolate")

# # --- Начальные условия ---
# def f_step(s): return np.heaviside(s,1)
# def f_smooth(s): return np.tanh(s)
# def f_sign_change(s): return np.sin(s)
# def f_two_fronts(s): return np.heaviside(s+2,1) + np.heaviside(s-3,1)

# # --- Фигура ---
# fig, ax = plt.subplots(figsize=(8,4))
# line1, = ax.plot([], [], color='orange', label='Step')
# line2, = ax.plot([], [], color='blue', label='Smooth')
# line3, = ax.plot([], [], color='green', label='Sign Change')
# line4, = ax.plot([], [], color='purple', label='Two Fronts')
# ax.set_xlim(-10, 10)
# ax.set_ylim(-1.2, 3)
# ax.set_xlabel('x')
# ax.set_ylabel('u(x,t)')
# ax.legend()

# # --- Обновление кадров ---
# def update(frame):
#     t = t_vals[frame]
#     s_arg = x - t
#     # нормализуем интеграл, чтобы char_arg был в пределах видимого диапазона
#     char_arg = -t + integral_interp(s_arg)
#     char_arg = np.clip(char_arg, -10, 10)
#     line1.set_data(x, f_step(char_arg))
#     line2.set_data(x, f_smooth(char_arg))
#     line3.set_data(x, f_sign_change(char_arg))
#     line4.set_data(x, f_two_fronts(char_arg))
#     return line1, line2, line3, line4

# ani = FuncAnimation(fig, update, frames=len(t_vals), blit=True)
# ani.save('transport_animation_a_x_minus_t_clipped.gif', writer='pillow', fps=10)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # --- Параметры ---
# x = np.linspace(-10, 10, 1000)
# t_vals = np.linspace(0, 10, 100)
# a = 1  # константа

# # --- Начальные условия ---
# def f_step(s): return np.heaviside(s,1)
# def f_smooth(s): return np.tanh(s)
# def f_sign_change(s): return np.sin(s)
# def f_two_fronts(s): return np.heaviside(s+2,1) + np.heaviside(s-3,1)

# # --- Фигура ---
# fig, ax = plt.subplots(figsize=(8,4))
# line1, = ax.plot([], [], color='orange', label='Step')
# line2, = ax.plot([], [], color='blue', label='Smooth')
# line3, = ax.plot([], [], color='green', label='Sign Change')
# line4, = ax.plot([], [], color='purple', label='Two Fronts')
# ax.set_xlim(-10, 10)
# ax.set_ylim(-1.2, 3)
# ax.set_xlabel('x')
# ax.set_ylabel('u(x,t)')
# ax.legend()

# # --- Функция обновления кадров ---
# def update(frame):
#     t = t_vals[frame]
#     char_arg = x - a*t
#     line1.set_data(x, f_step(char_arg))
#     line2.set_data(x, f_smooth(char_arg))
#     line3.set_data(x, f_sign_change(char_arg))
#     line4.set_data(x, f_two_fronts(char_arg))
#     return line1, line2, line3, line4

# # --- Анимация ---
# ani = FuncAnimation(fig, update, frames=len(t_vals), blit=True)

# # --- Сохраняем GIF ---
# ani.save('transport_animation_a_const.gif', writer='pillow', fps=10)

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import quad

# # a(x)
# def a(x):
#     return 1 + 0.5*np.sin(x)

# # характеристическая функция phi(x,t) = t - ∫ 1/a(s) ds
# def phi(x, t):
#     # интеграл может быть x < 0 → меняем пределы
#     val, _ = quad(lambda s: 1/a(s), 0, x)
#     return t - val

# # сетка
# xs = np.linspace(-10, 10, 400)
# ts = np.linspace(0, 10, 400)

# X, T = np.meshgrid(xs, ts)

# # вычисляем phi на сетке
# PHI = np.zeros_like(X)
# for i in range(len(ts)):
#     for j in range(len(xs)):
#         PHI[i, j] = phi(xs[j], ts[i])

# # создаём уровни
# levels = np.linspace(PHI.min(), PHI.max(), 25)

# plt.figure(figsize=(10, 6))
# contour = plt.contour(X, T, PHI, levels=levels, linewidths=1.5)
# plt.clabel(contour, inline=True, fontsize=8)
# plt.xlabel("x")
# plt.ylabel("t")
# plt.title("Характеристики для u_t + a(x) u_x = 0")
# plt.grid(True)
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # ----- Параметры сетки -----
# h = 0.1
# xmax = 20
# xgrid = np.arange(-xmax, xmax + h, h)
# Nx = len(xgrid)

# t_max = 6.0
# cfl = 0.5

# # ----- Уравнение и аналитика -----
# def velocity(x, t):
#     return x - t + 1

# def u_init(x):
#     return np.where(x >= 0, 1.0, 0.0)

# def u_analytic(x, t):
#     # Соответствует характеристикам уравнения x' = x - t + 1
#     val = np.exp(-t) * (x - t)
#     return np.where(val >= 0, 1.0, 0.0)

# # ----- TVD Лимитеры -----
# def mc_limiter(a, b):
#     """Monotonized Central limiter (более острый чем minmod)"""
#     res = np.zeros_like(a)
#     mask = a * b > 0
#     c = 0.5 * (a + b)
#     # Формула: sign(a) * min(2|a|, 2|b|, |(a+b)/2|)
#     res[mask] = np.sign(a[mask]) * np.minimum(
#         np.minimum(2 * np.abs(a[mask]), 2 * np.abs(b[mask])), 
#         np.abs(c[mask])
#     )
#     return res

# def godunov_flux(uL, uR, a):
#     return np.where(a >= 0, a * uL, a * uR)

# # ----- Расчет правой части dU/dt -----
# def calc_rhs(u, t):
#     # 1. Наклоны (с фиктивными ячейками на границах - outflow)
#     u_left = np.roll(u, 1)
#     u_left[0] = u[0]
#     u_right = np.roll(u, -1)
#     u_right[-1] = u[-1]
    
#     slopes = mc_limiter((u - u_left)/h, (u_right - u)/h)
    
#     # 2. Потоки на интерфейсах (всего Nx + 1 грань)
#     # Координаты граней x_{i-1/2}
#     x_faces = xgrid - h/2
#     # Добавим последнюю грань
#     x_faces = np.append(x_faces, xgrid[-1] + h/2)
    
#     a_faces = velocity(x_faces, t)
    
#     # Реконструкция значений на гранях
#     # uL_face[i] - значение слева от грани i
#     uL_face = u + (h/2) * slopes
#     # uR_face[i] - значение справа от грани i
#     uR_face = u - (h/2) * slopes
    
#     # Сдвиги для получения пар (uL, uR) на каждой грани
#     # На грани i: слева ячейка i-1, справа ячейка i
#     uL_edge = np.zeros(Nx + 1)
#     uR_edge = np.zeros(Nx + 1)
    
#     uL_edge[1:-1] = uL_face[:-1]
#     uL_edge[0] = u[0] # Граничное условие слева
#     uL_edge[-1] = uL_face[-1]
    
#     uR_edge[1:-1] = uR_face[1:]
#     uR_edge[0] = uR_face[0]
#     uR_edge[-1] = u[-1] # Граничное условие справа
    
#     fluxes = godunov_flux(uL_edge, uR_edge, a_faces)
    
#     # 3. dU/dt = -(F_{i+1/2} - F_{i-1/2})/h
#     rhs = -(fluxes[1:] - fluxes[:-1]) / h
#     return rhs

# # ----- Основной цикл решения -----
# u = u_init(xgrid)
# t = 0.0

# u_history = [u.copy()]
# t_history = [0.0]

# while t < t_max:
#     a_max = np.max(np.abs(velocity(xgrid, t)))
#     tau = cfl * h / a_max
#     if t + tau > t_max:
#         tau = t_max - t
    
#     # RK2 (Heun's Method)
#     k1 = calc_rhs(u, t)
#     u_star = u + tau * k1
    
#     k2 = calc_rhs(u_star, t + tau)
#     u = 0.5 * u + 0.5 * (u_star + tau * k2)
    
#     t += tau
#     u_history.append(u.copy())
#     t_history.append(t)

# # ----- Визуализация (Анимация) -----
# fig, ax = plt.subplots(figsize=(10, 6))
# line_num, = ax.plot(xgrid, u_history[0], 'r-', lw=2, label='TVD (MC + RK2)')
# line_ana, = ax.plot(xgrid, u_analytic(xgrid, 0), 'b--', lw=1.5, label='Аналитическое')

# ax.set_ylim(-0.2, 1.2)
# ax.set_xlim(-5, 15)
# ax.grid(True, alpha=0.3)
# ax.legend()
# title = ax.set_title(f"Time: 0.00")

# def update(frame):
#     line_num.set_ydata(u_history[frame])
#     line_ana.set_ydata(u_analytic(xgrid, t_history[frame]))
#     title.set_text(f"Time: {t_history[frame]:.2f}")
#     return line_num, line_ana, title

# ani = FuncAnimation(fig, update, frames=range(0, len(u_history), 5), interval=30, blit=True)
# plt.show()


# import numpy as np
# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # -------------------- Параметры --------------------
# xmax = 1.0
# xmin = -1.0
# len_x = 161
# xgrid = np.linspace(xmin, xmax, len_x)
# h = xgrid[1] - xgrid[0]

# tmax = 0.5
# nstep = 100
# tau = tmax / nstep
# theta = 0.8  # θ-схема

# # -------------------- Начальное и аналитическое условие --------------------
# def probfunc(x):
#     return 1.0 if x < 0 else 0.0  # ступенька

# def uAnalytic(x, t):
#     # пример аналитики для a = x - t
#     return 1.0 if x - t < 0 else 0.0

# # -------------------- Инициализация --------------------
# uTheta = np.zeros((nstep, len_x))
# uTheta[0, :] = np.array([probfunc(x) for x in xgrid])

# # -------------------- Основной цикл --------------------
# for n in range(nstep - 1):
#     tCurr = n * tau
#     lambdaGrid = (xgrid - tCurr) * tau / h

#     main_diag = 1 + theta * lambdaGrid
#     lower_diag = -theta * lambdaGrid[1:] / 2
#     upper_diag = -theta * lambdaGrid[:-1] / 2

#     diagonals = [main_diag, lower_diag, upper_diag]
#     offsets = [0, -1, 1]
#     ACN = diags(diagonals, offsets, format='csc')

#     rhs = np.zeros(len_x)
#     rhs[0] = uAnalytic(xmin, tCurr)
#     rhs[1:] = ((1 - theta) * (1 - lambdaGrid[1:] / 2) * uTheta[n, 1:] +
#                (1 - theta) * (lambdaGrid[1:] / 2) * uTheta[n, :-1])

#     uTheta[n + 1, :] = spsolve(ACN, rhs)
#     uTheta[n + 1, -1] = uAnalytic(xmax, tCurr)

# # -------------------- Анимация --------------------
# fig, ax = plt.subplots(figsize=(8,5))
# line, = ax.plot(xgrid, uTheta[0, :], color='blue')
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(-0.1, 1.1)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# ax.set_title('θ-схема для a = x - t')

# def update(frame):
#     line.set_ydata(uTheta[frame, :])
#     ax.set_title(f'θ-схема для a = x - t, t = {frame*tau:.3f}')
#     return line,

# ani = FuncAnimation(fig, update, frames=nstep, interval=50)
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -----------------------------
# Параметры сетки и времени
h = 0.5
tau = 0.012
xmax = 10
nstep = 1667
xgrid = np.arange(-xmax, xmax+h, h)

# -----------------------------
# Исходная функция и аналитическое решение
def probfunc(x):
    return np.heaviside(x, 1)  # UnitStep

def u_analytic(x, t):
    return probfunc(np.exp(-t)*(x-1-t)+np.exp(-t))

def uLeft(t):
    return u_analytic(-xmax, t)

def uRight(t):
    return u_analytic(xmax, t)

# -----------------------------
# TVD / MUSCL limiters
def phi_minmod(r):
    return np.maximum(0, np.minimum(1, r))

def phi_superbee(r):
    return np.maximum(0, np.maximum(np.minimum(2*r, 1), np.minimum(r, 2)))

# -----------------------------
# Инициализация массивов для всех схем
scheme_names = [
    "Analytic",
    "Явная против потока",
    "Консервативная против потока",
    "Лакс-Фридрихс",
    "Лакс-Вендрофф",
    "Явная Лакса",
    "Центральная с вязкостью",
    "Мак-Кормака",
    "Русанова",
    "Годунова",
    "TVD Minmod",
    "MUSCL Superbee"
]

schemes = {}
for name in scheme_names:
    schemes[name] = np.zeros((nstep+1, len(xgrid)))
    if name == "Analytic":
        schemes[name][0,:] = u_analytic(xgrid, 0)
    else:
        schemes[name][0,:] = probfunc(xgrid)

# -----------------------------
# Явная схема (против потока)
u = schemes["Явная против потока"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(1,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        if ai > 0:
            u[n+1,i] = u[n,i] - (abs(ai)*tau/h)*(u[n,i]-u[n,i-1])
        else:
            u[n+1,i] = u[n,i] - (abs(ai)*tau/h)*(u[n,i+1]-u[n,i])
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Явная схема консервативная (против потока)
u = schemes["Консервативная против потока"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(1,len(xgrid)-1):
        aiL = 0.5*((xgrid[i-1]-tCurr)+(xgrid[i]-tCurr))
        aiR = 0.5*((xgrid[i]-tCurr)+(xgrid[i+1]-tCurr))
        FLeft = aiL*u[n,i-1] if aiL>0 else aiL*u[n,i]
        FRight = aiR*u[n,i] if aiR>0 else aiR*u[n,i+1]
        u[n+1,i] = u[n,i] - (tau/h)*(FRight-FLeft)
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Лакс-Фридрихс
u = schemes["Лакс-Фридрихс"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(1,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        lam = ai*tau/h
        u[n+1,i] = 0.5*(u[n,i+1]+u[n,i-1]) - 0.5*lam*(u[n,i+1]-u[n,i-1])
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Лакс-Вендрофф
u = schemes["Лакс-Вендрофф"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(1,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        lam = ai*tau/h
        u[n+1,i] = u[n,i] - 0.5*lam*(u[n,i+1]-u[n,i-1]) + 0.5*lam**2*(u[n,i+1]-2*u[n,i]+u[n,i-1])
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Явная Лакса
u = schemes["Явная Лакса"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(1,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        u[n+1,i] = 0.5*(u[n,i+1]+u[n,i-1]) - (tau/(2*h))*(ai*u[n,i+1]-ai*u[n,i-1])
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Центральная с вязкостью
u = schemes["Центральная с вязкостью"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(1,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        central = -(tau/(2*h))*(ai*u[n,i+1]-ai*u[n,i-1])
        visc = (h**2/(2*tau))*(u[n,i+1]-2*u[n,i]+u[n,i-1])
        u[n+1,i] = u[n,i]+central+tau*visc
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Мак-Кормака
u = schemes["Мак-Кормака"]
for n in range(nstep):
    tCurr = n*tau
    uStar = np.zeros_like(xgrid)
    for i in range(len(xgrid)-1):
        ai = xgrid[i]-tCurr
        uStar[i] = u[n,i] - (tau/h)*(ai*u[n,i+1]-ai*u[n,i])
    for i in range(1,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        u[n+1,i] = 0.5*(u[n,i]+uStar[i] - (tau/h)*(ai*uStar[i]-ai*uStar[i-1]))
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Русанова
u = schemes["Русанова"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(1,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        if ai>0:
            fluxR = 0.5*ai*(u[n,i]+u[n,i+1]) - 0.5*ai*(u[n,i+1]-u[n,i])
            fluxL = 0.5*ai*(u[n,i-1]+u[n,i]) - 0.5*ai*(u[n,i]-u[n,i-1])
        else:
            fluxR = 0.5*ai*(u[n,i]+u[n,i+1]) + 0.5*abs(ai)*(u[n,i+1]-u[n,i])
            fluxL = 0.5*ai*(u[n,i-1]+u[n,i]) + 0.5*abs(ai)*(u[n,i]-u[n,i-1])
        u[n+1,i] = u[n,i] - (tau/h)*(fluxR-fluxL)
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Годунова
u = schemes["Годунова"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(1,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        fluxR = ai*u[n,i] if ai>0 else ai*u[n,i+1]
        fluxL = ai*u[n,i-1] if ai>0 else ai*u[n,i]
        u[n+1,i] = u[n,i] - (tau/h)*(fluxR-fluxL)
    u[n+1,0] = uLeft(tCurr)
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# TVD Minmod
u = schemes["TVD Minmod"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(2,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        lam = abs(ai)*tau/h
        if ai>0:
            delm = u[n,i]-u[n,i-1]
            delp = u[n,i+1]-u[n,i]
            delmleft = u[n,i-1]-u[n,i-2]
            delpleft = delm
        else:
            delm = u[n,i+1]-u[n,i]
            delp = u[n,i]-u[n,i-1]
            delmleft = delp
            delpleft = delm
        r = 0 if delp==0 else delm/delp
        phi = phi_minmod(r)
        rleft = 0 if delpleft==0 else delmleft/delpleft
        phileft = phi_minmod(rleft)
        correction = phi*delp - phileft*delm
        u[n+1,i] = u[n,i] - np.sign(ai)*lam*delm - (lam*(1-lam)/2)*correction
    u[n+1,0] = uLeft(tCurr)
    u[n+1,1] = u[n+1,2]
    u[n+1,-2] = u[n+1,-3]
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# MUSCL Superbee
u = schemes["MUSCL Superbee"]
for n in range(nstep):
    tCurr = n*tau
    for i in range(2,len(xgrid)-1):
        ai = xgrid[i]-tCurr
        lam = abs(ai)*tau/h
        if ai>0:
            delm = u[n,i]-u[n,i-1]
            delp = u[n,i+1]-u[n,i]
            delmleft = u[n,i-1]-u[n,i-2]
            delpleft = delm
        else:
            delm = u[n,i+1]-u[n,i]
            delp = u[n,i]-u[n,i-1]
            delmleft = delp
            delpleft = delm
        r = 0 if delp==0 else delm/delp
        phi = phi_superbee(r)
        rleft = 0 if delpleft==0 else delmleft/delpleft
        phileft = phi_superbee(rleft)
        correction = phi*delp - phileft*delm
        u[n+1,i] = u[n,i] - np.sign(ai)*lam*delm - (lam*(1-lam)/2)*correction
    u[n+1,0] = uLeft(tCurr)
    u[n+1,1] = u[n+1,2]
    u[n+1,-2] = u[n+1,-3]
    u[n+1,-1] = uRight(tCurr)

# -----------------------------
# Анимация
colors = ['k','r','g','b','m','c','orange','brown','purple','pink','lime','navy']

fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
lines = {}

for i,(key,color) in enumerate(zip(scheme_names, colors)):
    line, = ax.plot([], [], color=color, lw=2, label=key)
    lines[key] = line

ax.legend(loc='upper left', fontsize=8)

def init():
    for line in lines.values():
        line.set_data([], [])
    return list(lines.values())

def update(nframe):
    t = nframe * tau
    for key, data in schemes.items():
        lines[key].set_data(xgrid, data[nframe])
    ax.set_title(f"t = {t:.2f}")
    return list(lines.values())

# ---------- Анимация ----------
writer = FFMpegWriter(fps=20, metadata=dict(artist='Damir'), bitrate=1800)
ani = FuncAnimation(fig, update, frames=nstep+1, init_func=init, blit=True)

ani.save("all_schemes_full.mp4", writer=writer)
plt.show()
