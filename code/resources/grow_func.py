import numpy as np

GROW_FORWARD_INIT_WIDTH = 0.526
GROW_FORWARD_CURVATURE_CONST = 12
GROW_BACKWARD_INIT_WIDTH = 4.8019
GROW_BACKWARD_CURVATURE_CONST = 1.31447

def growno():
    def r(t): return 1
    def rt(t): return 0
    return r, rt

def growexp():
    rho = 0.3
    def r(t): return np.exp(rho * t)
    def rt(t): return rho*np.exp(rho*t)
    return  r, rt

def growlin():
    rho = 0.002 # lin
    def r(t): return 1 + rho*t
    def rt(t): return rho
    return  r, rt

def growlog():
    rho, xi = 0.01, 26.0 # 0.08, 2 # log
    def r(t): return np.exp(rho*t) / (1 + (np.exp(rho*t) - 1) / xi)
    def rt(t): return rho * r(t) * (1 - r(t) / xi)
    return r, rt

def growforward(width=1, time_mult=1):
    rho, xi = 0.727, 9.936
    init_width = GROW_FORWARD_INIT_WIDTH
    def r(t): return np.exp(rho*t/time_mult) / (1 + (np.exp(rho*t/time_mult) - 1) / xi)
    def rt(t): return rho/time_mult * r(t) * (1 - r(t) / xi)
    a, b, c, d = -0.037205, 0.454, -1.565, 2.682
    def radius(t): return width / init_width * (a*(t/time_mult)**3 + b*(t/time_mult)**2 + c*(t/time_mult) + d)
    return r, rt, radius

def growbackward(width=1, time_mult=1):
    rho, xi = -0.805, 1.068
    init_width = GROW_BACKWARD_INIT_WIDTH
    def r(t): return np.exp(rho*t/time_mult) / (1 + (np.exp(rho*t/time_mult) - 1) / xi)
    def rt(t): return rho/time_mult * r(t) * (1 - r(t) / xi)
    a, b, c, d = 0.037205, -0.251, 0.281, 1.535
    def radius(t): return width / init_width * (a*(t/time_mult)**3 + b*(t/time_mult)**2 + c*(t/time_mult) + d)
    return r, rt, radius