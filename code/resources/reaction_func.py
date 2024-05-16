import numpy as np

def initalise_bump(shape):
    a  = np.zeros(shape)
    step = 0.3
    if len(a.shape) == 1:
        a[int(a.shape[0] / 2)] = step
    elif len(a.shape) == 2:
        a[int(a.shape[0] / 2), int(a.shape[1] / 2)] = step

    return(
        a,
        np.zeros(shape)
    )

def fh_ng():
    Da, Db, alpha, beta = 0.01, 1, -0.005, 10
    def Ra(U): return U[0] - U[0] ** 3 - U[1] + alpha
    def Rb(U): return (U[0] - U[1]) * beta
    return [Da, Db], [Ra, Rb], initalise_bump, (-1,1)

def schnak():
    Da, Db, alpha, beta = 0.01, 1., 0.1, 0.9 
    def Ra(U): return alpha - U[0] + U[0] ** 2 * U[1]
    def Rb(U): return beta - U[0] ** 2 * U[1]

    def initalise_schnak_random(shape):
        sol = (alpha + beta, beta / (alpha + beta)**2)
        step = 0.005
        return(
            np.random.normal(loc=sol[0], scale=step, size=shape),
            np.random.normal(loc=sol[1], scale=step, size=shape),
        )
    
    return [Da, Db], [Ra, Rb], initalise_schnak_random

def gray_scott():
    Da, Db = 0.01,  0.005
    alpha, beta = 0.04, 0.06
    def Ra(U): return -U[0]*U[1]**2 + alpha * (1-U[0])
    def Rb(U): return U[0]*U[1]**2 - (alpha + beta) * U[1]

    def initalise_gray_scott(shape):
        alpha, beta = 0.4, 0.6
        v0 = 0
        u0 = beta * v0 / alpha + v0 + 1
        a  = np.ones(shape) * u0
        b = np.ones(shape) * v0
        if len(a.shape) == 1:
            a[int(a.shape[0] / 2)] += 0.3
            b[int(a.shape[0] / 2)] += 0.3
        elif len(a.shape) == 2:
            a[int(a.shape[0] / 2), int(a.shape[1] / 2)] += 0.3
            b[int(a.shape[0] / 2), int(a.shape[1] / 2)] += 0.3
        return(
            a,
            b
        )

    return [Da, Db], [Ra, Rb], initalise_gray_scott

def tbv_model_nondim():   
    sigma = 0.75
    Dt = 0 
    Db = 0.01 / sigma
    Dv = 0.0001 / sigma

    alpha = 0.04
    delta = 0.03
    rho = 0.1
    k = 16
    eta = 2
    beta = 1.5
    mu = 0.2
    def f(U): 
        return alpha * np.reciprocal(1 + U[2]) - delta * U[0] + np.divide(rho * U[0] ** 2, (k ** 2 + U[0] ** 2))
    def g(U): 
        return (1 - U[1]) * U[1] - eta * U[2] ** 2 * U[1]
    def h(U): 
        return beta * np.reciprocal(1 + U[0]) * U[2] ** 2 * U[1] - mu * U[2]    

    def initalise_tbv_nondim(shape):
        np.random.seed(1851321)
        sol = [0.904423, 0.516369, 0.491747]
        a  = np.ones(shape) * sol[0]
        b = np.ones(shape) * sol[1]
        step = 0.1
        return(
            a,
            b,
            np.random.normal(loc=sol[2], scale=step, size=shape)
        )
    
    return [Dt, Db, Dv], [f, g, h], initalise_tbv_nondim

def tbv_model_dim():
    Dt, Db, Dv = 0., 0.01, 0.0001

    weight_size_ratio = 0.04

    rhoo = 1.5 * weight_size_ratio
    deltat = 0.0225
    rhot = 3.75 * weight_size_ratio
    kt = 800 * weight_size_ratio
    ro = 0.005 / weight_size_ratio

    sigma = 0.75
    rt = 0.02 / weight_size_ratio
    gamma = 0.15
    KB = 2500 * weight_size_ratio
    rhob = 3.75e-05 / (weight_size_ratio**2)
    rhov = 2.25e-06 / (weight_size_ratio**2)

    def f(U): 
        T,B,V = U
        return rhoo * (1 / (1 + ro * V)) - deltat * T + rhot * T ** 2 / (kt ** 2 + T ** 2)
    def g(U): 
        T,B,V = U
        return sigma * (1 - B / KB) * B - rhob * V ** 2 * B
    def h(U): 
        T,B,V = U
        return rhov * (1 / (1 + rt * T)) * V ** 2 * B - gamma * V
    
    def initalise_tbv_dim(shape):
        np.random.seed(1851321)
        sol = [1.80885, 51.6369,3.93398] 
        a  = np.ones(shape) * sol[0]
        b = np.ones(shape) * sol[1]
        step = 0.1
        return(
            a,
            b,
            np.random.normal(loc=sol[2], scale=step, size=shape)
        )
    
    return [Dt, Db, Dv], [f, g, h], initalise_tbv_dim
