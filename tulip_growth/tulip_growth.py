"""
Analysis on Tulip Growth and Curvature
Finds parameters for tulip growth dynamics
Author: Aidan Wong
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

def log_graph(x, y, guess, L0, show_cov=False):
    # Finds curve of best fit on a logistic function
    # Returns constants and coefficients used
    
    def test_func(t, rho, xi):
        return L0 * np.exp(rho*t) / (1 + (np.exp(rho*t) - 1) / xi)
    
    # Find curve of best fit
    param, param_cov = curve_fit(test_func, x, y, p0=guess, maxfev=20000)    
    rho, xi = param
    xfit = np.linspace(min(x),max(x), 100)
    yfit = test_func(xfit, rho, xi)
    
    # Find Uncertainty of fit parameters
    rho_uncer=np.sqrt(param_cov[0,0])
    xi_uncer=np.sqrt(param_cov[1,1]) 
    yerr = np.abs(test_func(x,rho_uncer,xi_uncer))
        
    # Graph x vs y
    plt.scatter(x, y, s=16, color="black")
    if show_cov: plt.errorbar(x, y, yerr=yerr,color='black',fmt=".", label="Uncertainty")
    plt.plot(xfit, yfit, label="time_multurve of Best Fit")
    return param, (rho_uncer, xi_uncer)

def wolpert_graph(x, y, guess=np.ones(4), show_cov=False):
    # Finds curve of best fit given a function
    # Returns constants and coefficients used

    def test_func(T, rhoo, deltat, rhoa, k):
        return rhoo - deltat * (T) + rhoa * (T) ** 2 / (k ** 2 + (T) ** 2)
    
    # Find curve of best fit
    param, param_cov = curve_fit(test_func, x, y, p0=guess, maxfev=20000)    
    rhoo, deltat, rhoa, k = param
    xfit = np.linspace(min(x),max(x), 100)
    yfit = test_func(xfit, rhoo, deltat, rhoa, k)

    # Find Uncertainty of fit parameters
    param_uncer = np.sqrt(np.diag(param_cov))
    rhoo_unc, deltat_unc, rhoa_unc, k_unc = param_uncer
    yerr = np.abs(test_func(x, rhoo_unc, deltat_unc, rhoa_unc, k_unc))

    # Graph x vs y
    plt.scatter(x, y, s=16, color="black")
    if show_cov: plt.errorbar(x, y, yerr=yerr, color='black',fmt=".")
    plt.plot(xfit, yfit, label="Wolpert Function of Best Fit")
    return param, param_uncer

def cubic_graph(x, y, show_cov=False):
    # Finds curve of best fit given a function
    # Returns constants and coefficients used
    
    def test_func(t, a, b, c, d):
        return a*t**3 + b*t**2 + c*t + d
    
    # Find curve of best fit
    param, param_cov = curve_fit(test_func, x, y, maxfev=20000)    
    a, b, c, d = param
    xfit = np.linspace(min(x),max(x), 100)
    yfit = test_func(xfit, a, b, c, d)

    # Find Uncertainty of fit parameters
    param_uncer = np.sqrt(np.diag(param_cov))
    a_unc, b_unc, c_unc, d_unc = param_uncer
    yerr = np.abs(test_func(x, a_unc, b_unc, c_unc, d_unc))

    # Graph x vs y
    plt.scatter(x, y, s=16, color="black")
    if show_cov: plt.errorbar(x, y, yerr=yerr,color='black',fmt=".", label="Experimental Data")
    plt.plot(xfit, yfit, label="Function of Best Fit")
    return param, param_uncer

def curver(X, y, r, new_L0, original_L0, flip=False):
    N = np.shape(y)[0]
    L, r = np.ptp(y, axis=1).reshape(N,1), r.reshape(N, 1)
    theta_arr = 0.5 * L / r * np.linspace(-1, 1, np.shape(y)[1])
    if flip == True:
        x = -original_L0 / new_L0 * r * np.cos(theta_arr)
        xx = x + (X - np.min(x, axis=1)).reshape(N, 1)
    else:
        x = original_L0 / new_L0 * r * np.cos(theta_arr)
        xx = x + (X - np.max(x, axis=1)).reshape(N, 1)
    yy = r * np.sin(theta_arr)
    return xx, yy

def para(x, y, a):
    N = np.shape(y)[0]
    hL = 0.5 * np.ptp(y, axis=1)
    def arc_length(b): return 0.5 * b * np.sqrt(4 * a**2 * b**2 + 1) + np.arcsinh(2 * a * b) / (4 * a) - hL
    hy = fsolve(arc_length, hL)
    yy = hy.reshape(N, 1) * np.linspace(-1, 1, np.shape(y)[1])
    xx = -a.reshape(N, 1) * yy ** 2 + x.reshape(len(x), 1)
    return xx, yy

def extract_chord_length_and_width(forward_growth=False):
    data = np.loadtxt("tulip_growth/tulip_growth.csv", delimiter=",", skiprows=1, usecols=(0,1,2))
    data = data.T
    
    # dist: distance of the chord along proximodistal axis of petal
    # a: half chord length along mediolateral axis
    # b: chord width along proximodistal axis
    dist, a, b = map(np.array, data)
    N = len(dist)

    if forward_growth:
        a, b = map(np.flipud, (a, b))

    # printed tulip to real tulip scale factor
    scale = 0.263
    dist, a, b = map(lambda x: scale * x, (dist, a, b))

    # radius of circle based on chord
    r = (a**2 + b**2) / (2 * b)
    arc_length = 2 * r * np.arcsin(a / r)

    r = (a**2 + b**2) / (2 * b)
    a1, b1 = 0.05 * scale, 0.05 * scale
    uncer_r = np.sqrt((a**2 * a1**2)/b**2 + ((a**2 - b**2)**2*b1**2)/(4*b**4))
    uncer_l = np.sqrt((4*b**2*(a1**2*b**2 + a**2*b1**2) - 4*a*b*(2*a1**2*b**2 + (a**2 - b**2)*b1**2)*np.arctan((2*a*b)/(a**2 - b**2)) + \
                        (a**4*b1**2 + b**4*b1**2 + 2*a**2*b**2*(2*a1**2 - b1**2))*np.arctan((2*a*b)/(a**2 - b**2))**2)/b**4)
    
    return dist, arc_length, r

def display_param(title, param_names, param, param_uncer):
    # Prints out the value of each parameter along with its sqrt(covariance) and covariance
    border = 100 * "#"
    print(border + "\n" + title + "\n")
    for i in range(len(param)):
        print(param_names[i] + " = {:.6f} ± {:.6f}, Cov R^2 = {:.6f}".format(param[i], param_uncer[i], param_uncer[i]**2))

def draw_log_graph(x, y, init_width, forward_growth=False):
    # Determine and graph the growth of the petal cells

    param_names = ["rho", "xi"]

    # Different guesses for forwards and backwards petal growth
    if forward_growth: 
        guess = np.array([0.727, 9.936])
    else:
        guess = np.array([-0.7, 1.01])
    
    param, param_uncer = log_graph(x, y, guess, init_width)

    display_param("Logistic Growth", param_names, param, param_uncer)
    
    # plt.title("Logistic Growth of Petal Width")
    plt.xlabel("Proximodistal Axis (cm)", fontsize=16)
    plt.ylabel("Arc Length of Band (cm)", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()
    return param
    
def draw_radial_graph(x, y):
    # Determine the value of the radius across time

    cub_title = "Cubic Radial Function"
    wol_title = "Wolpert Radial Function"
    cub_param_names = "abcd"
    wol_param_names = ["rhoo", "deltat", "rhoa", "k"]

    guess = np.array([3, 2, 20, 10]) # parameter guess for Wolpert function
    
    cub_param, cub_param_uncer = cubic_graph(x, y)
    wol_param, wol_param_uncer = wolpert_graph(x, y, guess)
    
    display_param(cub_title, cub_param_names, cub_param, cub_param_uncer)
    display_param(wol_title, wol_param_names, wol_param, wol_param_uncer)

    plt.xlabel("Proximodistal Axis (cm)", fontsize=16)
    plt.ylabel("Radius (cm)", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()
    return cub_param

def draw_petal(log_param, cub_param, new_L0, original_L0, forward_growth):
    # Draw a sample petal

    N = 30 # Number of arcs
    res = 80 # Number of points for each arcs
    time_mult = 1 # Scale factor of the time

    rho, xi = log_param

    # Create an array of vertical lines of length 'arc_length' at positions T1
    T1 = np.linspace(0, 6.35 * time_mult, N) # 6.35 is the real petal length from data
    arc_length =  new_L0 * np.exp(rho*T1/time_mult) / (1 + (np.exp(rho*T1/time_mult) - 1) / xi)
    r = new_L0 / original_L0 * np.poly1d(cub_param)(T1)
    x1 = arc_length.reshape(len(arc_length), 1) * np.linspace(0, 1, res)
    x1 -= np.median(x1, axis=1).reshape(len(x1), 1)
    t1 = T1.reshape(N, 1) * np.ones(res)
    
    # Curve the straight lines
    t2, x2 = curver(T1, x1, r, new_L0, original_L0, flip=(not forward_growth))

    # Graph the total result
    for i in range(1, N): 
        plt.plot(t1[i], x1[i], c="b")
        plt.plot(t2[i], x2[i], c="g")
    plt.xlabel("Proximodistal Axis (cm)", fontsize=16)
    plt.ylabel("Mediolateral Axis (cm)", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()

def main():
    # Determine if you want forward or backwards growth (small to big, or big to small)
    forward_growth = True
    init_widths = (0.526, 4.8019) # Real arc lengths of the petal in cm

    if forward_growth: 
        init_width = np.min(init_widths)
    else:
        init_width = np.max(init_widths)

    # Draw arc growth and radius vs proximodistal axis
    dist, arc_length, radius = extract_chord_length_and_width(forward_growth)
    log_param = draw_log_graph(dist, arc_length, init_width, forward_growth)
    cub_param = draw_radial_graph(dist, radius)

    # Draw sample petal from parameter results
    custom_init_width = 10
    draw_petal(log_param, cub_param, custom_init_width, init_width, forward_growth)

main()