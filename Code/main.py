import math
import random as rd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from expressions import *


# Source: CODATA (2022)

c = 299_792_458                     # speed of light
e = 1.602_176_634e-19               # elemtary charge
m_e = 9.109_383_7139e-31            # electron mass
m_p = 1.672_621_925_95e-27          # proton mass
h = 6.626_070_15e-34                # Planck's constant
hbar = h / (2 * math.pi)            # reduced planck's constant
u = 1.66e-27                        # atomic mass
epsilon_0 = 8.854_187_8188e-12      # vacuum permittivity
mu_0 = 1.256_637_061_27e-6          # vacuum permeability
N_A = 6.022_140_76e23               # Avogadro's constant
alpha = 7.297_352_5643e-3           # fine structure constant
k_B = 1.380649e-23                  # Blotzman constant
G = 6.67430e-11                     # gravitational constant

m_D = 3.34358377683e-27              # deuteron mass


def vorbereitung():
    print("--- Vorbereitung ---")

    lambda_H = 288 * (m_e + m_p) * epsilon_0 ** 2 * h ** 3 * c / (5 * m_e * m_p * e ** 4)
    lambda_D = 288 * (m_e + m_D) * epsilon_0 ** 2 * h ** 3 * c / (5 * m_e * m_D * e ** 4)
    delta_lambda = lambda_H - lambda_D
    rel_difference = delta_lambda / lambda_H

    print(f"lambda_H = {lambda_H:.6g}")
    print(f"lambda_D = {lambda_D:.6g}")
    print(f"delta_lambda = {delta_lambda:.6g}")
    print(f"rel_difference = {rel_difference:.6g}")

def tv1():
    print("--- Teilversuch 1 ---")

    measured_angles_1 = [ 67 + 55 / 60, 98 + 25 / 60, 93 + 20 / 60, 91 + 5 / 60, 85 + 55 / 60 ]
    measured_angles_2 = [ 69 + 45 / 60, 94 + 50 / 60, 92 + 50 / 60, 92 + 10 / 60, 91 + 50 / 60, 91 + 30 / 60, 91 + 15 / 60, 89 + 20 / 60, 89 + 5 / 60, 87 + 0 / 60, 84 + 55 / 60, 83 + 55 / 60, 82 + 25 / 60 ]

    eval_tv1(measured_angles_1, "Messreihe 1")
    eval_tv1(measured_angles_2, "Messreihe 2", indices=[0, 8, 9, 10], calculate_rydberg=True)

    g = 1 / (1200 / 1e-3)
    b = 2.5e-3
    A = b / g
    l = 656e-9
    dl = l / A
    print(f"A = {A:.3f}, dl = {dl:.3g}")

def eval_tv1(measured_angles, name, indices=None, calculate_rydberg=False):
    print(name)

    literature_values = [ 656.3e-9, 486.1e-9, 434.0e-9, 410.2e-9 ]

    rydberg_values = [ 0 ] * len(literature_values)

    measured_angles_var = [ Var(a, 5 / 60, "") for a in measured_angles ]
    conversion_factor = Const(2 * math.pi / 180) # multiply by 2 and convert to radians
    alphas_var = [ Mult(Sub(measured_angles_var[i], measured_angles_var[0]), conversion_factor) for i in range(1, len(measured_angles)) ]

    # for a in alphas_var:
    #     print(a.eval())

    # sin alpha = lambda / g
    # lambda = g * sin alpha
    g = Const(1 / (1200 / 1e-3)) # 1200 Striche / mm

    lambdas_var = [ Mult(Add(Sin(alpha), Const(math.sin(2 * math.pi / 180))), g) for alpha in alphas_var ]
    for i in range(len(lambdas_var)):
        value = lambdas_var[i].eval()
        uncertainty = gaussian(lambdas_var[i], measured_angles_var)
        text = f"Value = {value:12.4g}, Uncertainty = {uncertainty:12.3g}"
        # text = f"    \\qty\u007b{value * 1e9:5.4g} \\pm {uncertainty * 1e9:5.3g}\u007d\u007b\\nm\u007d"
        index = -1
        if indices is None:
            index = i
        elif i in indices:
            index = indices.index(i)
            rydberg_values[index] = value
        if index >= 0:
            deviation = abs(value - literature_values[index]) / uncertainty
            text += f", deviation = {deviation:12.3g}"
        print(text)
    
    if not calculate_rydberg:
        return

    lambda_inv = [ 1 / l for l in rydberg_values]
    x_values = [ (1 / 2**2 - 1 / m**2) for m in range(3, 7) ]

    params, cov = np.polyfit(x_values, lambda_inv, 1, cov=True)
    uncertainties = np.sqrt(np.diag(cov))
    print("Fitparameter (m, b):")
    print("Werte:          " + "".join(f"{p:10.4g}" for p in params))
    print("Unsicherheiten: " + "".join(f"{u:10.4g}" for u in uncertainties))

    (m, b) = params

    R_y = 1.09737e7
    deviation_unc = abs(m - R_y) / uncertainties[0]
    deviation_perc = abs(m - R_y) / R_y
    print(f"Abweichung von R_y: {deviation_unc:.2f} Standardabweichungen, {deviation_perc:.2f} %")


    xrange = np.linspace(min(x_values), max(x_values), 100)
    fit = np.polyval(params, xrange)

    pp = PdfPages(f"../Abbildungen/Graph_TV1.pdf")
    fig = plt.figure()

    plt.plot(x_values, lambda_inv, "o", label="Messwerte")
    plt.plot(xrange, fit, "-", label=f"Fit (y = mx + b, m = {m:.4g}, b = {b:.4g}")

    plt.xlabel("$1 / n^2 - 1 / m^2$ ($n = 2$)")
    plt.ylabel("$1 / \\lambda$ [$m^{-1}$]")
    # plt.title("Zündspannung vs. Abstand * Druck")
    plt.legend()

    fig.tight_layout()
    pp.savefig()
    pp.close()

def tv2():
    print("--- Teilversuch 2 ---")
    
    alpha_0 = Var(360 - (353 + 30 / 60), 5 / 60, "")
    alpha_1 = Var(45 + 50 / 60, 5, "")
    alpha = Mult(Const(2 * math.pi / 180), Sub(alpha_1, alpha_0))
    g = Const(1 / (2400 / 1e-3))
    d = Var(0.235, 0.002, "")
    dt = Var(42e-6, 4e-6, "")
    dt_CCD = Var(8.2e-3, 0.001e-3, "")
    b_CCD = Const(0.02875)
    dl = Mult(Div(Mult(g, Sin(alpha)), d), Mult(Div(dt, dt_CCD), b_CCD))

    variables = [alpha_0, alpha_1, d, dt, dt_CCD ]

    value = dl.eval()
    uncertainty = gaussian(dl, variables)
    literature = 1.78576e-10
    deviation = abs(value - literature) / uncertainty
    print(f"Value = {value:12.4g}, Uncertainty = {uncertainty:12.4g}, Deviation = {deviation:.2f}")

vorbereitung()
tv1()
tv2()
