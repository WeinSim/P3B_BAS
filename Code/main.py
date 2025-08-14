import math
import random as rd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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

m_D = 3.34358377683-27              # deuteron mass


def vorbereitung():
    lambda_H = 288 * (m_e + m_p) * epsilon_0 ** 2 * h ** 3 * c / (5 * m_e * m_p * e ** 4)
    lambda_D = 288 * (m_e + m_D) * epsilon_0 ** 2 * h ** 3 * c / (5 * m_e * m_D * e ** 4)
    delta_lambda = lambda_H - lambda_D
    rel_difference = delta_lambda / lambda_H

    print(f"lambda_H = {lambda_H:.6g}")
    print(f"lambda_D = {lambda_D:.6g}")
    print(f"delta_lambda = {delta_lambda:.6g}")
    print(f"rel_difference = {rel_difference:.6g}")

vorbereitung()
