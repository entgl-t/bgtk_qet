import matplotlib.pyplot as plt
import mpmath as mp
from bgtk_qet_sp.function_approximation.taylor_exp import TaylorSeries
import numpy as np
from sympy import symbols,  asin, exp

beta = 5
order = 33
center = 0
a = -np.sin(1)
b =  np.sin(1)
num_points = 64

z = symbols('x')

function = lambda x: mp.exp((-beta / 2) * (mp.asin(x)** 2))
function_t = exp((-beta/2)*(asin(z)**2))  #exp((-beta/2)*(asin(z)**2))

coeffs_t = TaylorSeries(function_t, order, z, center)
poly_coeffs = coeffs_t.get_coefficients()


print('Coefficients: ', poly_coeffs)
x = np.linspace(a, b, num_points)


y_approx = np.polyval(poly_coeffs[::-1], x)
y_exact = np.array([function(x_i) for x_i in x])

plt.xlabel( r"$\bar{x}$")
plt.ylabel(r"$f(x)$")
plt.plot(x, y_exact, "xr", label="Target func "+r'$exp(-\frac{\beta}{2}x^2)$ ')
plt.plot(x, y_approx,"g", label="Polyn. approx")
plt.title(r'$exp(-\frac{\beta}{2}x^2)$ ')
plt.show()

