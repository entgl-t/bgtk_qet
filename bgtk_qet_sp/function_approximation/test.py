import matplotlib.pyplot as plt
import mpmath as mp
from taylor_exp import TaylorSeries
import numpy as np
from sympy import symbols,  asin, exp, tanh
from bgtk_qet_sp.qet_state_prep.utils import get_num_qubits, get_gaussian_params

approx_type = 'taylor'
func_type = 'gauss'
N = np.sqrt(2**get_num_qubits())
beta = get_gaussian_params()
order = 33
center = 0
a = -(1/N)*np.sin(1)
b = (1/N)*np.sin(1)
num_points = 64

z = symbols('x')
if func_type =='gauss':
    function = lambda x: mp.exp((-beta / 2) * (mp.asin(N*x)** 2))
    function_t = exp((-beta/2)*(asin(N*z)**2))  #exp((-beta/2)*(asin(z)**2))
else:
    function= lambda x: mp.tanh(mp.asin(x))
    function_t = tanh(asin(z))  #lambda x: torch.tanh(torch.asin(x))



#function_t = lambda x: torch.exp((-beta/2)*torch.asin(x)**2) if approx_type=='taylor' else lambda x: mp.exp((-0.5)*(mp.asin(x)**2))
#poly_coeffs, max_error = remez(function, order, a, b)

if approx_type == 'taylor':

   ''' coeffs_t= TaylorSeries(function_t, order, center)
    poly_coeffs = coeffs_t.get_coefficients()

    coeffs_t.print_coefficients()
    coeffs_t.print_equation()'''

   coeffs_t = TaylorSeries(function_t, order, z, center)
   poly_coeffs = coeffs_t.get_coefficients()

else:
    #function_t =  lambda x: mp.exp( (-beta/2) * (mp.asin(x) ** 2)) if func_type=='gauss' else lambda x: mp.tanh(mp.asin(x))
    poly_coeffs, max_error = remez(function, order, a, b)

print('Coefficients: ', poly_coeffs)
x = np.linspace(a, b, num_points)


y_approx = np.polyval(poly_coeffs[::-1], x)
y_exact = np.array([function(x_i) for x_i in x])
plt.plot(x, y_exact)
plt.plot(x, y_approx, 'x')
plt.title(r'$f(x)$ v. $P^*_{4}(x)$')
plt.show()