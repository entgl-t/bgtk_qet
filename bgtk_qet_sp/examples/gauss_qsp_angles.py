import numpy as np
import torch
from bgtk_qet_sp.qsp_phases.qsp_angles_trainer import QSP_Model_Trainer
from bgtk_qet_sp.qsp_phases.qsp_spo import QSP_Circ_Fit
from bgtk_qet_sp.utils import  generate_many_sro,get_degree_polyn_approx,save_polynom_coeffs
from bgtk_qet_sp.function_approximation.taylor_exp import  TaylorSeries
from bgtk_qet_sp.utils import get_gaussian_params,get_num_qubits,func_max_val
from sympy import symbols,  asin, exp

num_qubits = get_num_qubits()
N = np.sqrt(2**num_qubits)
z = symbols('x')

# Approximate even function with even d, odd function with odd d
a = -np.sin(1)
b=np.sin(1)

center = 0.0

num_samples =2**num_qubits
beta = get_gaussian_params()
f_type = 'gauss'
degree = get_degree_polyn_approx(f_type)# dim(phi) = d + 1,

optimizer = 'sgd' # types: 1)sgd 2)adam
learning_rate = 1e-2
b1 = 0.5
b2 = 0.999
loss_reduction = 'sum'
optimizer_params = (optimizer,learning_rate,b1,b2,loss_reduction)

a_vals =  np.linspace(a, b, num_samples)

save_pol_path = '../function_approximation/'

function_t = exp((-beta/2)*((asin(z))**2))

poly_coeffs = TaylorSeries(function_t, degree, z, center).get_coefficients()
print('Taylor series, polynomial coeffs', function_t, poly_coeffs)
save_polynom_coeffs(poly_coeffs,save_pol_path, f_type,'taylor',degree,center)


def custom_poly(coeffs, x):
    """A custom polynomial of degree <= d and parity d % 2"""
    return torch.tensor(np.sum([c * (x ** i) for i, c in enumerate(coeffs)]), dtype=torch.float)

poly_func =lambda x: np.sum([c * (x ** i) for i, c in enumerate(poly_coeffs)])

#We want to get phases for rescaled polynomial approximation f(x)/|f|_max
y_true = custom_poly(poly_coeffs,a_vals)/np.abs(func_max_val(poly_func,[a,b],num_samples))
threshold = 1e-12
multi_proc = False

qsp_model_runner = QSP_Model_Trainer(QSP_Circ_Fit, degree, num_samples, a_vals, generate_many_sro, y_true, threshold,f_type = f_type,optim_params=optimizer_params)

num_iter = 500000
qsp_model_runner.execute(num_iter = num_iter)
#qsp_model_runner.execute_bfgs(num_iter = num_iter, lr = learning_rate,history_size=10, max_iter = 4)
qsp_model_runner.plot_result()


'''
if __name__ = '__main__':
    parser = argparse.ArgumentParser(description='Training of QSP phases')
    parser.add_argument('-func', help='choose functions: gauss or tanh', choices=['gauss', 'tanh'], default=['gauss'])
    parser.add_argument('-degrees', help='degrees of approx. polynomials',  default=[15, 25, 16, 20])
    parser.add_argument('-num_sampl', help='num of samples for remez alg',  default=32)
    parser.add_argument('-approx_err', help='approximation threshold',  default=0.05)
    parser.add_argument('-num_iter', help='number iterations',  default=10000)
    parser.add_argument('-lr', help='learinig rate for gradient descent',  default=1e-3)
    args = parser.parse_args()

    calc_qsp_phases(args)'''

