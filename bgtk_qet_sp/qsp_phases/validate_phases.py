import numpy as np
import torch
from qsp_spo import QSP_Func_Fit
from bgtk_qet_sp.qet_state_prep.utils import  get_degree_polyn_approx, validate_phases , expected_func_polyn

a = -np.sin(1)
b=np.sin(1)
num_samples = 64


func = 'x^2' #variants: tanh or gauss
degree = get_degree_polyn_approx(func)
phases = []
# Approximate even function with even d, odd function with odd d
if func == 'tanh':
    if degree == 33:
        phases = torch.load('../tanh_phases/rescaled/tanh_qsp_angles_deg_33_error_0.000112739740870893_num_sampl_64.pt')
elif func == 'gauss':
    if degree == 32:
        phases = torch.load(
            '../examples/gaussian_phases/rescaled/gauss_qsp_angles_deg_32_error_7.896923079897533e-07_num_sampl_64.pt')
elif func == 'sinh':
    if degree == 33:
        phases = torch.load('../sinh_phases/rescaled/sinh_qsp_angles_deg_33_error_0.00028641422977671027_num_sampl_64.pt')
elif func == 'x':
    if degree == 33:
        phases = torch.load('../x_phases/rescaled/x_qsp_angles_deg_33_error_1.8342734620091505e-05_num_sampl_64.pt')
elif func == 'x^2':
    if degree == 32:
        phases = torch.load('../x^2_phases/rescaled/x^2_qsp_angles_deg_32_error_9.35878706513904e-05_num_sampl_64.pt')
else:
    poly_deg, phases =(0,[])



#poly_coeffs, _ = remez(function, degree, a, b)
#poly_coeffs = TaylorSeries(function, degree, z, center).get_coefficients()


#theta_vals = np.linspace(0, np.pi, num_samples)
a_vals =  np.linspace(a, b, num_samples)
polynom = expected_func_polyn(func,'../') #custom_poly(poly_coeffs,a_vals)
y_true = map(polynom, a_vals)


#a_vals = np.linspace(a, b, num_samples)
#y_true = custom_poly(poly_coeffs,a_vals)


validate_phases(a_vals,y_true,phases, QSP_Func_Fit,degree,num_samples,f_type = func )