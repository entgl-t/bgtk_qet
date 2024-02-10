import pennylane as qml
import torch
from itertools import product
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from sympy import symbols
import os
cwd = os.path.dirname(os.getcwd())
print('vvvvvvvvvvv', cwd)
def rotation_mat(x):
    """Given a fixed value 'a', compute the signal rotation matrix W(a).
    (requires -1 <= 'a' <= 1)
    """
    diag = x#np.cos(theta)
    off_diag =np.sqrt((1 - x**2))* 1j #np.sqrt((1 - np.cos(theta)**2))* 1j
    W = [[diag, off_diag], [off_diag, diag]]

    return W

def generate_many_sro(x_vals):
    """Given a tensor of possible 'a' vals, return a tensor of W(a)"""
    w_array = []
    for x in x_vals:
        w = rotation_mat(x)
        w_array.append(w)

    return torch.tensor(w_array, dtype=torch.complex64, requires_grad=False)



def QSP_circ(phi, W):
    """This circuit applies the SPO. The components in the matrix
    representation of the final unitary are polynomials!
    """
    qml.Hadamard(wires=0)  # set initial state |+>
    for angle in phi[:-1]:
        qml.RZ(angle, wires=0)
        qml.QubitUnitary(W, wires=0)

    qml.RZ(phi[-1], wires=0)  # final rotation
    qml.Hadamard(wires=0)  # change of basis |+> , |->

    return


def bin_to_num(binary_num, n,neg=True):
    if neg==True:
        if int(binary_num[0]) == 0: #sign bit
            return  int(binary_num[1:],2)
        else:
            return (int(binary_num[1:],2) - 2**(n-1))
    else:
        return int(binary_num,2)

def bin_to_sin(binary_num, n,neg=True):
    num = bin_to_num(binary_num, n,neg=True)
    x_bar = rescaled_x(num, n, neg)
    return np.sin(x_bar)


def get_list_bin_strings(n,neg=True):
    # generate 2^n bit strings
    bin_strs = {''.join(p): bin_to_num(''.join(p),n,neg) for p in product('10', repeat=n)}
    return dict(sorted(bin_strs.items(), key=lambda item: item[1]))


def gaussian_arcsin(x, beta = 1):
        # Gaussian function
        return np.exp((-beta / 2) * (np.arcsin(x)) ** 2)

def func_polyn_approx(coeffs,x):
        return np.polyval(coeffs, x)



def get_gauss_polyn_coeffs(deg,path):
    coeffs = []
    if deg == 32:
        coeffs = torch.load(path +'function_approximation/polynom_approx_coeffs/gauss_degree_32__taylor_center_0.0.pt')
        coeffs = np.array(coeffs)
    return coeffs

def get_tanh_polyn_coeffs(deg,path):
    coeffs = []
    if deg == 33:
        coeffs = torch.load(path +'function_approximation/polynom_approx_coeffs/tanh_degree_33__taylor_center_0.0.pt')
        coeffs = np.array(coeffs)
    return coeffs

def get_sinh_polyn_coeffs(deg,path):
    coeffs = []
    if deg == 33:
        coeffs = torch.load(path + 'function_approximation/polynom_approx_coeffs/sinh_degree_33__taylor_center_0.0.pt')
        coeffs = np.array(coeffs)
    return coeffs

def get_x_polyn_coeffs(deg,path):
    coeffs = []
    if deg == 33:
        coeffs = torch.load(path + 'function_approximation/polynom_approx_coeffs/x_degree_33__taylor_center_0.0.pt')
        coeffs = np.array(coeffs)
    return coeffs
def get_x_2_polyn_coeffs(deg,path):
    coeffs = []
    if deg == 32:
        coeffs = torch.load(path + 'function_approximation/polynom_approx_coeffs/x^2_degree_32__taylor_center_0.0.pt')
        coeffs = np.array(coeffs)
    return coeffs
def get_gaussian_phases(deg,path):
    phases = []
    degree = 0

    if deg ==9:
        phases = torch.load(cwd +'/gaussian_phases/gauss_qsp_angles_deg_9.pt').detach()

        phases = torch.load('gaussian_phases/gauss_qsp_angles_deg_22_error_0.011517447419464588_num_sampl_64.pt').detach()
    elif deg == 32:
        phases = torch.load(path + 'gaussian_phases/gauss_qsp_angles_deg_32_error_4.981482604193843e-08_num_sampl_64.pt').detach()

    else:
        phases = torch.load('gaussian_phases/gauss_qsp_angles_deg_16_threshold_0.05_num_sampl_32.pt').detach()
    degree = len(phases) - 1

    return (degree, phases.numpy())

def get_x_phases(deg,path):
    phases = []
    degree = 0

    if deg==33:
        phases = torch.load( path + 'x_phases/rescaled/x_qsp_angles_deg_33_error_1.8342734620091505e-05_num_sampl_64.pt').detach()
    degree = len(phases) - 1

    return (degree, phases.numpy())

def get_x_2_phases(deg,path):
    phases = []
    degree = 0

    if deg==32:
        phases = torch.load(path + 'x^2_phases/rescaled/x^2_qsp_angles_deg_32_error_9.35878706513904e-05_num_sampl_64.pt').detach()
    degree = len(phases) - 1

    return (degree, phases.numpy())
def get_sinh_phases(deg,path):
    phases = []
    degree = 0

    if deg==33:
        phases = torch.load(path + 'sinh_phases/rescaled/sinh_qsp_angles_deg_33_error_0.00028641422977671027_num_sampl_64.pt').detach()
    degree = len(phases) - 1

    return (degree, phases.numpy())
def get_tanh_phases(deg, path):
    phases = []
    degree = 0
    print(path +'/tanh_phases/rescaled/tanh_qsp_angles_deg_33_error_0.0002866635040845722_num_sampl_64.pt')

    if deg == 23:
        phases = torch.load('tanh_phases/tanh_qsp_angles_deg_23_error_0.00025455025024712086_num_sampl_64.pt').detach()
    elif deg == 33:
        phases = torch.load(path + 'tanh_phases/rescaled/tanh_qsp_angles_deg_33_error_1.4806808394496329e-05_num_sampl_64.pt').detach()

    else:
        pass
    degree = len(phases) - 1

    return (degree, phases.numpy())

def get_gaussian_params():
    beta = 5
    return beta

def get_degree_polyn_approx(ftype):
    if ftype == 'gauss':
        return 32
    elif ftype == 'tanh':
        return 33
    elif ftype == 'sinh':
        return 33
    elif ftype == 'x':
        return 33
    elif ftype == 'x^2':
        return 32
    else:
        return 30



def rescaled_x(x,num_qubits,signed):
   return (2 * x) / (2 ** num_qubits) if signed else x/ (2 ** num_qubits) # 2x/N

def sine_rescaled_x(x,num_qubits,signed):
    if signed:
        t = (2 * x) / (2 ** num_qubits)
    else:
        t = x / (2 ** num_qubits)
    N = np.sqrt(2**num_qubits)

    return (1/N)*np.sin(t)

'''def expected_function(binary_num,num_qubits,signed):
   #Now only for gaussian
   beta = get_gaussian_params()
   num = bin_to_num(binary_num, num_qubits,neg=signed)
   x = rescaled_x(num, num_qubits)
   return mp.exp((-beta * 0.5) * (x ** 2))'''

def func_max_val(func, diap, num_points):
    return np.max([func(s) for s in np.linspace(float(diap[0]),float(diap[1]), num_points)])


def expected_func_polyn(func_type,path):
    beta = get_gaussian_params()
    z = symbols('x')
    degree = get_degree_polyn_approx(func_type)

    #gauss = lambda x: mp.exp((-beta * 0.5) * (x ** 2))
    # gauss = mp.exp((-beta * 0.5) * (mp.asin(1*(1/N)*mp.sin(x)))**2)
    tan_h = lambda x: mp.tanh(x)

    if func_type == 'gauss':
        # Actually we approximate rescaled version of desired function
        poly_coeffs = get_gauss_polyn_coeffs(degree,path)

        gauss = lambda k: np.sum([c*(k**i) for i,c in enumerate(poly_coeffs)])
        '''if rescale == True:
             gauss_rescaled= lambda t: (1/func_max_val(gauss,diap,num_points))* gauss(t)
             return gauss_rescaled'''

        #poly_coeffs = TaylorSeries(function_t, degree, z, center).get_coefficients()

        return gauss

    elif func_type=='tanh':
            poly_coeffs = get_tanh_polyn_coeffs(degree,path)
            tanh = lambda k: np.sum([c * (k ** i) for i, c in enumerate(poly_coeffs)])

            return tanh
    elif func_type == 'sinh':
        poly_coeffs = get_sinh_polyn_coeffs(degree,path)
        sinh = lambda k: np.sum([c * (k ** i) for i, c in enumerate(poly_coeffs)])

        return sinh
    elif func_type == 'x':
        poly_coeffs = get_x_polyn_coeffs(degree,path)
        x = lambda k: np.sum([c * (k ** i) for i, c in enumerate(poly_coeffs)])

        return x
    elif func_type == 'x^2':
        poly_coeffs = get_x_2_polyn_coeffs(degree,path)
        x_2 = lambda k: np.sum([c * (k ** i) for i, c in enumerate(poly_coeffs)])

        return x_2
    else:
        return 0

def expected_function(func_type, binary_num,num_qubits, signed):

    beta = get_gaussian_params()
    num = bin_to_num(binary_num, num_qubits, neg=signed)
    x = rescaled_x(num,num_qubits,signed)
    N = mp.sqrt(2**num_qubits)
    gauss = mp.exp((-beta * 0.5) * (x ** 2))
    #gauss = mp.exp((-beta * 0.5) * (mp.asin(1*(1/N)*mp.sin(x)))**2)
    tanh = mp.tanh(x)
    #tanh = mp.tanh(mp.asin((1/N)*mp.sin(x)))
    sin = mp.sin(x)
    x = mp.asin(x)

    if func_type == 'gauss':
        return gauss
    elif func_type=='tanh':
        return tanh
    elif func_type =='sine':
        return sin
    elif func_type =='x':
        return x
    else:
        return 0
def get_num_qubits():
    return 6

def plot_result(x_vals, output_states, target_function ,exp_states = [], show=True, f_type = 'gauss'):
        title = ''
        if f_type == 'tanh':
            title = r"$\tilde{f} = \tanh(\bar{x})$"
        elif f_type == 'gauss':
            title = r"$\tilde{f} = e^{-\frac{5}{2}\bar{x}^2}$"
        elif f_type == 'sinh':
            title = r"$\tilde{f} = sinh(\bar{x})$"
        elif f_type == 'x':
            title = r"$\tilde{f} = \bar{x}$"
        elif f_type == 'sine':
            title = r"$\tilde{f} = \sin(x)$"
        elif f_type == 'x^2':
            title = r"$\tilde{f} = \bar{x}^2$"
        print('fffffffffffffffffffffffffffffffff',target_function,exp_states )
        """Plot the results"""
        plt.title(title, fontsize='large')
        #plt.rc('axes', axisbelow=True)
        plt.xlabel("Input points "+ r"$\bar{x}$")
        plt.ylabel("Amplitudes "+ r"$\tilde{f}(\bar{x})$")
        plt.plot(x_vals, target_function, ".r", label="Target func "+r"$\tilde{f}(\bar{x})$")
        plt.plot(x_vals, output_states ,".g", label="Output amplitudes")
        plt.grid()
        if len(exp_states) > 0:
             plt.plot(x_vals, exp_states ,".b", label="normalized "+ r"$\tilde{f}(\bar{x})$")
        plt.legend(loc=2)

        if show:
            plt.show()


def custom_poly(coeffs, x):
    """A custom polynomial of degree <= d and parity d % 2"""

    return torch.tensor(np.polyval(coeffs[::-1], x) , requires_grad=False, dtype=torch.float)


def get_trace_distance(ampl1, ampl2):
    ampl1 = np.absolute(ampl1)
    ampl2 = np.absolute(ampl2)
    return np.sqrt(1 - np.sum(np.multiply(ampl1,ampl2))**2)

def get_mse(list1, list2):
    list1 = list(map(abs, list1))
    list2 = list(map(abs, list2))
    return np.square(np.subtract(list1, list2)).mean()
def get_mae(list1, list2):
    list1 = list(map(abs, list1))
    list2 = list(map(abs, list2))
    return np.absolute(np.subtract(list1, list2)).mean()

def validate_phases(x_vals,y_true,phases,qsp_circ,degree, num_samples, f_type):

        model = qsp_circ(degree=degree, num_vals=num_samples)
        model.phi =  torch.nn.Parameter(phases)
        y_pred = model(generate_many_sro(x_vals))

        title = ''
        if f_type == 'tanh':
            title = r"$\tilde{f} = \tanh(\bar{x})$"
        elif f_type == 'gauss':
            title = r"$\tilde{f} = e^{-\frac{5}{2}\bar{x}^2}$"
        elif f_type == 'sinh':
            title = r"$\tilde{f} = sinh(\bar{x})$"
        elif f_type == 'x':
            title = r"$\tilde{f} = \bar{x}$"
        elif f_type == 'sine':
            title = r"$\tilde{f} = \sin(x)$"
        elif f_type == 'x^2':
            title = r"$\tilde{f} = \bar{x}^2$"

        plt.title('Single qubit QSP phases for '+ title, fontsize='large')
        plt.xlabel("Input values " + r"$\bar{x}$")
        plt.ylabel("Amplitudes " + r"$\tilde{f}(\bar{x})$")
        plt.plot(x_vals, list(y_true), "--b", label="target func")
        plt.plot(x_vals, y_pred.tolist(), ".g", label="optim params")
        plt.legend(loc=1)

        plt.show()


        # plt.rc('axes', axisbelow=True)


        plt.grid()
def save_polynom_coeffs(coeffs,path, func_type, approx_method, degree,center):
    torch.save(coeffs, path + func_type +'_degree_'+ str(degree)+ '__'+approx_method+'_center_'+str(center)+'.pt')



