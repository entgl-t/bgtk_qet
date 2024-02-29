from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from func_block_encoding import FuncBlockEncodingQSVT
from constants import simulator
from qiskit.circuit.library import MCMT
import qiskit.quantum_info as qi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from bgtk_qet_sp.utils import  bin_to_num, get_gaussian_phases, get_trace_distance,get_sinh_phases\
    ,get_num_qubits, rescaled_x, get_x_phases, plot_result, get_tanh_phases\
    ,expected_func_polyn,func_max_val,get_degree_polyn_approx,bin_to_sin,get_x_2_phases


class QSVTAmplitudeAmplification:
    def __init__(self, func_type, phases,poly_deg,num_qubits=2, binary_num='00',signed=True,parity = 'even',f_parity='even', plus_states=False):
        self.func_type = func_type
        self.plus_states = True
        self.phases = phases
        self.num_qubits = num_qubits
        self.binary_num = binary_num
        self.signed = signed
        self.parity = parity
        self.f_parity = f_parity
        self.num_ancila = 4 if self.parity not in ['odd','even'] else 3
        self.poly_deg = poly_deg
        self.diap = [-np.sin(1), np.sin(1)] if self.func_type in ['gauss'] else [0, np.sin(1)]
        self.n_rounds = self.get_n_rounds()
        self.phi = np.arccos(np.sin(self.get_theta())/self.get_success_ampl())/2
        self.func_block_encoding = FuncBlockEncodingQSVT(self.func_type,self.phases, self.poly_deg, self.num_qubits, self.binary_num, self.signed,self.parity,self.plus_states).func_block_encoding()

    def get_Nf_tilde(self,polyn, x_vals):
        return  np.sqrt(np.sum([polyn(x) ** 2 for x in x_vals]))

    def get_success_ampl(self):
        expected_polynom = expected_func_polyn(self.func_type,cwd + '/function_approximation/')
        x_vals = np.linspace(self.diap[0],self.diap[1], 2**self.num_qubits)
        fill_frac = self.get_Nf_tilde(expected_polynom,x_vals)/(np.sqrt(2**self.num_qubits)* func_max_val(expected_polynom,self.diap,2**self.num_qubits))
        return  fill_frac

    def get_n_rounds(self):
        fill_frac = self.get_success_ampl()
        temp = np.pi/(4*np.arcsin(fill_frac)) - 0.5
        n_rounds = np.ceil(temp)
        return  int(n_rounds)

    def get_theta(self):
        return np.pi/(4*self.n_rounds + 2)

    # Anti-controlled Z-gate
    def anti_controlled_z(self,r_start, r_end):
        acz = QuantumCircuit(r_end - r_start + 1, r_end - r_start + 1, name='anti_cz')
        ccz = MCMT('cz', r_end - r_start, 1)
        for i in range(r_start, r_end + 1):
            acz.x(i)
        acz.compose(ccz, qubits=[i for i in range(r_start, r_end + 1)], inplace=True)

        for i in range(r_start, r_end + 1):
            acz.x(i)
        return acz


    def f_ampl_amplification_test(self,initialize = True, measure = True):
        circuit_f = self.func_block_encoding
        circuit_f = circuit_f.to_instruction()
        num_ancila = 4 if self.parity == 'undef' else 3
        circuit_ampl = QuantumCircuit(QuantumRegister(self.num_qubits + self.num_ancila),
                                   ClassicalRegister(self.num_qubits + self.num_ancila))
        # Circuit initialization
        ancila_qubits = ''.join(['0' for i in range(num_ancila)])
        if initialize:
             circuit_ampl.initialize(self.binary_num + ancila_qubits, circuit_ampl.qubits)

        for i in range(self.num_qubits + self.num_ancila - 1, self.num_ancila - 1, -1):
            circuit_ampl.h(i)
        #circuit_ampl.barrier([i for i in range(self.num_qubits + self.num_ancila)])
        circuit_ampl.ry(2 * self.phi, 0)
        circuit_ampl.append(circuit_f, [i for i in range(1, self.num_qubits + num_ancila)])

        # theta = np.arcsin(1/np.sqrt(2**(num_qubits/2)))#2*np.arccos(1/np.sqrt(2**num_qubits))
        for r in range(self.n_rounds):
            circuit_ampl.compose(self.anti_controlled_z(0, num_ancila - 1), qubits=[i for i in range(num_ancila)], inplace=True)
            circuit_ampl.ry(-2 * self.phi, 0)
            circuit_ampl.append(circuit_f.inverse(), [i for i in range(1, self.num_qubits + num_ancila)])
            for i in range(self.num_qubits + num_ancila - 1, num_ancila - 1, -1):
                circuit_ampl.h(i)

            #circuit_ampl.barrier([i for i in range(self.num_qubits + num_ancila)])
            circuit_ampl.compose(self.anti_controlled_z(0, num_ancila + self.num_qubits - 1),
                                 qubits=[i for i in range(num_ancila + self.num_qubits)], inplace=True)
            for i in range(self.num_qubits + num_ancila - 1, num_ancila - 1, -1):
                circuit_ampl.h(i)
            #circuit_ampl.barrier([i for i in range(self.num_qubits + num_ancila)])
            circuit_ampl.ry(2 * self.phi, 0)
            circuit_ampl.append(circuit_f, [i for i in range(1, self.num_qubits + num_ancila)])
        # Map the quantum measurement to the classical bits
        if measure:
            circuit_ampl.measure([i for i in range(self.num_qubits + num_ancila)], [k for k in range(self.num_qubits + num_ancila)])
        return circuit_ampl

    def plot_result(self, x_vals, y_pred, y_true, show=True):
        """Plot the results"""
        plt.title("Amplitude amplification")
        plt.plot(x_vals, y_true, ".r", label="target func")
        plt.plot(x_vals, y_pred, ".g", label="qsvt approx")
        plt.legend(loc=1)
        if show:
            plt.show()

    def draw(self):
        aampl_circ = self.f_ampl_amplification_test()
        # Draw the circuit
        aampl_circ.draw("mpl")
        plt.show()
        #print(aampl_circ.decompose())
        #return(sine_circuit)

    def simulate(self, num_shots):
        print('Number of rounds:', self.n_rounds)
        print('Phi:', self.phi)
        compiled_circuit = transpile(self.f_ampl_amplification_test(initialize=True,measure=True),simulator)
        aampl_job = simulator.run(compiled_circuit, shots=num_shots)
        aampl_result = aampl_job.result()
        aampl_counts = aampl_result.get_counts(compiled_circuit)
        print("\nTotal counts are:", aampl_counts)
        result = dict(filter(lambda item: '000' == item[0][-3:], aampl_counts.items()))
        print("\nFiltered counts are:", result)

        x_vals = []
        list_out = []
        list_func = []
        expected_polynom = expected_func_polyn(self.func_type,cwd + '/function_approximation/')
        N_tilde_f = self.get_Nf_tilde(expected_polynom,[bin_to_sin(key[:-self.num_ancila],self.num_qubits, neg=self.signed) for key in result.keys()])

        for key,val in result.items():
            output_num = np.sqrt(val / num_shots)
            num = bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)
            x_bar = rescaled_x(num, self.num_qubits, self.signed)
            expected_out = expected_polynom(np.sin(x_bar))
            output_num = (-1) * output_num if self.f_parity == 'odd' and x_bar < 0 else output_num
            x_vals.append(x_bar)
            list_out.append(output_num)
            list_func.append(expected_out)
            print('\n\nInput binary/decimal: ',
                  str(key[:-self.num_ancila]) + '/' + str(bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)),
                  '\nOutput num: ', output_num, '\nExpected num: ', (1 / N_tilde_f) * expected_out, '\nRatio exp_out/out: ',
                  ((1 / N_tilde_f) * expected_out) / output_num)
        exp_state = [i * (1 / N_tilde_f) for i in list_func]
        print('\nTrace distance between expected state and outputs state: ', get_trace_distance(exp_state, list_out))
        plot_result(x_vals, list_out, list_func, exp_state)

    def unitary_simulation(self):
        print('Number of rounds:', self.n_rounds)
        print('Phi:', self.phi)
        circuit_aa = self.f_ampl_amplification_test(initialize=False, measure=False)
        num_ancila = 4 if self.parity == 'undef' else 3
        ancila_qubits = ''.join(['0' for i in range(num_ancila)])
        op = qi.Operator(circuit_aa)
        sv = qi.Statevector.from_label(self.binary_num + ancila_qubits)
        sv = sv.evolve(op)
        print('Statevector: ',sv.to_dict())
        result = dict(filter(lambda item: '000' == item[0][-3:], sv.to_dict().items()))
        print("\nFiltered counts are:", result)
        x_vals = []
        list_out = []
        list_func = []
        expected_polynom = expected_func_polyn(self.func_type, cwd + '/function_approximation/')
        N_tilde_f = self.get_Nf_tilde(expected_polynom,
                                      [bin_to_sin(key[:-self.num_ancila], self.num_qubits, neg=self.signed) for key in
                                       result.keys()])
        for key, val in result.items():
            output_num = np.real(val)
            num = bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)
            x_bar = rescaled_x(num, self.num_qubits, self.signed)
            expected_out = expected_polynom(np.sin(x_bar))
            x_vals.append(x_bar)
            list_out.append(output_num)
            list_func.append(expected_out)
            print('\n\nInput binary/decimal: ',
                  str(key[:-self.num_ancila]) + '/' + str(
                      bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)),
                  '\nOutput num: ', output_num, '\nExpected num: ', (1 / N_tilde_f) * expected_out,
                  '\nRatio exp_out/out: ',
                  ((1 / N_tilde_f) * expected_out) / output_num)
        exp_state = [i * (1 / N_tilde_f) for i in list_func]
        print('\nTrace distance between expected state and outputs state: ', get_trace_distance(exp_state, list_out))
        print('Output amplitudes :', list_out)
        plot_result(x_vals, list_out, list_func, exp_state,f_type=self.func_type)

if __name__ == '__main__':
    num_qubits = get_num_qubits()
    binary_num = ''.join(['0' for i in range(num_qubits)])
    # poly_deg = 20
    func = 'gauss'  # variants: tanh or gauss
    degree = get_degree_polyn_approx(func)

    cwd = os.path.dirname(os.getcwd())
    # Approximate even function with even d, odd function with odd d
    if func == 'gauss':
        poly_deg, phases = get_gaussian_phases(degree, cwd + '/examples/')
    else:
        poly_deg, phases = (0, [])

    #phases = 2*phases
    phases = phases[::-1]
    parity = 'even' if poly_deg % 2 == 0 else 'odd'
    f_parity = 'even' if func in ['gauss','x^2'] else 'odd'
    signed = True #if func == 'gauss' else False
    circ_type = 'reg'  # variants: temp or reg
    print('Phases', phases, poly_deg)
    test = QSVTAmplitudeAmplification(func_type = func,phases=phases,poly_deg=poly_deg,num_qubits=num_qubits
                                      , binary_num=binary_num,signed = signed,parity = parity,f_parity = f_parity
                                     )
    #test.simulate(100000)
    test.unitary_simulation()
    test.draw()
