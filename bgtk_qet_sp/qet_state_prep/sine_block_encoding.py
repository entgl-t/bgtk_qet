from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from constants import simulator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from bgtk_qet_sp.utils import bin_to_num,expected_function, rescaled_x, get_mae, plot_result, get_mse
import matplotlib.pyplot as plt
import qiskit.quantum_info as qi


class SineBlockEncoding:
    '''
    Signed numbers: |X_nX_(n-1)...X_0 >, first bit is sign bit
    '''
    def __init__(self, num_qubits=2, binary_num='00',signed = True, plus_state=True):
        self.num_qubits = num_qubits
        self.binary_num =binary_num #reverse order:from right to left
        self.plus_state = plus_state
        self.signed = signed
        self.func_type = 'sine'

    def sine_block_circuit(self):
        # circuit_sin = QuantumCircuit(num_qubits+1, num_qubits+1, name='sine_circ')
        circuit_sin = QuantumCircuit(QuantumRegister(self.num_qubits + 1), name=r'$U_{sin}$')
        # Add a control RY gate with angles 2**(l-num_qubits) on control qubit l and target qubit 0
        for l in range(1, self.num_qubits + 1):
            if l == self.num_qubits:
                if self.signed:
                    circuit_sin.cry(2 * (-(2**0)), l, 0)
                else:
                    circuit_sin.cry(2 * (2 ** (l - self.num_qubits)), l, 0)
            else:
                circuit_sin.cry(2 * (2 ** (l - self.num_qubits)), l, 0)

        circuit_sin.x(0)
        return circuit_sin

    def sine_block_encoding_test(self):
        circuit_sin = self.sine_block_circuit()

        circuit_sin = circuit_sin.to_instruction()


        circuit_sin_enc = QuantumCircuit(QuantumRegister(self.num_qubits + 1), ClassicalRegister(self.num_qubits + 1) )
        ancila_qubit = '0'

        if self.plus_state:
            self.binary_num = ''.join(['0' for i in range(self.num_qubits)])
            circuit_sin_enc.initialize(self.binary_num + ancila_qubit, circuit_sin_enc.qubits)
            for k in range(1, self.num_qubits + 1):
                circuit_sin_enc.h(k)
        else:
            # Circuit initialization
            circuit_sin_enc.initialize(self.binary_num + ancila_qubit, circuit_sin_enc.qubits)

        circuit_sin_enc.append(circuit_sin, [i for i in range(self.num_qubits + 1)])
        circuit_sin_enc.barrier([i for i in range(self.num_qubits + 1)])

        # Map the quantum measurement to the classical bits
        circuit_sin_enc.measure([i for i in range(self.num_qubits+1)], [k for k in range(self.num_qubits+1)])
        return circuit_sin_enc

    def draw(self):
        sine_circuit = self.sine_block_encoding_test()
        # Draw the circuit
        sine_circuit.decompose().draw(output = "mpl")
        plt.show()

    def simulate2(self, num_shots):

        compiled_circuit = transpile(self.sine_block_encoding_test(),simulator)

        sine_job = simulator.run(compiled_circuit, shots=num_shots)
        sine_result = sine_job.result()
        sine_counts = sine_result.get_counts(compiled_circuit)
        print("\nTotal counts are:", sine_counts)
        #plot_histogram(sine_counts)
        result = dict(filter(lambda item: '0' == item[0][-1], sine_counts.items()))
        print("\nFiltered counts are:", result)
        x_vals = []
        list_out = []
        list_func = []
        N_sin = np.sqrt(2**self.num_qubits)
        if self.plus_state == False:

            result = list(result.values())[0] if result != {} else 0

            output_num = np.sqrt(result / num_shots)

            expected_out= expected_function(self.func_type,self.binary_num, self.num_qubits,self.signed)  # sin(2x/N)

            print('Input binary/decimal: ',str(self.binary_num) + '/' + str(bin_to_num(self.binary_num, self.num_qubits,neg=self.signed)), '\nOutput num: ', output_num,'\nExpected num: ',expected_out,'\nMSE: ',get_mse([expected_out],[output_num]))
        else:
            for key,val in result.items():
                output_num =np.sqrt(val / num_shots)
                #absolute value, because circuit output is positive
                expected_out = (1/N_sin)*expected_function(self.func_type, key[:-1], self.num_qubits, self.signed)

                num = bin_to_num(key[:-1], self.num_qubits,neg=self.signed)
                x_bar = rescaled_x(num, self.num_qubits, self.signed)

                x_vals.append(x_bar)
                output_num = (-1) * output_num if x_bar < 0 else output_num

                list_out.append(output_num)
                list_func.append(expected_out)
                print('\n\nInput binary/decimal: ',
                     str(key[:-1]) + '/' + str(bin_to_num(key[:-1], self.num_qubits, neg=self.signed)),
                     '\nOutput num: ', output_num, '\nExpected num: ', expected_out)
            print('\nMAE: ', get_mae(list_func, list_out))
            plot_result(x_vals,list_out,list_func,f_type = 'sine')

    def unitary_simulation(self):
        circuit_sin = self.sine_block_circuit()
        circuit_sin = circuit_sin.to_instruction()
        circuit_sin_enc = QuantumCircuit(QuantumRegister(self.num_qubits + 1))

        ancila_qubit = '0'
        if self.plus_state:
            for k in range(1, self.num_qubits + 1):
                circuit_sin_enc.h(k)
            self.binary_num = ''.join(['0' for i in range(self.num_qubits)])

        circuit_sin_enc.append(circuit_sin, [i for i in range(self.num_qubits + 1)])

        self.binary_num = self.binary_num + ancila_qubit

        op = qi.Operator(circuit_sin_enc)
        sv = qi.Statevector.from_label(self.binary_num)
        sv = sv.evolve(op)
        result = dict(filter(lambda item: '0' == item[0][-1], sv.to_dict().items()))
        print("\nFiltered counts are:", result)
        x_vals = []
        list_out = []
        list_func = []
        N_sin = np.sqrt(2 ** self.num_qubits)
        if self.plus_state == False:
            output_num = list(result.values())[0] if result != {} else 0
            #output_num = np.sqrt(result / num_shots)
            expected_out = expected_function(self.func_type, self.binary_num, self.num_qubits, self.signed)  # sin(2x/N)
            print('Input binary/decimal: ',
                  str(self.binary_num) + '/' + str(bin_to_num(self.binary_num, self.num_qubits, neg=self.signed)),
                  '\nOutput num: ', output_num, '\nExpected num: ', expected_out, '\nMSE: ',
                  get_mse([expected_out], [output_num]))
        else:
            for key, val in result.items():
                output_num = val
                # absolute value, because circuit output is positive
                expected_out = (1 / N_sin) * expected_function(self.func_type, key[:-1], self.num_qubits, self.signed)
                num = bin_to_num(key[:-1], self.num_qubits, neg=self.signed)
                x_bar = rescaled_x(num, self.num_qubits, self.signed)
                x_vals.append(x_bar)
                #output_num = (-1) * output_num if x_bar < 0 else output_num
                list_out.append(output_num)
                list_func.append(expected_out)
                print('\n\nInput binary/decimal: ',
                      str(key[:-1]) + '/' + str(bin_to_num(key[:-1], self.num_qubits, neg=self.signed)),
                      '\nOutput num: ', output_num, '\nExpected num: ', expected_out)
            print('\nMAE: ', get_mae(list_func, list_out))

            plot_result(x_vals, list_out, list_func, f_type='sine')

if __name__ == '__main__':
    num_qubits = 6
    binary_num = '0000000000'
    plus_state = True
    test = SineBlockEncoding(num_qubits,binary_num,signed=True, plus_state = plus_state)
    test.simulate2(100000)
    #test.unitary_simulation()
    test.draw()