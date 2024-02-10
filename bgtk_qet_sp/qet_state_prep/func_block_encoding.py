
from qiskit import QuantumCircuit, transpile,execute, Aer,QuantumRegister, ClassicalRegister
from sine_block_encoding import SineBlockEncoding
from constants import backend, simulator
from qiskit.circuit.library import MCMT
import os
import qiskit.quantum_info as qi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from utils import  bin_to_num, get_gaussian_phases, bin_to_sin,get_sinh_phases, get_tanh_phases, get_x_phases,get_x_2_phases, rescaled_x
from utils import plot_result,get_trace_distance,get_degree_polyn_approx,expected_func_polyn,func_max_val


class FuncBlockEncodingQSVT:
    def __init__(self,func_type, phases,poly_deg,num_qubits=2, binary_num='00',signed = True,parity = 'even',f_parity='even', plus_states=False):
        self.func_type = func_type
        self.plus_states = plus_states
        self.phases = phases
        self.num_qubits = num_qubits
        self.binary_num = binary_num
        self.signed = signed
        self.parity = parity
        self.f_parity = f_parity
        self.num_ancila = 3 if self.parity not in ['odd','even'] else 2
        self.poly_deg = poly_deg
        self.sine_block_encoding = SineBlockEncoding(self.num_qubits, self.binary_num, self.signed).sine_block_circuit()


    def c_projector(self,phase):
        # circuit = QuantumCircuit(num_ancila, num_ancila, name='Proj')
        circuit = QuantumCircuit(QuantumRegister(self.num_ancila), name='Proj')
        circuit.cx(self.num_ancila - 1, self.num_ancila - 2, ctrl_state=0)
        circuit.rz(phase, self.num_ancila - 2)  # In qiskit angle will be divided by 2 therefore 2*angle as input
        # circuit.crz(phases[d], 0,1,ctrl_state=0)
        circuit.cx(self.num_ancila - 1, self.num_ancila - 2, ctrl_state=0)
        return circuit



    def func_block_encoding(self):

        circuit_sin = self.sine_block_encoding
        circuit_sin = circuit_sin.to_instruction()
        num_ancila = 3 if self.parity == 'undef' else 2
        # circuit_f= QuantumCircuit(num_qubits + num_ancila, num_qubits + num_ancila, name='f_circ')
        circuit_f = QuantumCircuit(QuantumRegister(self.num_qubits + self.num_ancila), name=r'$U_{\tilde{f}}$')

        for i in range(self.num_ancila- 1):
            circuit_f.h(i)

        if self.parity == 'odd':

            circuit_f.compose(self.c_projector(self.phases[self.poly_deg]), qubits=[i for i in range(num_ancila)], inplace=True)

            for d in range(((self.poly_deg -1) // 2),0,-1 ):

                circuit_f.append(circuit_sin,  [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])
                circuit_f.compose(self.c_projector(self.phases[2 * d ]), qubits=[i for i in range(num_ancila)],  inplace=True)

                circuit_f.append(circuit_sin.inverse(), [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])
                circuit_f.compose(self.c_projector(self.phases[2 * d - 1]), qubits=[i for i in range(num_ancila)],inplace=True)



            circuit_f.append(circuit_sin, [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])
            circuit_f.compose(self.c_projector(self.phases[0]), qubits=[i for i in range(num_ancila)], inplace=True)




        elif self.parity == 'even':

            circuit_f.compose(self.c_projector(self.phases[self.poly_deg]), qubits=[i for i in range(num_ancila)], inplace=True)

            for d in range( (self.poly_deg // 2)-1, -1,-1):

                circuit_f.append(circuit_sin, [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])
                circuit_f.compose(self.c_projector(self.phases[2 * d +1]), qubits=[i for i in range(num_ancila)], inplace=True)

                # circuit_sin_sub = circuit_sin if d%2==0 else circuit_sin.inverse()
                circuit_f.append(circuit_sin.inverse(), [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])
                circuit_f.compose(self.c_projector(self.phases[2 * d ]), qubits=[i for i in range(num_ancila)],inplace=True)


        else:
            pass
        for i in range(num_ancila - 1):
            circuit_f.h(i)

        #==========================Delete
        #for i in range(self.num_ancila- 1):
        #    circuit_f.h(i)
        #=================================
        # circuit_f.barrier([i for i in range(num_qubits + num_ancila)])
        return circuit_f

    def func_block_encoding_reverse_ord(self):

        circuit_sin = self.sine_block_encoding
        circuit_sin = circuit_sin.to_instruction()
        num_ancila = 3 if self.parity == 'undef' else 2
        # circuit_f= QuantumCircuit(num_qubits + num_ancila, num_qubits + num_ancila, name='f_circ')
        circuit_f = QuantumCircuit(QuantumRegister(self.num_qubits + self.num_ancila), name=r'$U_{\tilde{f}}$')

        for i in range(self.num_ancila- 1):
            circuit_f.h(i)

        if self.parity == 'odd':

            circuit_f.compose(self.c_projector(self.phases[0]), qubits=[i for i in range(num_ancila)], inplace=True)
            circuit_f.append(circuit_sin, [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])

            for d in range(1,((self.poly_deg -1) // 2 +1 )):

                circuit_f.compose(self.c_projector(self.phases[2 * d - 1 ]), qubits=[i for i in range(num_ancila)],  inplace=True)
                circuit_f.append(circuit_sin,  [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])

                circuit_f.compose(self.c_projector(self.phases[2 * d ]), qubits=[i for i in range(num_ancila)],inplace=True)
                circuit_f.append(circuit_sin.inverse(), [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])



            circuit_f.compose(self.c_projector(self.phases[self.poly_deg]), qubits=[i for i in range(num_ancila)],
                              inplace=True)


        elif self.parity == 'even':


            for d in range(0, (self.poly_deg // 2)):

                circuit_f.compose(self.c_projector(self.phases[2 * d]), qubits=[i for i in range(num_ancila)], inplace=True)
                circuit_f.append(circuit_sin, [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])

                # circuit_sin_sub = circuit_sin if d%2==0 else circuit_sin.inverse()
                circuit_f.compose(self.c_projector(self.phases[2 * d +1 ]), qubits=[i for i in range(num_ancila)],inplace=True)
                circuit_f.append(circuit_sin.inverse(), [i for i in range(num_ancila - 1, self.num_qubits + num_ancila)])

            circuit_f.compose(self.c_projector(self.phases[self.poly_deg]), qubits=[i for i in range(num_ancila)], inplace=True)

        else:
            pass
        for i in range(num_ancila - 1):
            circuit_f.h(i)

        #==========================Delete
        #for i in range(self.num_ancila- 1):
        #    circuit_f.h(i)
        #=================================
        # circuit_f.barrier([i for i in range(num_qubits + num_ancila)])
        return circuit_f

    def func_block_encoding_test(self, plus_state=False):

        circuit_func = self.func_block_encoding()
        #circuit_func = self.func_block_encoding()


        circuit_func = circuit_func.to_instruction()

        circuit_f = QuantumCircuit(QuantumRegister(self.num_qubits + self.num_ancila), ClassicalRegister(self.num_qubits + self.num_ancila))

        # Circuit initialization
        ancila_qubits = ''.join(['0' for i in range(self.num_ancila)])


        if plus_state:
            self.binary_num = ''.join(['0' for i in range(self.num_qubits)])
            circuit_f.initialize(self.binary_num + ancila_qubits, circuit_f.qubits)
            for k in range(self.num_ancila, self.num_qubits + self.num_ancila):
                circuit_f.h(k)
        else:
            circuit_f.initialize(self.binary_num + ancila_qubits, circuit_f.qubits)

        circuit_f.append(circuit_func, [i for i in range(self.num_qubits + self.num_ancila)])

        circuit_f.barrier([i for i in range(self.num_qubits + self.num_ancila)])
        # Map the quantum measurement to the classical bits

        circuit_f.measure([i for i in range(self.num_qubits + self.num_ancila)], [k for k in range(self.num_qubits + self.num_ancila)])

        return circuit_f

    def draw(self, circ_type = 'temp'):

        circuit_f = self.func_block_encoding_test() #if circ_type != 'temp' else self.temp_func_block_encoding_test()

        # Draw the circuit
        circuit_f.decompose().draw("mpl")
        plt.show()

        #print(circuit_f.decompose())
        #return(sine_circuit)


    def get_unitary_from_circuit(self):
        from qiskit import Aer
        backend = Aer.get_backend('unitary_simulator')
        job = execute(self.func_block_encoding_test(plus_state=self.plus_states), backend, shots=8192)
        result = job.result()
        print(result.get_unitary(self.func_block_encoding_test(plus_state=self.plus_states), 3))


    def simulate(self, num_shots, circ_type = 'temp'):

        #compiled_circuit = transpile(self.func_block_encoding_test(plus_state=self.plus_states),simulator)
        if circ_type == 'temp':
            compiled_circuit = transpile(self.temp_func_block_encoding_test(plus_state=self.plus_states),simulator)
        else:
            compiled_circuit = transpile(self.func_block_encoding_test(plus_state=self.plus_states), simulator)

        func_job = simulator.run(compiled_circuit, shots=num_shots)
        func_result = func_job.result()
        func_counts = func_result.get_counts(compiled_circuit)
        print("\nTotal counts are:", func_counts)
        #plot_histogram(func_counts)

        #result = dict(filter(lambda item: '00' == item[0][-2:], func_counts.items()))
        result = dict(filter(lambda item: '00' == item[0][-2:], func_counts.items()))
        print("\nFiltered counts are:", result)
        x_vals = []
        list_out = []
        list_func = []
        c = (1/np.sqrt(2))
        diap = [-np.sin(1),np.sin(1)] if self.signed else [0,np.sin(1)]
        #center = 0.5 if self.func_type == 'tanh' else 0.0
        center = 0.0
        expected_polynom = expected_func_polyn(self.func_type, '')



        #N_tilde_f = np.sqrt(np.sum([expected_polynom(rescaled_x(bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed), self.num_qubits, self.signed))**2 for key in result.keys()]))
        N_tilde_f = np.sqrt(np.sum([expected_polynom(bin_to_sin(key[:-self.num_ancila],self.num_qubits, neg=self.signed))**2 for key in result.keys()]))



        print('N_f: \n',N_tilde_f)
        err_coeff = (N_tilde_f/(np.sqrt(2**self.num_qubits)))

        if self.plus_states == False:
            result = list(result.values())[0] if result != {} else 0

            output_num = np.sqrt(result / num_shots)

            num = bin_to_num(self.binary_num, self.num_qubits, neg=self.signed)
            x_bar = rescaled_x(num, self.num_qubits, self.signed)
            #expected_out= expected_function(self.func_type,self.binary_num, self.num_qubits,self.signed)
            expected_out= expected_polynom(x_bar)

            print('Input binary/decimal: ',str(self.binary_num) + '/' + str(bin_to_num(self.binary_num, self.num_qubits,neg=self.signed)), '\nOutput num: ', output_num,'\nExpected num: ',expected_out,'\nRation exp_out/out: ')
        else:
            for key,val in result.items():
                output_num = np.sqrt(val / num_shots)#np.sqrt(2**self.num_qubits)*np.sqrt(val / num_shots)

                num_test = ''

                #expected_out = expected_function(self.func_type,key[:-self.num_ancila],self.num_qubits,self.signed)
                num = bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)
                x_bar = rescaled_x(num, self.num_qubits, self.signed)
                expected_out = expected_polynom(np.sin(x_bar))/np.abs(func_max_val(expected_polynom,diap,2**self.num_qubits))


                output_num = (-1)*output_num if self.f_parity == 'odd' and x_bar < 0 else output_num

                x_vals.append(x_bar)
                #x_vals.append(sine_rescaled_x(bin_to_num(key[:-self.num_ancila], self.num_qubits,neg=self.signed),self.num_qubits, self.signed))
                list_out.append(output_num)
                list_func.append(expected_out)

                '''print('\n\nInput binary/decimal: ',
                     str(key[:-self.num_ancila]) + '/' + str(bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)),
                     '\nOutput num: ', output_num, '\nExpected num: ', (1/Nf)*expected_out, '\nRatio exp_out/out: ',
                      ((1/Nf)*expected_out) / output_num)'''
                print('\n\nInput binary/decimal: ',
                     str(key[:-self.num_ancila]) + '/' + str(bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)),
                     '\nOutput num: ', output_num, '\nExpected num: ', err_coeff*(1/N_tilde_f)*expected_out, '\nRatio: ',(err_coeff*(1/N_tilde_f)*expected_out)/output_num,'Reverse num',
                      num_test)
                #Nf = Nf + np.abs(expected_out)**2

            #Nf = np.sqrt(Nf)
            exp_state= [i * err_coeff*(1/N_tilde_f)  for i in list_func] #[0 for i in list_func]
            print('\nTrace distance between expected state and outputs state: ', get_trace_distance(exp_state, list_out))
            #r"$\exp^{-\frac{5}{2}*x^2}$"
            plot_result(x_vals,list_out,list_func, exp_state,f_type=self.func_type)

    def unitary_simulation(self):
        #circuit_func = self.func_block_encoding()
        circuit_func = self.func_block_encoding_reverse_ord()


        circuit_func = circuit_func.to_instruction()

        circuit_f = QuantumCircuit(QuantumRegister(self.num_qubits + self.num_ancila))
        ancila_qubits = ''.join(['0' for i in range(self.num_ancila)])


        ancila_qubit = '0'
        if self.plus_states:
            self.binary_num = ''.join(['0' for i in range(self.num_qubits)])

            for k in range(self.num_ancila, self.num_qubits + self.num_ancila):
                circuit_f.h(k)

        self.binary_num = self.binary_num + ancila_qubits
        circuit_f.append(circuit_func, [i for i in range(self.num_qubits + self.num_ancila)])

        op = qi.Operator(circuit_f)
        sv = qi.Statevector.from_label(self.binary_num)
        sv = sv.evolve(op)
        print('gggggggggggggggg',sv.to_dict())

        result = dict(filter(lambda item: '00' == item[0][-2:], sv.to_dict().items()))
        print("\nFiltered counts are:", result)
        x_vals = []
        list_out = []
        list_func = []
        c = (1 / np.sqrt(2))
        diap = [-np.sin(1), np.sin(1)] if self.signed else [0, np.sin(1)]
        # center = 0.5 if self.func_type == 'tanh' else 0.0
        center = 0.0
        expected_polynom = expected_func_polyn(self.func_type, '')

        # N_tilde_f = np.sqrt(np.sum([expected_polynom(rescaled_x(bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed), self.num_qubits, self.signed))**2 for key in result.keys()]))
        N_tilde_f = np.sqrt(np.sum(
            [expected_polynom(bin_to_sin(key[:-self.num_ancila], self.num_qubits, neg=self.signed)) ** 2 for key in
             result.keys()]))

        print('N_f: \n', N_tilde_f)
        err_coeff = (N_tilde_f / (np.sqrt(2 ** self.num_qubits)))

        if self.plus_states == False:
            result = list(result.values())[0] if result != {} else 0

            output_num = result

            num = bin_to_num(self.binary_num, self.num_qubits, neg=self.signed)
            x_bar = rescaled_x(num, self.num_qubits, self.signed)
            # expected_out= expected_function(self.func_type,self.binary_num, self.num_qubits,self.signed)
            expected_out = expected_polynom(x_bar)

            print('Input binary/decimal: ',
                  str(self.binary_num) + '/' + str(bin_to_num(self.binary_num, self.num_qubits, neg=self.signed)),
                  '\nOutput num: ', output_num, '\nExpected num: ', expected_out, '\nRation exp_out/out: ')
        else:
            for key, val in result.items():
                output_num = val  # np.sqrt(2**self.num_qubits)*np.sqrt(val / num_shots)

                num_test = ''

                # expected_out = expected_function(self.func_type,key[:-self.num_ancila],self.num_qubits,self.signed)
                num = bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)
                x_bar = rescaled_x(num, self.num_qubits, self.signed)
                expected_out = expected_polynom(np.sin(x_bar)) / np.abs(
                    func_max_val(expected_polynom, diap, 2 ** self.num_qubits))

                #output_num = (-1) * output_num if self.f_parity == 'odd' and x_bar < 0 else output_num

                x_vals.append(x_bar)
                # x_vals.append(sine_rescaled_x(bin_to_num(key[:-self.num_ancila], self.num_qubits,neg=self.signed),self.num_qubits, self.signed))
                list_out.append(output_num)
                list_func.append(expected_out)

                print('\n\nInput binary/decimal: ',
                      str(key[:-self.num_ancila]) + '/' + str(
                          bin_to_num(key[:-self.num_ancila], self.num_qubits, neg=self.signed)),
                      '\nOutput num: ', output_num, '\nExpected num: ', err_coeff * (1 / N_tilde_f) * expected_out,
                      '\nRatio: ', (err_coeff * (1 / N_tilde_f) * expected_out) / output_num, 'Reverse num',
                      num_test)
                # Nf = Nf + np.abs(expected_out)**2

            # Nf = np.sqrt(Nf)
            exp_state = [i * err_coeff * (1 / N_tilde_f) for i in list_func]  # [0 for i in list_func]
            print('\nTrace distance between expected state and outputs state: ',
                  get_trace_distance(exp_state, list_out))
            # r"$\exp^{-\frac{5}{2}*x^2}$"
            plot_result(x_vals, list_out, list_func, exp_state, f_type=self.func_type)

if __name__ == '__main__':

    #TODO: dete temp_func_block_encoding_test
    #Improvements: added phase phi_0 to circuit
    #Successful setting:1)circ_type:'temp',pol: odd, degree 9, phi_0
    #                   2)func:tanh, circ_type:'reg',pol: odd, degree 15/25, without phi_0, withour init H
    #beta = 1

    #poly_deg = 20
    func = 'gauss' #variants: tanh or gauss
    degree = get_degree_polyn_approx(func)

    # Approximate even function with even d, odd function with odd d
    if func == 'tanh':
        poly_deg, phases = get_tanh_phases(degree,'')

    elif func == 'gauss':
        poly_deg, phases = get_gaussian_phases(degree,'')
    elif func == 'sinh':
        poly_deg, phases = get_sinh_phases(degree,'')
    elif func == 'x':
        poly_deg, phases = get_x_phases(degree, '')
    elif func == 'x^2':
        poly_deg, phases = get_x_2_phases(degree, '')
    else:
        poly_deg, phases =(0,[])



    #phases = 2*phases #becaouse we double it in RZ
    #phases = phases[::-1] #acorodint to phases order in QSP
    print('phases2:', phases)

    parity = 'even' if poly_deg%2 == 0 else 'odd'
    f_parity = 'even' if func in['gauss','x^2'] else 'odd'
    signed = True #if func in ['gauss'] else False

    circ_type = 'reg' #variants: temp or reg

    print('Phases', phases, poly_deg)

    num_qubits = 6 #get_num_qubits()
    binary_num = ''.join(['0' for i in range(num_qubits)])
    test = FuncBlockEncodingQSVT(func_type =func,phases=phases
                                 ,poly_deg=poly_deg,num_qubits=num_qubits
                                 , binary_num=binary_num,signed = signed
                                 ,parity = parity
                                 ,f_parity = f_parity
                                 , plus_states = True)
    #test.draw(circ_type = circ_type)
    #test.get_unitary_from_circuit()
    #test.simulate(100000, circ_type)
    test.unitary_simulation()
    test.draw('reg')