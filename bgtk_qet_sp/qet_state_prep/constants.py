from qiskit import QuantumCircuit, transpile,execute, Aer,QuantumRegister
from qiskit_aer import AerSimulator
# Use Aer's AerSimulator
simulator = AerSimulator()
backend = Aer.get_backend('statevector_simulator')
#backend = AerSimulator(method ='statevector')
num_qubits = 2  # number of qubits without ancilla qubits
#sign_bit = '0'      #sign of number: 0 if positive, 1 if negative
#binary_num = sign_bit + '00'  # number (7-2^(4-1)) in binary form, corresponds to initial state |1111>
binary_num = '00'
num_shots = 1000    # number of circuit starts
validation_threshold = 0.05 # threshold for checking sin(x) circuit
d = 9              #Degree of the polynomial approximation
#beta = 1
f_circ_ancila = 1