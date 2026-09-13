import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import Aer 
from qiskit import *
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator

from qiskit.visualization import plot_histogram
import numpy as np
import math
from poisson2 import *



simulator = Aer.get_backend('aer_simulator')

imax = 15
# f(i) = i / imax

# c = 1/2
c = 1/4
def operatorF():
    base = QuantumRegister(4, 'base')
    rotation = QuantumRegister(1, 'rotation')
    measurement = ClassicalRegister(1, 'measurement')
    circ = QuantumCircuit(base, rotation, measurement)
    circ.ry(np.pi/2 - 2*c,4)
    for i in range(4):
        circ.cry(2**(i+2)/imax*c,i,4)
    #circ.draw('mpl',fold=30, filename="operatorF.png")
    return circ

def operatorA():
    qc = poissonCircuit()  # 5 base, 6 ancilla, 4 measurement
    f = operatorF()        # 4 base, 1 rotation
    circuit = qc.compose(f, qubits=[1,2,3,4,5])
    circuit.measure([5],[0])
    circuit.draw('mpl',fold=30, filename="FullCircuit.png")
    return circuit

circ = operatorA()
qc_compiled = transpile(circ, simulator)
result = simulator.run(qc_compiled, shots=100000).result()
counts = result.get_counts(qc_compiled)
plot_histogram(counts, title='Quantum Amplitude Estimation classical')
plt.show()
print(counts)
print((counts['0001']/100000 + c - 1/2)*imax/2/c)

