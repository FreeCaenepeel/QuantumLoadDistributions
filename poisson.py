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


simulator = Aer.get_backend('aer_simulator')
kappa = 1

def poisson(k):
    return kappa**k*np.exp(-kappa)/math.factorial(k)

def integral(i, j):
    return sum([poisson(k) for k in range(i,j+1)])

def f(l,r):
    return np.arccos(np.sqrt(integral(l, math.floor(l+(r-l)/2))/integral(l,r)))

def binaryRepresentation(x):
    a = math.floor(x)
    b = x - a
    bits = []
    if a == 0: bits += [0]
    while (a != 0):
        a = a/2
        if (a != math.floor(a)): bits += [1]
        else: bits += [0]
        a = math.floor(a)
    while (len(bits) < 5):
        c = 2*b
        if c >= 1: bits += [1]
        else: bits += [0]
        b = c - math.floor(c)
    return bits
 

# IQFT
def qft_inverse5(circuit, q):
    circuit.h(q[4])
    circuit.cp(-2*np.pi/2**2,q[4],q[3])
    circuit.h(q[3])
    circuit.cp(-2*np.pi/2**3,q[4],q[2])
    circuit.cp(-2*np.pi/2**2,q[3],q[2])
    circuit.h(q[2])
    circuit.cp(-2*np.pi/2**4,q[4],q[1])
    circuit.cp(-2*np.pi/2**3,q[3],q[1])
    circuit.cp(-2*np.pi/2**2,q[2],q[1])
    circuit.h(q[1])
    circuit.cp(-2*np.pi/2**5,q[4],q[0])
    circuit.cp(-2*np.pi/2**4,q[3],q[0])
    circuit.cp(-2*np.pi/2**3,q[2],q[0])
    circuit.cp(-2*np.pi/2**2,q[1],q[0])
    circuit.h(q[0])
    circuit.swap(q[0],q[4])
    circuit.swap(q[1],q[3])
    return circuit

# IQFT
def qft5(circuit, q):
    circuit.swap(q[1],q[3])
    circuit.swap(q[0],q[4])
    circuit.h(q[0])
    circuit.cp(2*np.pi/2**2,q[1],q[0])
    circuit.cp(2*np.pi/2**3,q[2],q[0])
    circuit.cp(2*np.pi/2**4,q[3],q[0])
    circuit.cp(2*np.pi/2**5,q[4],q[0])
    circuit.h(q[1])
    circuit.cp(2*np.pi/2**2,q[2],q[1])
    circuit.cp(2*np.pi/2**3,q[3],q[1])
    circuit.cp(2*np.pi/2**4,q[4],q[1])
    circuit.h(q[2])
    circuit.cp(2*np.pi/2**2,q[3],q[2])
    circuit.cp(2*np.pi/2**3,q[4],q[2])
    circuit.h(q[3])
    circuit.cp(2*np.pi/2**2,q[4],q[3])
    circuit.h(q[4])
    return circuit

def qft_inverse6(circuit, q):
    circuit.h(q[5])
    circuit.cp(-2*np.pi/2**2,q[5],q[4])
    circuit.h(q[4])
    circuit.cp(-2*np.pi/2**3,q[5],q[3])
    circuit.cp(-2*np.pi/2**2,q[4],q[3])
    circuit.h(q[3])
    circuit.cp(-2*np.pi/2**4,q[5],q[2])
    circuit.cp(-2*np.pi/2**3,q[4],q[2])
    circuit.cp(-2*np.pi/2**2,q[3],q[2])
    circuit.h(q[2])
    circuit.cp(-2*np.pi/2**5,q[5],q[1])
    circuit.cp(-2*np.pi/2**4,q[4],q[1])
    circuit.cp(-2*np.pi/2**3,q[3],q[1])
    circuit.cp(-2*np.pi/2**2,q[2],q[1])
    circuit.h(q[1])
    circuit.cp(-2*np.pi/2**6,q[5],q[0])
    circuit.cp(-2*np.pi/2**5,q[4],q[0])
    circuit.cp(-2*np.pi/2**4,q[3],q[0])
    circuit.cp(-2*np.pi/2**3,q[2],q[0])
    circuit.cp(-2*np.pi/2**2,q[1],q[0])
    circuit.h(q[0])
    circuit.swap(q[1],q[4])
    circuit.swap(q[0],q[5])
    circuit.swap(q[2],q[3])
    return circuit

def qft6(circuit, q):
    circuit.swap(q[1],q[4])
    circuit.swap(q[0],q[5])
    circuit.swap(q[2],q[3])
    circuit.h(q[0])
    circuit.cp(2*np.pi/2**2,q[1],q[0])
    circuit.cp(2*np.pi/2**3,q[2],q[0])
    circuit.cp(2*np.pi/2**4,q[3],q[0])
    circuit.cp(2*np.pi/2**5,q[4],q[0])
    circuit.cp(2*np.pi/2**6,q[5],q[0])
    circuit.h(q[1])
    circuit.cp(2*np.pi/2**2,q[2],q[1])
    circuit.cp(2*np.pi/2**3,q[3],q[1])
    circuit.cp(2*np.pi/2**4,q[4],q[1])
    circuit.cp(2*np.pi/2**5,q[5],q[1])
    circuit.h(q[2])
    circuit.cp(2*np.pi/2**2,q[3],q[2])
    circuit.cp(2*np.pi/2**3,q[4],q[2])
    circuit.cp(2*np.pi/2**4,q[5],q[2])
    circuit.h(q[3])
    circuit.cp(2*np.pi/2**2,q[4],q[3])
    circuit.cp(2*np.pi/2**3,q[5],q[3])
    circuit.h(q[4])
    circuit.cp(2*np.pi/2**2,q[5],q[4])
    circuit.h(q[5])
    return circuit


def circuit():
    base = QuantumRegister(5, 'base')
    ancilla = QuantumRegister(6, 'ancilla')
    measurement = ClassicalRegister(4, 'measurement')
    circ = QuantumCircuit(base, ancilla, measurement)
    # Step 1
    prob = f(0,15)
    bits = binaryRepresentation(prob)
    for (i,b) in enumerate(bits):
        if b == 1: circ.x(5+i)
        circ.cry(2**(1-i),5+i,1)
    # uncompute
    for (i,b) in enumerate(bits):
        if b == 1: circ.x(5+i)
    
    # Step 2
    # define operator
    XX_matrix = np.array([[np.exp(1j * 2 * np.pi * 2 * f(0,7) / 4.0), 0]
                     ,[0, np.exp(1j * 2 * np.pi * 2 * f(8,15) / 4.0 )]
                     ])

    XX_operator = Operator(XX_matrix)

    XX2_matrix = np.matmul(XX_matrix, XX_matrix)
    XX2_operator = Operator(XX2_matrix)

    XX4_matrix = np.matmul(XX2_matrix, XX2_matrix)
    XX4_operator = Operator(XX4_matrix)

    XX8_matrix = np.matmul(XX4_matrix, XX4_matrix)
    XX8_operator = Operator(XX8_matrix)

    XX16_matrix = np.matmul(XX8_matrix, XX8_matrix)
    XX16_operator = Operator(XX16_matrix)

    cX = QuantumCircuit(1)
    cX.append(XX_operator, [0])
    custom = cX.to_gate().control(1)

    cX2 = QuantumCircuit(1)
    cX2.append(XX2_operator, [0])
    custom2 = cX2.to_gate().control(1)

    cX4 = QuantumCircuit(1)
    cX4.append(XX4_operator, [0])
    custom4 = cX4.to_gate().control(1)

    cX8 = QuantumCircuit(1)
    cX8.append(XX8_operator, [0])
    custom8 = cX8.to_gate().control(1)

    cX16 = QuantumCircuit(1)
    cX16.append(XX16_operator, [0])
    custom16 = cX16.to_gate().control(1)

    # hadamard on ancillaAngle
    circ.h(ancilla[0])
    circ.h(ancilla[1])
    circ.h(ancilla[2])
    circ.h(ancilla[3])
    circ.h(ancilla[4])

    circ.append(custom, [ancilla[0],1])
    circ.append(custom2,[ancilla[1],1])
    circ.append(custom4,[ancilla[2],1])
    circ.append(custom8,[ancilla[3],1])
    circ.append(custom16,[ancilla[4],1])

    # perform inverse QFT
    qft_inverse5(circ, ancilla)
    
    
    # rotate next clock qubit
    circ.cry(1/2**3, ancilla[0], 2)
    circ.cry(1/2**2, ancilla[1], 2)
    circ.cry(1/2**1, ancilla[2], 2)
    circ.cry(1/2**0, ancilla[3], 2)
    circ.cry(2, ancilla[4], 2)

    circ.swap(1,2)

    # uncompute
    qft5(circ,ancilla)


    XXinv_matrix = np.array([[np.exp(-1j * 2 * np.pi * 2 * f(0,7) / 4.0), 0]
                     ,[0, np.exp(-1j * 2 * np.pi * 2 * f(8,15) / 4.0)]
                     ])
    XXinv_operator = Operator(XXinv_matrix)

    XX2inv_matrix = np.matmul(XXinv_matrix, XXinv_matrix)
    XX2inv_operator = Operator(XX2inv_matrix)

    XX4inv_matrix = np.matmul(XX2inv_matrix, XX2inv_matrix)
    XX4inv_operator = Operator(XX4inv_matrix)

    XX8inv_matrix = np.matmul(XX4inv_matrix, XX4inv_matrix)
    XX8inv_operator = Operator(XX8inv_matrix)

    cXinv = QuantumCircuit(1)
    cXinv.append(XXinv_operator, [0])
    custominv = cXinv.to_gate().control(1)

    cX2inv = QuantumCircuit(1)
    cX2inv.append(XX2inv_operator, [0])
    custom2inv = cX2inv.to_gate().control(1)

    cX4inv = QuantumCircuit(1)
    cX4inv.append(XX4inv_operator, [0])
    custom4inv = cX4inv.to_gate().control(1)

    cX8inv = QuantumCircuit(1)
    cX8inv.append(XX8inv_operator, [0])
    custom8inv = cX8inv.to_gate().control(1)

    XX16inv_matrix = np.matmul(XX8inv_matrix, XX8inv_matrix)
    XX16inv_operator = Operator(XX16inv_matrix)

    cX16inv = QuantumCircuit(1)
    cX16inv.append(XX16inv_operator, [0])
    custom16inv = cX16inv.to_gate().control(1)

    # beware of the swap!
    circ.append(custominv, [ancilla[0],2])
    circ.append(custom2inv,[ancilla[1],2])
    circ.append(custom4inv,[ancilla[2],2])
    circ.append(custom8inv,[ancilla[3],2])
    circ.append(custom16inv,[ancilla[4],2])

    # hadamard on ancillaAngle
    circ.h(ancilla[0])
    circ.h(ancilla[1])
    circ.h(ancilla[2])
    circ.h(ancilla[3])
    circ.h(ancilla[4])

    # Step 3
    # define operator
    XX_matrix = np.array([[np.exp(1j * 2 * np.pi * 2 * f(0,3) / 4.0), 0, 0, 0]
                     ,[0, np.exp(1j * 2 * np.pi * 2 * f(4,7) / 4.0), 0, 0]
                     ,[0, 0, np.exp(1j * 2 * np.pi * 2 * f(8,11) / 4.0), 0]
                     ,[0, 0, 0, np.exp(1j * 2 * np.pi * 2 * f(12,15) / 4.0)]
                     ])

    XX_operator = Operator(XX_matrix)

    XX2_matrix = np.matmul(XX_matrix, XX_matrix)
    XX2_operator = Operator(XX2_matrix)

    XX4_matrix = np.matmul(XX2_matrix, XX2_matrix)
    XX4_operator = Operator(XX4_matrix)

    XX8_matrix = np.matmul(XX4_matrix, XX4_matrix)
    XX8_operator = Operator(XX8_matrix)

    XX16_matrix = np.matmul(XX8_matrix, XX8_matrix)
    XX16_operator = Operator(XX16_matrix)

    XX32_matrix = np.matmul(XX16_matrix, XX16_matrix)
    XX32_operator = Operator(XX32_matrix)

    cX = QuantumCircuit(2)
    cX.append(XX_operator, [0,1])
    custom = cX.to_gate().control(1)

    cX2 = QuantumCircuit(2)
    cX2.append(XX2_operator, [0,1])
    custom2 = cX2.to_gate().control(1)

    cX4 = QuantumCircuit(2)
    cX4.append(XX4_operator, [0,1])
    custom4 = cX4.to_gate().control(1)

    cX8 = QuantumCircuit(2)
    cX8.append(XX8_operator, [0,1])
    custom8 = cX8.to_gate().control(1)

    cX16 = QuantumCircuit(2)
    cX16.append(XX16_operator, [0,1])
    custom16 = cX16.to_gate().control(1)

    cX32 = QuantumCircuit(2)
    cX32.append(XX32_operator, [0,1])
    custom32 = cX32.to_gate().control(1)

    # hadamard on ancillaAngle
    circ.h(ancilla[0])
    circ.h(ancilla[1])
    circ.h(ancilla[2])
    circ.h(ancilla[3])
    circ.h(ancilla[4])
    circ.h(ancilla[5])

    circ.append(custom, [ancilla[0],1,2])
    circ.append(custom2,[ancilla[1],1,2])
    circ.append(custom4,[ancilla[2],1,2])
    circ.append(custom8,[ancilla[3],1,2])
    circ.append(custom16,[ancilla[4],1,2])
    circ.append(custom32,[ancilla[5],1,2])

    # perform inverse QFT
    qft_inverse6(circ,ancilla)

    # rotate next clock qubit
    circ.cry(1/2**4, ancilla[0], 3)
    circ.cry(1/2**3, ancilla[1], 3)
    circ.cry(1/2**2, ancilla[2], 3)
    circ.cry(1/2**1, ancilla[3], 3)
    circ.cry(1, ancilla[4], 3)
    circ.cry(2, ancilla[5], 3)

    # rotate base qubits
    circ.swap(1,2)
    circ.swap(1,3)
    # uncompute
    qft6(circ,ancilla)

    XXinv_matrix = np.array([[np.exp(-1j * 2 * np.pi * 2 * f(0,3) / 4.0), 0, 0, 0]
                     ,[0, np.exp(-1j * 2 * np.pi * 2 * f(4,7) / 4.0), 0, 0]
                     ,[0, 0, np.exp(-1j * 2 * np.pi * 2 * f(8,11) / 4.0), 0]
                     ,[0, 0, 0, np.exp(-1j * 2 * np.pi * 2 * f(12,15) / 4.0)]
                     ])
    XXinv_operator = Operator(XXinv_matrix)

    XX2inv_matrix = np.matmul(XXinv_matrix, XXinv_matrix)
    XX2inv_operator = Operator(XX2inv_matrix)

    XX4inv_matrix = np.matmul(XX2inv_matrix, XX2inv_matrix)
    XX4inv_operator = Operator(XX4inv_matrix)

    XX8inv_matrix = np.matmul(XX4inv_matrix, XX4inv_matrix)
    XX8inv_operator = Operator(XX8inv_matrix)

    cXinv = QuantumCircuit(2)
    cXinv.append(XXinv_operator, [0,1])
    custominv = cXinv.to_gate().control(1)

    cX2inv = QuantumCircuit(2)
    cX2inv.append(XX2inv_operator, [0,1])
    custom2inv = cX2inv.to_gate().control(1)

    cX4inv = QuantumCircuit(2)
    cX4inv.append(XX4inv_operator, [0,1])
    custom4inv = cX4inv.to_gate().control(1)

    cX8inv = QuantumCircuit(2)
    cX8inv.append(XX8inv_operator, [0,1])
    custom8inv = cX8inv.to_gate().control(1)

    XX16inv_matrix = np.matmul(XX8inv_matrix, XX8inv_matrix)
    XX16inv_operator = Operator(XX16inv_matrix)

    cX16inv = QuantumCircuit(2)
    cX16inv.append(XX16inv_operator, [0,1])
    custom16inv = cX16inv.to_gate().control(1)

    XX32inv_matrix = np.matmul(XX16inv_matrix, XX16inv_matrix)
    XX32inv_operator = Operator(XX32inv_matrix)

    cX32inv = QuantumCircuit(2)
    cX32inv.append(XX32inv_operator, [0,1])
    custom32inv = cX32inv.to_gate().control(1)

    # beware of the swap
    circ.append(custominv, [ancilla[0],2,3])
    circ.append(custom2inv,[ancilla[1],2,3])
    circ.append(custom4inv,[ancilla[2],2,3])
    circ.append(custom8inv,[ancilla[3],2,3])
    circ.append(custom16inv,[ancilla[4],2,3])
    circ.append(custom32inv,[ancilla[5],2,3])

    # hadamard on ancillaAngle
    circ.h(ancilla[0])
    circ.h(ancilla[1])
    circ.h(ancilla[2])
    circ.h(ancilla[3])
    circ.h(ancilla[4])
    circ.h(ancilla[5])

    # Step 4
    # define operator
    XX_matrix = np.array([[np.exp(1j * 2 * np.pi * 2 * f(0,1) / 4.0), 0, 0, 0, 0, 0, 0, 0]
                     ,[0, np.exp(1j * 2 * np.pi * 2 * f(2,3) / 4.0), 0, 0, 0, 0, 0, 0]
                     ,[0, 0, np.exp(1j * 2 * np.pi * 2 * f(4,5) / 4.0), 0, 0, 0, 0, 0]
                     ,[0, 0, 0, np.exp(1j * 2 * np.pi * 2 * f(6,7) / 4.0), 0, 0, 0, 0]
                     ,[0, 0, 0, 0, np.exp(1j * 2 * np.pi * 2 * f(8,9) / 4.0), 0, 0, 0]
                     ,[0, 0, 0, 0, 0, np.exp(1j * 2 * np.pi * 2 * f(10,11) / 4.0), 0, 0]
                     ,[0, 0, 0, 0, 0, 0, np.exp(1j * 2 * np.pi * 2 * f(12,13) / 4.0), 0]
                     ,[0, 0, 0, 0, 0, 0, 0, np.exp(1j * 2 * np.pi * 2 * f(14,15) / 4.0)]
                     ])

    XX_operator = Operator(XX_matrix)

    XX2_matrix = np.matmul(XX_matrix, XX_matrix)
    XX2_operator = Operator(XX2_matrix)

    XX4_matrix = np.matmul(XX2_matrix, XX2_matrix)
    XX4_operator = Operator(XX4_matrix)

    XX8_matrix = np.matmul(XX4_matrix, XX4_matrix)
    XX8_operator = Operator(XX8_matrix)

    XX16_matrix = np.matmul(XX8_matrix, XX8_matrix)
    XX16_operator = Operator(XX16_matrix)

    XX32_matrix = np.matmul(XX16_matrix, XX16_matrix)
    XX32_operator = Operator(XX32_matrix)

    cX = QuantumCircuit(3)
    cX.append(XX_operator, [0,1,2])
    custom = cX.to_gate().control(1)

    cX2 = QuantumCircuit(3)
    cX2.append(XX2_operator, [0,1,2])
    custom2 = cX2.to_gate().control(1)

    cX4 = QuantumCircuit(3)
    cX4.append(XX4_operator, [0,1,2])
    custom4 = cX4.to_gate().control(1)

    cX8 = QuantumCircuit(3)
    cX8.append(XX8_operator, [0,1,2])
    custom8 = cX8.to_gate().control(1)

    cX16 = QuantumCircuit(3)
    cX16.append(XX16_operator, [0,1,2])
    custom16 = cX16.to_gate().control(1)

    cX32 = QuantumCircuit(3)
    cX32.append(XX32_operator, [0,1,2])
    custom32 = cX32.to_gate().control(1)

    # hadamard on ancillaAngle
    circ.h(ancilla[0])
    circ.h(ancilla[1])
    circ.h(ancilla[2])
    circ.h(ancilla[3])
    circ.h(ancilla[4])
    circ.h(ancilla[5])

    circ.append(custom, [ancilla[0],1,2,3])
    circ.append(custom2,[ancilla[1],1,2,3])
    circ.append(custom4,[ancilla[2],1,2,3])
    circ.append(custom8,[ancilla[3],1,2,3])
    circ.append(custom16,[ancilla[4],1,2,3])
    circ.append(custom32,[ancilla[5],1,2,3])

    # perform inverse QFT
    qft_inverse6(circ,ancilla)

    # rotate next clock qubit
    circ.cry(1/2**4, ancilla[0], 4)
    circ.cry(1/2**3, ancilla[1], 4)
    circ.cry(1/2**2, ancilla[2], 4)
    circ.cry(1/2**1, ancilla[3], 4)
    circ.cry(1, ancilla[4], 4)
    circ.cry(2, ancilla[5], 4)

    # rotate base qubits
    circ.swap(3,4)
    circ.swap(2,3)
    circ.swap(1,2)
    # uncompute
    qft6(circ,ancilla)

    XXinv_matrix = np.array([[np.exp(-1j * 2 * np.pi * 2 * f(0,1) / 4.0), 0, 0, 0, 0, 0, 0, 0]
                     ,[0, np.exp(-1j * 2 * np.pi * 2 * f(2,3) / 4.0), 0, 0, 0, 0, 0, 0]
                     ,[0, 0, np.exp(-1j * 2 * np.pi * 2 * f(4,5) / 4.0), 0, 0, 0, 0, 0]
                     ,[0, 0, 0, np.exp(-1j * 2 * np.pi * 2 * f(6,7) / 4.0), 0, 0, 0, 0]
                     ,[0, 0, 0, 0, np.exp(-1j * 2 * np.pi * 2 * f(8,9) / 4.0), 0, 0, 0]
                     ,[0, 0, 0, 0, 0, np.exp(-1j * 2 * np.pi * 2 * f(10,11) / 4.0), 0, 0]
                     ,[0, 0, 0, 0, 0, 0, np.exp(-1j * 2 * np.pi * 2 * f(12,13) / 4.0), 0]
                     ,[0, 0, 0, 0, 0, 0, 0, np.exp(-1j * 2 * np.pi * 2 * f(14,15) / 4.0)]
                     ])
    XXinv_operator = Operator(XXinv_matrix)

    XX2inv_matrix = np.matmul(XXinv_matrix, XXinv_matrix)
    XX2inv_operator = Operator(XX2inv_matrix)

    XX4inv_matrix = np.matmul(XX2inv_matrix, XX2inv_matrix)
    XX4inv_operator = Operator(XX4inv_matrix)

    XX8inv_matrix = np.matmul(XX4inv_matrix, XX4inv_matrix)
    XX8inv_operator = Operator(XX8inv_matrix)

    cXinv = QuantumCircuit(3)
    cXinv.append(XXinv_operator, [0,1,2])
    custominv = cXinv.to_gate().control(1)

    cX2inv = QuantumCircuit(3)
    cX2inv.append(XX2inv_operator, [0,1,2])
    custom2inv = cX2inv.to_gate().control(1)

    cX4inv = QuantumCircuit(3)
    cX4inv.append(XX4inv_operator, [0,1,2])
    custom4inv = cX4inv.to_gate().control(1)

    cX8inv = QuantumCircuit(3)
    cX8inv.append(XX8inv_operator, [0,1,2])
    custom8inv = cX8inv.to_gate().control(1)

    XX16inv_matrix = np.matmul(XX8inv_matrix, XX8inv_matrix)
    XX16inv_operator = Operator(XX16inv_matrix)

    cX16inv = QuantumCircuit(3)
    cX16inv.append(XX16inv_operator, [0,1,2])
    custom16inv = cX16inv.to_gate().control(1)

    XX32inv_matrix = np.matmul(XX16inv_matrix, XX16inv_matrix)
    XX32inv_operator = Operator(XX32inv_matrix)

    cX32inv = QuantumCircuit(3)
    cX32inv.append(XX32inv_operator, [0,1,2])
    custom32inv = cX32inv.to_gate().control(1)

    # beware of the swap
    circ.append(custominv, [ancilla[0],2,3,4])
    circ.append(custom2inv,[ancilla[1],2,3,4])
    circ.append(custom4inv,[ancilla[2],2,3,4])
    circ.append(custom8inv,[ancilla[3],2,3,4])
    circ.append(custom16inv,[ancilla[4],2,3,4])
    circ.append(custom32inv,[ancilla[5],2,3,4])

    # hadamard on ancillaAngle
    circ.h(ancilla[0])
    circ.h(ancilla[1])
    circ.h(ancilla[2])
    circ.h(ancilla[3])
    circ.h(ancilla[4])
    circ.h(ancilla[5])
    
    circ.measure([1,2,3,4],measurement)
   # circ.draw('mpl',fold=30, filename="FullCircuit.png")
    return circ
    
    



circ = circuit()
qc_compiled = transpile(circ, simulator)
result = simulator.run(qc_compiled, shots=10000).result()
counts = result.get_counts(qc_compiled)
plot_histogram(counts, title='Poisson distribution (lambda = 1)')
plt.show()

