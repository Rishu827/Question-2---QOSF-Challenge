import qiskit as qk
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    
    q = qk.QuantumRegister(6)
    
    # Creating Classical Bits
    c = qk.ClassicalRegister(6)

    # Creating quantum circuit 
    #Sender's side, where q[0] is sent as it is and q[1] is sent by using encoding technique discussed in solution
    circuit = qk.QuantumCircuit(q, c)
    circuit.cnot(q[1],q[2])
    circuit.cnot(q[1],q[3])
    circuit.h(q[1])
    circuit.h(q[2])
    circuit.h(q[3])
    #Ancillaries
    circuit.h(q[4])
    circuit.h(q[5])
    #Receiver's side
    circuit.cnot(q[4],q[1])
    circuit.cnot(q[4],q[2])
    circuit.cnot(q[5],q[1])
    circuit.cnot(q[5],q[3])
    circuit.h(q[1])
    circuit.h(q[2])
    circuit.h(q[3])
    #measuring ancillaries to know the error syndrome and conclude corresponding 
    circuit.h(q[4])
    circuit.h(q[5])
    # Measure the values
    circuit.measure(q,c)
    print (circuit)

    # Using Qiskit Aer's Qasm Simulator
    simulator = qk.BasicAer.get_backend('qasm_simulator')

    
    # Simulating the circuit using the simulator to get the result
    job = qk.execute(circuit, simulator, shots = 1024)
    result = job.result()
    counts = result.get_counts(circuit)
    print(counts)
       