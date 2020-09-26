import qiskit as qk
import numpy as np
import matplotlib.pyplot as plt
import math

def gradient_descent():
    
    '''
    Description: This function serves the purpose of calclating the most optimal value of thetha for our Rx gate which is applied to qubit 0 initially in |0> state
    
    Returns:
    
    Rx_thetha - angle in radians, which corresponds to most optimal solution of our function
    
    '''

    #Get 1000 evenly spaced numbers between -1 and 3 (arbitratil chosen to ensure steep curve)
    x = np.linspace(0,13,10000)

    #Plot the curve
    f2 = np.vectorize(function1)
    plt.title("probability curve")
    plt.plot(x, f2(x))
    plt.show()

    Rx_thetha = step(0.5, 0, 0.00001, 0.01)

    return Rx_thetha


def function1(x):
    '''
    Description: This formally defines thw function and calculate its value at x. Our function is built upon Rx gate
    
    Arguments:
    
    x - numerical input value, at which we want to calculate the derivative 
    
    Returns:
    
    f_x - value of function1 at input point x
    
    '''

    return ((math.cos(x/2)**2) - (math.sin(x/2)**2))**2


def deriv_f1(x):
    
    '''
    Description: This function calculates the derivative of function f1 at point x.
    
    Arguments:
    
    x - numerical input value, at which we want to calculate the derivative 
    
    Returns:
    
    dx - value of derivative of f1 at input point x
    
    '''
    
    dx = (-2)*(math.sin(x))*((math.cos(x/2)**2) - (math.sin(x/2)**2)) #derivative of f1
    return dx


def step(x_new, x_prev, precision, learn_rate):
    
    '''
    Description: This function takes in an initial or previous value for x, updates it based on steps taken via the learning rate and outputs the most minimum value of x that reaches the precision satisfaction.
    
    Arguments:
    
    x_new - a starting value of x that will get updated based on the learning rate
    
    x_prev - the previous value of x that is getting updated to the new one
    
    precision - a precision that determines the stop of the stepwise descent 
    
    learn_rate - the learning rate (size of each descent step)
    
    Output:
    
    1. Prints out the latest new value of x which equates to the minimum we are looking for
    2. Prints out the the number of x values which equates to the number of gradient descent steps
    3. Plots a first graph of the function with the gradient descent path
    4. Plots a second graph of the function with a zoomed in gradient descent path in the important area

    Returns:
    
    x_new - value of x which gives the minimal solution for the given function, as calculated by Gradient Descent method
    
    '''
    
    # create empty lists where the updated values of x and y wil be appended during each iteration
    
    x_list, y_list = [x_new], [function1(x_new)]
    # keep looping until your desired precision
    while abs(x_new - x_prev) > precision:
        print(x_prev)
        
        # change the value of x
        x_prev = x_new
        
        # get the derivation of the old value of x
        dx = - deriv_f1(x_prev)
        
        # get your new value of x by adding the previous, the multiplication of the derivative and the learning rate
        x_new = x_prev + (learn_rate * dx)
        
        # append the new value of x to a list of all x-s for later visualization of path
        x_list.append(x_new)
        
        # append the new value of y to a list of all y-s for later visualization of path
        y_list.append(function1(x_new))

    print ("Local minimum occurs at: "+ str(x_new))
    print ("Number of steps: " + str(len(x_list)))

    x = np.linspace(0,13,10000)

    f2 = np.vectorize(function1)

    plt.subplot(1,2,1)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,f2(x), c="r")
    plt.title("Gradient descent technique on probability function construcuted for Rx Gate")

    plt.subplot(1,2,2)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,f2(x), c="r")
    plt.xlim([1.2,1.8])
    plt.title("Zoomed in Gradient descent to Key Area")
    
    plt.show()

    return x_new


def compare_result(results):

    '''
    Description: This function is designed to compare the results of various simulations
    
    Output:
    
    1) Prints the bar plot for all 4 different measurement simulations 
    
    '''

    count = 1
    for i in results:
        total = 0
        states = list(i.keys()) 
        values = list(i.values()) 

        for j in values:
            total += j
        for j in range(len(values)):
            values[j] = values[j]*100/total

        plt.subplot(2,2,count)
        plt.bar(states, values, color ="r") 
        #plt.xlabel("States found in measurement") 
        plt.ylabel("Outcome in Percentage") 
        plt.title("Results with "+str(int(total))+" measurements")
        plt.ylim([0,100])
        plt.plot()
        count += 1

    plt.show() 



if __name__ == '__main__':
    
    # Finding the most optimal angle for Rx gate
    Rx_thetha = 0
    Rx_thetha = gradient_descent()

    # Creating Qubits
    q = qk.QuantumRegister(2)
    # Creating Classical Bits
    c = qk.ClassicalRegister(2)

    # Creating quantum circuit
    circuit = qk.QuantumCircuit(q, c)
    # Applying Rx gate to |0> state to convert it into superposition of |0> and |1> with equal probabilities
    circuit.rx(Rx_thetha,q[0])
    # Applying Ry gate to |0> state to convert it into |1> state
    circuit.ry(3.14,q[1])
    # Applying a CNOT gate on target qubit q[1] and control qubit q[0]
    circuit.cnot(q[0],q[1])
    # Measure the values
    circuit.measure(q,c)
    print (circuit)

    # Using Qiskit Aer's Qasm Simulator
    simulator = qk.BasicAer.get_backend('qasm_simulator')

    input_shots = [1,10,100,1000]
    output_count = []
    for i in input_shots:
        # Simulating the circuit using the simulator to get the result
        job = qk.execute(circuit, simulator, shots = i)
        result = job.result()

        # Getting the aggregated binary outcomes of the circuit.
        counts = result.get_counts(circuit)
        print("For", i, "simulations", "the results are :")
        print(counts)
        output_count.append(counts)

    compare_result(output_count)
