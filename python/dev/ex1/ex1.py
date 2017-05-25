import csv
import numpy as np

def compute_cost(X,y, theta):
    J = 0

    m = len(X);

    J = (1/(2*m)) * sum((X.dot(theta) - y)**2);

    return J

def gradient_descent(X,y,theta, alpha, iterations):
    m = len(X)
    
    J_history = [0]*iterations

    for i in range(iterations):
        thetaTemp = theta
        dif =X.dot(thetaTemp)- y;
        mult = np.multiply(np.vstack([dif, dif]).T,X)
        theta = theta - (alpha/m)*sum(mult)
        J_history[i] = compute_cost(X,y,theta)
        print (J_history[i])
        print(theta)


with open('ex1data1.py.txt') as csvfile:
    read = csv.DictReader(csvfile)
    data=list(read)

m = len(data)

#read data into array
XyList = list((  float(data[i]['x1']), float(data[i]['x2']) ) for i in range(m))

Xy = np.array(XyList)

X = Xy[:,0]

y = Xy[:,1]

#extend with ones
X = np.vstack([np.ones(m), X]).T

theta = np.array([0, 0])

iterations=1500

alpha = 0.01

cost = compute_cost(X,y,theta)

print(cost)

print(sum(X))

gradient_descent(X,y, theta, alpha, iterations)
