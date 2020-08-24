#!/usr/bin/env python3
# _*_ coding: utf8

import numpy as np

# update
def update():
    global  zj, net_evaluation
    zj = basics.dot(A)
    net_evaluation = cj - zj
    zvalue = basics.dot(rhs)
    print(zj, net_evaluation, zvalue)
    print(A)
    
def optimality_test(): 
    optimal = np.all(net_evaluation <= 0)
    print("Optimality test: \n", net_evaluation)
    if optimal:
        print("Optimal Solution found")
    else:
        print("Next Iteration")
    return optimal

def feasibility_test():
    global entry, pivot, leaving, ratios
    entry = np.argmax(net_evaluation)
    print("Feasibility: \n", np.where(A[:, entry] == 0))
    mask = np.where(A[:, entry] == 0)
    # A[mask[0], entry] = 10
    print("feasibility\n", A)
    ratios = rhs / A[:, entry]
    leaving = np.argmin(ratios)
    basics[leaving] = cj[entry]  # update basic
    pivot = leaving, entry
    print(pivot)

def row_operations():
    for i in range(len(A)):
        if i == leaving:   # same row
            continue
        factor = -A[i, entry]  # negative factor
        A[i, :] += factor*A[leaving]
        rhs[i] += factor*rhs[leaving]
    print(A, rhs)

def simplex():
    # initilization
    global basics
    basics = np.zeros(len(rhs))    
    update()
    optimality_test()
    feasibility_test()
    row_operations()
    
    update()
    optimality_test()
    # feasibility_test()
    
if __name__ == "__main__"    :
    # Data
    A = np.array([[1, 1, 1, 0],[2, 1, 0, 1]])
    cj = np.array([3, 4, 0, 0])
    rhs = np.array([450, 600])
    simplex()
    
    
    
    