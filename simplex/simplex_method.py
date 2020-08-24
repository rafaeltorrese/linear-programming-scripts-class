#!/usr/bin/env python3
# _*_ coding: utf8

import numpy as np
# np.set_printoptions(suppress=True)

# update
def update(A, rhs, cj, basics):
    A = A.astype(np.float)
    zj = basics.dot(A)
    net_evaluation = cj - zj
    zvalue = basics.dot(rhs)
    return zj, net_evaluation, zvalue
    

def optimality_test(net_evaluation): 
    optimal = np.all(net_evaluation <= 0)
    if optimal:
        print("Optimal Solution found")
    return optimal

def feasibility_test(A, rhs, net_evaluation, basics ):
    entry = np.argmax(net_evaluation)
    A = A.astype(np.float)
    zero_values = np.any(A[:,entry])
    if zero_values:
        indx_zeros = np.where(A[:,entry] <= 0)[0]   # index with zero values in column selected
        A[indx_zeros, entry] = 0.00001
        # print("Feasibility\n", A, "\n")
    ratios = rhs / A[:, entry]
    leaving = np.argmin(ratios)
    basics[leaving] = cj[entry]  # update basic
    pivot = leaving, entry
    return entry, leaving, basics

def row_operations(A, entry, leaving):
    if A[leaving, entry] != 1:
        A = A[leaving] / A[leaving, entry]
    for i in range(len(A)):
        if i == leaving:   # same row
            continue
        factor = -A[i, entry]  # negative factor
        A[i, :] += factor*A[leaving]
        rhs[i] += factor*rhs[leaving]
    return A, rhs

def simplex(M, c, r):
    'Simplex algoritthm'
    # initilization
    basics = np.zeros(len(rhs))    
    status = False
    iteration = 0
    while not status:
        # Update
        zj, net, objvalue = update(M, r, c, basics)
        
        # Optimality
        status = optimality_test(net)
        if status:
            break
            
        
        # Feasibility
        entry, leaving, basics = feasibility_test(M, r, net, basics)
        M, r = row_operations(M, entry, leaving)
        iteration += 1
        print(f'Interation number: {iteration}')
    print(M,rhs, objvalue )
    
    
    
if __name__ == "__main__"    :
    # Data
    A = np.array([[1, 1, 1, 0],[2, 1, 0, 1]])
    cj = np.array([3, 4, 0, 0])
    rhs = np.array([450, 600])
    simplex(A, cj, rhs)
    
    
    
    