#!/usr/bin/env python3
# _*_ coding: utf8

import numpy as np
np.set_printoptions(suppress=True)

# update
def update(A, rhs, cj, basics):
    zj = basics.dot(A)
    net_evaluation = cj - zj
    zvalue = basics.dot(rhs)
    print(zj, net_evaluation, zvalue)
    return zj, net_evaluation, zvalue
    

def optimality_test(net_evaluation): 
    optimal = np.all(net_evaluation <= 0)
    if optimal:
        print("Optimal Solution found")
    return optimal

def feasibility_test(A, rhs, net_evaluation, basics ):
    entry = np.argmax(net_evaluation)
    zero_values = np.any(A[:,entry])
    if zero_values:
        indx_zeros = np.where(A[:,entry] <= 0)[0]   # index with zero values in column selected
        A[indx_zeros, entry] = 0.00001
    ratios = rhs / A[:, entry]
    leaving = np.argmin(ratios)
    basics[leaving] = cj[entry]  # update basic
    return entry, leaving, basics

def row_operations(A, r, entry, leaving):
    print(f"pivot: {leaving, entry}")    
    if A[leaving, entry] != 1:
        pivot = A[leaving, entry]
        A[leaving] = A[leaving] / pivot
        r[leaving] = r[leaving] / pivot
    for i in range(len(A)):
        if i == leaving:   # same row
            continue
        factor = -A[i, entry]  # negative factor
        A[i, :] += factor*A[leaving]
        r[i] += factor*r[leaving]
    return A, r

def simplex(M, c, r):
    'Simplex algoritthm'
    # initilization
    M = M.astype(np.float)
    basics = np.zeros(len(r))    
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
        M, r = row_operations(M, r, entry, leaving)
        iteration += 1
        print(f'Interation number: {iteration}')
    # print(rhs, objvalue )
    
    
    
if __name__ == "__main__"    :
    # Data
    A = np.array([[-1, 1, 0, 1, 0, 0],
                  [0, -1, 2, 0, 1, 0], 
                  [1, 1, 1, 0, 0, 1]])
    cj = np.array([12, 15, 14, 0, 0, 0])
    rhs = np.array([0, 0, 100])
    simplex(A, cj, rhs)
    
    
    
    