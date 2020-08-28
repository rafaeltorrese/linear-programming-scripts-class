#!/usr/bin/env python3
# _*_ coding: utf8

from itertools import chain
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

def create_array(data):
    "Create a matrix as numpy array from string data"
    if ";" in data:
        matrix_list = [row.split() for row in data.split(";")]
        m = len(matrix_list)  # number of constraints
        n = len(matrix_list[0])  # number of variables
        matrix = np.array(matrix_list, dtype=np.float).reshape(m, n)
    else:
        matrix = np.array(data.strip().split(), dtype=np.float)
    return matrix

def simplex(M, c, r):
    'Simplex algoritthm'
    # initilization
    M = M.astype(np.float)
    basics = np.zeros(len(r))    
    optimal = False
    iteration = 0
    while not status:
        # Update
        zj, net, objvalue = update(M, r, c, basics)
        
        # Optimality
        optimal = optimality_test(net)
        if optimal:
            break
            
        
        # Feasibility
        entry, leaving, basics = feasibility_test(M, r, net, basics)
        M, r = row_operations(M, r, entry, leaving)
        iteration += 1
        print(f'Interation number: {iteration}')
    # print(rhs, objvalue )
    
    
    
if __name__ == "__main__"    :
    # Data
    A = create_array("  1 2 1 ;  1 0 2   ;    1 4 0")
    cj = create_array("3 2 5")
    rhs = create_array(" 430 460 420 ")
    print(rhs)
    # A = np.array([[-1, 1, 0, 1, 0, 0],
    #               [0, -1, 2, 0, 1, 0], 
    #               [1, 1, 1, 0, 0, 1]])
    # cj = np.array([12, 15, 14, 0, 0, 0])
    # rhs = np.array([0, 0, 100])
    simplex(A, cj, rhs)
    
    
    print(list(chain("A B C".split(), "D E".split())))
    