#!/usr/bin/env python3
# _*_ coding: utf8

import numpy as np
np.set_printoptions(suppress=True)

# update
def update(A, rhs, cj, basics):
    zj = basics.dot(A)
    net_evaluation = cj - zj
    zvalue = basics.dot(rhs)
    # print(zj, net_evaluation, zvalue, rhs)
    return zj, net_evaluation, zvalue


def optimality_test(net_evaluation):
    optimal = np.all(net_evaluation <= 0)
    if optimal:
        print("Optimal Solution found")
    return optimal

def feasibility_test(A, rhs, net_evaluation, basics, cj ):
    entry = np.argmax(net_evaluation)
    zero_values = np.any(A[:,entry])
    if zero_values:
        indx_zeros = np.where(A[:,entry] == 0)   # index with zero values in the column key
        A[indx_zeros, entry] = 1e-20
    if np.any(rhs == 0) and np.any(A[:,entry] < 0):  # degeneracy
        rhs_zeros = np.where(rhs == 0)  # indexes where rhs is zero
        rhs[rhs_zeros] = 1e-20
    ratios = rhs / A[:, entry]  # dividing by entry column of A
    index_ratios = np.where(ratios < 0)
    ratios[index_ratios] = np.infty
    leaving = np.argmin(ratios)
    basics[leaving] = cj[entry]  # update basic
    return entry, leaving, basics

def row_operations(A, r, entry, leaving):
    num_rows = A.shape[0]
    pivot = A[leaving, entry]
    if pivot != 1:
        A[leaving] = A[leaving] / pivot
        r[leaving] = r[leaving] / pivot
    for i in range(num_rows):
        if i == leaving:   # same row
            continue
        factor = -A[i, entry]  # negative factor
        A[i, :] += factor*A[leaving]
        r[i] += factor*r[leaving]
    return A, r

def create_array(data):
    "Create a matrix as numpy array from string data"
    if ";" in data:
        return np.array([row.split() for row in data.split(";")], dtype=np.float)
    return np.array(data.strip().split(), dtype=np.float)

def simplex(M, c, r):
    'Simplex algoritthm'
    # initilization
    basics = np.zeros(len(r))
    optimal = False
    iteration = 0
    while not optimal:
        # Update
        zj, net, objvalue = update(M, r, c, basics)
        # Optimality
        optimal = optimality_test(net)
        if optimal:
            break
        # Feasibility
        entry, leaving, basics = feasibility_test(M, r, net, basics, c)
        M, r = row_operations(M, r, entry, leaving)
        iteration += 1
        print(f'Interation: {iteration}')
        print(f"Leaving: Row{leaving + 1},  Entry: Column{entry + 1}")
        print(M, "\n")
    print(rhs, objvalue )



if __name__ == "__main__"    :
    # Data
    # A = create_array("1 2 1 1 0 0; 3 0 2 0 1 0; 1 4 0 0 0 1")
    # cj = create_array("3 2 5 0 0 0" )
    # rhs = create_array("430 460 420")

    A = create_array("20 9 6 1 1 0; 10 4 2 1 0 1")
    cj = create_array("240 104 60 19 0 0" )
    rhs = create_array("20 10")
    simplex(A, cj, rhs)



