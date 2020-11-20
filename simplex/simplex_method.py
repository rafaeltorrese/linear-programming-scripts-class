#!/usr/bin/env python3
# _*_ coding: utf8

import numpy as np
np.set_printoptions(suppress=True)

# update
def update(A, rhs, cj, basics):
    zj = basics.dot(A)
    net_evaluation = cj - zj
    zvalue = basics.dot(rhs)
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
    index_ratios = np.where(ratios < 0)  # negative ratios
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

def simplex(M, c, r, nvars):
    """ Simplex algoritthm

    Parameters:
    ------------
    M: matrix
        Body matrix

    c: array
        Coefficients in objective function

    r: array
        Right-hand side vector

    nvars: int
        Number of variables (x1, x2, x3, ..., xn)

    """
    # initilization
    positions = np.where(c == 0)[0]
    solution_vector = np.zeros(c.size)  # [0, 0 ,0 , .. , 0]
    basics = c[nvars: ].astype(float)  # creates a copy
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
        positions[leaving] = entry
        M, r = row_operations(M, r, entry, leaving)
        iteration += 1
        print(f'Interation: {iteration}')
        print(f"Leaving: Row{leaving + 1},  Entry: Column{entry + 1}")
        print(M, "\n")
    solution_vector[positions] = rhs
    print( solution_vector)
    print(objvalue )



if __name__ == "__main__"    :
    # Data
    # A = create_array("1 2 1 1 0 0; 3 0 2 0 1 0; 1 4 0 0 0 1")
    # cj = create_array("3 2 5 0 0 0" )
    # rhs = create_array("430 460 420")

    # A = create_array("20 9 6 1 1 0; 10 4 2 1 0 1")
    # cj = create_array("240 104 60 19 0 0" )
    # rhs = create_array("20 10")


    # A = create_array("2 3 2 1 0 0; 4 0 3 0 1 0; 2 5 0 0 0 1")
    # cj = create_array("4 3 6 0 0 0" )
    # rhs = create_array("440 470 430")

    # A = create_array("6 4  1 0 0 0;1 2 0 1 0 0;-1 1 0 0 1 0; 0 1 0 0 0 1")
    # cj = create_array("5 4 0 0 0 0" )
    # rhs = create_array("24 6 1 2")

    # A = create_array("1 4 1 0 0; 3 1 0 1 0; 1 1 0 0 1")
    # cj = create_array("2 5 0 0 0" )
    # rhs = create_array("24 21   9")

    cj = create_array("10 5 7 0 0 0" )
    A = create_array("1 1 1 1 0 0; 3 1 2 0 1 0; 1 0 0 0 0 1")
    rhs = create_array("800 1000 150")


    simplex(A, cj, rhs, nvars=3)


