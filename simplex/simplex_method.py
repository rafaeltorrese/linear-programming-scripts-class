#!/usr/bin/env python3
# _*_ coding: utf8

import numpy as np
np.set_printoptions(precision=1, suppress=True)

# update
def update(A, rhs, cj, basics):
    zj = basics.dot(A)
    net_evaluation = cj - zj
    zvalue = basics.dot(rhs)
    return zj, net_evaluation, zvalue


def optimality_test(net_evaluation, sense):
    optimal = np.all(sense * net_evaluation <= 0)
    if optimal:
        print("Optimal Solution found")
    return optimal

def feasibility_test(A, rhs, net_evaluation, basics, cj , sense):
    entry = np.argmax(sense * net_evaluation)
    A[ : , entry] = np.where(A[:,entry] == 0, 1e-20, A[ : , entry]) # if there exists zero values in key column
    if np.any(A[ : , entry] < 0):  # degeneracy, if rhs has zero values and entry column has negative values
        rhs = np.where(rhs == 0, 1e-20, rhs)  # indexes where rhs is zero
    ratios = rhs / A[:, entry]  # dividing by entry column of A
    index_ratios = np.where(ratios < 0)[0]  # if there are negative ratios
    ratios[index_ratios] = np.infty  # penalty  negative ratios, this values are not taking into account
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
        A[i, :] += factor*A[leaving]  # update row
        r[i] += factor*r[leaving]  # update rhs row
    return A, r

def create_array(data):
    "Create a matrix as numpy array from string data"
    if ";" in data:
        return np.array([row.split() for row in data.split(";")], dtype=np.float)
    return np.array(data.strip().split(), dtype=np.float)

def simplex(M, c, r, nvars, direction=1):
    """ Simplex algorithm

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

    direction: {1,-1}
        For maximization problems use 1, or -1 in minimization problems instead 
    """
    # initilization
    positions = np.where(M[ : , nvars:] == 1)[1] + nvars # only positions of columns with values equal to one
    solution_vector = np.zeros(c.size)  # [0, 0 ,0 , .. , 0]
    basics = c[positions].astype(float)  # creates a copy
    optimal = False
    iteration = 0
    while not optimal:
        # Update
        zj, net, objvalue = update(M, r, c, basics)
        # Optimality
        optimal = optimality_test(net, direction)
        if optimal:
            break
        # Feasibility
        entry, leaving, basics = feasibility_test(M, r, net, basics, c, direction)
        M, r = row_operations(M, r, entry, leaving)
        iteration += 1
        print(f'Iteration: {iteration}')
        variable_leaving = positions[leaving]   # only for tracking position variable ("x1, x2, ... , etc")
        print(f"Leaving: Variable {variable_leaving + 1},  Entering: Variable {entry + 1}")
        print(M, "\n")
        positions[leaving] = entry
    solution_vector[positions] = rhs
    print(solution_vector)
    print(objvalue )
    print(r)
    print(positions + 1)



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

    # cj = create_array("12 20 0 0 1000 1000" ) 
    # A = create_array("6 8 -1 0 1 0; 7 12 0 -1 0 1")
    # rhs = create_array("100 120")

    # cj = create_array("3 -1  0  0 0 -1000" ) 
    # A = create_array("2 1 1 0 0 0; 1 3 0 -1 0 1; 0 1 0 0 1 0")
    # rhs = create_array("2 3 4")

    simplex(A, cj, rhs, nvars=3, direction=1)


