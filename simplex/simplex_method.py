#!/usr/bin/env python3
# _*_ coding: utf8 _*_

import numpy as np
np.set_printoptions(precision=4, suppress=True)

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
    ratios = np.full(rhs.size, np.inf)  # size of Cb with infinite expressions 
    entry = np.argmax(sense * net_evaluation)
    entry_column = A[ : , entry]

    positive_values = np.where(entry_column > 0)[0] # if there exists zero values in key column
    ratios[positive_values] = rhs[positive_values] / entry_column[positive_values]

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
    #
    slacks = np.where(c[nvars:] == 0)[0]
    num_slacks = slacks.size
    num_artificials = len(c[nvars+num_slacks:])
    #
    x_labels = ["X"+str(i) for i in range(1,nvars+1)]
    slack_lables = ["S"+str(i+1) for i in range(num_slacks)]
    artificial_labels = ["A"+str(i+1) for i in range(num_artificials)]
    labels = x_labels + slack_lables + artificial_labels
    #
    basics = c[positions].astype(float)  # creates a copy of basis values
    optimal = False
    iteration = 0
    while not optimal:
        # Update
        zj, net, objvalue = update(M, r, c, basics)
        print(dict(zip(labels, solution_vector)), objvalue, "\n")
        # Optimality
        optimal = optimality_test(net, direction)
        # 
        if optimal:
            break
        # Feasibility
        entry, leaving, basics = feasibility_test(M, r, net, basics, c, direction)
        variable_leaving = positions[leaving]   # only for tracking position variable ("x1, x2, ... , etc")
        positions[leaving] = entry
        iteration += 1
        print(f'Iteration: {iteration}')
        print(f"Leaving: {labels[variable_leaving]},  Entering: {labels[entry]}")
        M, r = row_operations(M, r, entry, leaving)  # Gauss Jordan
        #
        solution_vector[positions] = rhs
        solution_vector[variable_leaving] = 0
        print(M, "\n")
    return A, rhs, objvalue, solution_vector


def two_phase(M, c, r, nvars, direction=1):
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
    two_phase_objective = np.where((c == -1000)|(c == 1000), 1, 0).astype(float)
    positions = np.where(M[ : , nvars:] == 1)[1] + nvars # only positions of columns with values equal to one
    solution_vector = np.zeros(c.size)  # [0, 0 ,0 , .. , 0]
    basics = two_phase_objective[positions].astype(float)  # creates a copy
    #
    slacks = np.where(c[nvars:] == 0)[0]
    num_slacks = slacks.size
    num_artificials = len(c[nvars+num_slacks:])
    #
    x_labels = ["X"+str(i) for i in range(1,nvars+1)]
    slack_lables = ["S"+str(i+1) for i in range(num_slacks)]
    artificial_labels = ["A"+str(i+1) for i in range(num_artificials)]
    labels = x_labels + slack_lables + artificial_labels
    optimal = False
    iteration = 0
    print("="*20 + " Starting Phase I " + "="*20 )
    while not optimal:
        # Update
        zj, net, objvalue = update(M, r, two_phase_objective, basics)
        print(dict(zip(labels,solution_vector)), objvalue, "\n")
        # Optimality
        optimal = optimality_test(net, -1)
        if optimal:
            break
        # Feasibility
        entry, leaving, basics = feasibility_test(M, r, net, basics, two_phase_objective, -1)
        variable_leaving = positions[leaving]   # only for tracking position variable ("x1, x2, ... , etc")
        positions[leaving] = entry
        iteration += 1
        print(f'Iteration: {iteration}')
        print(f"Leaving: {labels[variable_leaving]},  Entering: {labels[entry]}")
        M, r = row_operations(M, r, entry, leaving)  # Gauss Jordan
        #
        solution_vector[positions] = r
        solution_vector[variable_leaving] = 0
        print(M, "\n")
        
    #
    #
    print(20*"=" + " Starting Phase II " + 20*"=")
    basics = c[positions].astype(float)  # creates a copy
    optimal = False
    while not optimal:
        # Update
        zj, net, objvalue = update(M, r, c, basics)
        print(dict(zip(labels,solution_vector)), objvalue, "\n")        
        # Optimality
        optimal = optimality_test(net, direction)
        # 
        if optimal:
            break
        # Feasibility
        entry, leaving, basics = feasibility_test(M, r, net, basics, c, direction)
        variable_leaving = positions[leaving]   # only for tracking position variable ("x1, x2, ... , etc")
        positions[leaving] = entry
        
        iteration += 1
        print(f'Iteration: {iteration}')
        print(f"Leaving: {labels[variable_leaving]},  Entering: {labels[entry]}")
        
        M, r = row_operations(M, r, entry, leaving)  # Gauss Jordan
        #
        solution_vector[positions] = rhs
        solution_vector[variable_leaving] = 0
        
        print(M, "\n")
        
    return A, rhs, objvalue, solution_vector

if __name__ == "__main__":
    # Data

    # max nvar=3
    # cj = create_array("3 2 5 0 0 0" )
    # A = create_array("1 2 1 1 0 0; 3 0 2 0 1 0; 1 4 0 0 0 1")
    # rhs = create_array("430 460 420")

    # cj = create_array("240 104 60 19 0 0" )
    # A = create_array("20 9 6 1 1 0; 10 4 2 1 0 1")
    # rhs = create_array("20 10")

    # cj = create_array("4 3 6 0 0 0" )
    # A = create_array("2 3 2 1 0 0; 4 0 3 0 1 0; 2 5 0 0 0 1")
    # rhs = create_array("440 470 430")

    # cj = create_array("5 4 0 0 0 0" )
    # A = create_array("6 4  1 0 0 0;1 2 0 1 0 0;-1 1 0 0 1 0; 0 1 0 0 0 1")
    # rhs = create_array("24 6 1 2")

    # A = create_array("1 4 1 0 0; 3 1 0 1 0; 1 1 0 0 1")
    # cj = create_array("2 5 0 0 0" )
    # rhs = create_array("24 21   9")

    # cj = create_array("10 5 7 0 0 0" )
    # A = create_array("1 1 1 1 0 0; 3 1 2 0 1 0; 1 0 0 0 0 1")
    # rhs = create_array("800 1000 150")

    # nvar max example 2.9-3 gupta
    # cj = create_array("2 1 0 0 0 0")
    # A = create_array("1	2 1 0 0 0;1 1 0 1 0 0; 1 -1 0 0 1 0; 1 -2 0	0 0 1")
    # rhs = create_array("10 6 2 1")

    # nvar=2 
    # cj = create_array("12 20 0 0 1000 1000" )
    # A = create_array("6 8 -1 0 1 0; 7 12 0 -1 0 1")
    # rhs = create_array("100 120")

    # cj = create_array("3 -1  0  0 0 -1000" )
    # A = create_array("2 1 1 0 0 0; 1 3 0 -1 0 1; 0 1 0 0 1 0")
    # rhs = create_array("2 3 4")

    # nvar=2 max example 2.9-2
    cj = create_array("2 3  0 -1000 0 0 0 0" )
    A = create_array("\
    1 1 1 0 0 0 0 0;\
    0 1 0 1 0 0 0 -1;\
    0 1 0 0 1 0 0 0;\
    -1 1 0 0 0 1 0 0;\
    1 0 0 0 0 0 1 0")
    rhs = create_array("30 3 12 0 20")

    # nvar=2 maximization
    # cj = create_array("3 4 0 0 0 0 -1000 -1000")
    # A = create_array("5 4 1 0 0  0 0 0;3 5 0 1 0 0 0 0;5 4 0 0 -1 0 1 0;8 4 0 0 0 -1 0 1")
    # rhs = create_array("200 150 100 80")

    # cj = create_array("2  1 0.25 0 0 0 -1000")
    # A = create_array("4 6 3 1 0 0 0; 3 -6 -4 0 1 0 0; 2 3 -5 0 0 -1 1")
    # rhs = create_array("8 1 4")

    # Max nvars=3
    # cj = create_array("5 -4 3 0 0 0 -1000")
    # A = create_array("2 1 -6 0 0 0 1; 6 5 10 0 1 0 0; 8 -3 6 0 0 1 0")
    # rhs = create_array("20 76 50")


    # #max nvar=3
    # A = create_array("1 1 1 1 0 0 0 0 0 0; -1 3 0 0 1 0 0 0 0 0; 1 -3 0 0 0 1 0 0 0 0; 0 2 -1 0 0 0 1 0 0 0; 0 -2 1 0 0 0 0 1 0 0; 0 1 0 0 0 0 0 0 1 0; 1 0 0 0 0 0 0 0 0 1")
    # rhs = create_array("800 0 0 0 0 1000 150")
    # cj = create_array("10 5 7 0 0 0 0 0 0 0")



    #max nvar=2
    # A = create_array("4 2 1 0 0; 2.5 0.6 0 1 0; 1 2 0 0 1")
    # rhs = create_array("2000 1500 600")
    # cj = create_array("25 45 0 0 0")
    
    body, solution, zvalue, vector = simplex(A, cj, rhs, nvars=2, direction=1)

