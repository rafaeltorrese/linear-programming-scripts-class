# -*- coding: utf-8 -*-
#/usr/bin/env python3
# _*_ coding: utf8 _*_

import numpy as np
np.set_printoptions(precision=4, suppress=True)


def simplex(M, coefobj, b, nvars, sense=1):
    """
    Dual Simplex Method

    Parameters
    -------------
    M: ndarray
        Matrix with technological coefficients

    coefobj: array
        Coefficients of objective function

    b: 1D-array
        Right-hand side values

    nvars: int
        Number of decision variables. Don't take into account either slack variables of artificial variables.

    sense: {+1, -1}
        Sense of objective function, +1 for maximization problems, -1 for minimization problems

    """
    M = M.astype(float)
    positions_basics = np.where(M[:, nvars: ] == 1)[1] + nvars  # get positions for initial  basic solutions. Only columns with number one
    solution_vector = np.zeros(coefobj.size)
    cb = coefobj[positions_basics].astype(np.float)
    iteration = 0



    while True:
    # for _ in range(5):
        # update

        zj = cb.dot(M)
        profit = coefobj - zj
        zvalue = cb.dot(b)  # solutions is the rhs (right-hand side vector)

        # optimality_test
        # print(f"solution: {solution_vector}")
        if np.all(sense *  profit <= 0):
            print("\nOptimal Solution Found")
            break

        # pivot
        ratios = np.full_like(cb, np.infty)
        entering_position = np.argmax(sense * profit)  # position of maximum value

        entering_column = M[:, entering_position].astype(float)
        positive_values = np.where(entering_column > 0)[0]
        ratios[positive_values] = b[positive_values] / entering_column[positive_values]

        leaving_position = ratios.argmin()
        cb[leaving_position] = coefobj[entering_position]  # update basic variable coefficients

        # Gauss-Jordan
        num_rows = M.shape[0]
        pivot_element = M[leaving_position, entering_position]
        if pivot_element != 1:
            M[leaving_position] = M[leaving_position] / pivot_element
            b[leaving_position] = b[leaving_position] / pivot_element
        for i in range(num_rows):
            if i == leaving_position:
                continue
            factor = -M[i, entering_position]
            M[i, :] += factor * A[leaving_position]  #  M[i, :] = -M[i, entering_position] * A[leaving_position] + M[i, :]
            b[i] += factor * b[leaving_position]  # update rhs row

        #
        leaving_variable = positions_basics[leaving_position]  # getting leaving_variable from position_basics
        positions_basics[leaving_position] = entering_position
        solution_vector[leaving_variable] = 0
        solution_vector[positions_basics] = b
        iteration += 1

        #
        print(f"\nIteration: {iteration}")
        print(f"Leaving: Variable {leaving_variable + 1},  Entering: Variable {entering_position + 1}")
        print(f"{M}")
    print(f"{solution_vector} {zvalue}")



def create_array(data):
    "Create a matrix as numpy array from string data"
    if ";" in data:
        return np.array([row.split() for row in data.split(";")], dtype=np.float)
    return np.array(data.strip().split(), dtype=np.float)


if __name__ == "__main__":




    #minimize nvars=2
    # cj = create_array("12 20 0 0 1000 1000" )
    # A = create_array("6 8 -1 0 1 0; 7 12 0 -1 0 1")
    # rhs = create_array("100 120")


    # cj = create_array("3 4 0 0 0 0 -1000 -1000")
    # A = create_array("5 4 1 0 0  0 0 0;3 5 0 1 0 0 0 0;5 4 0 0 -1 0 1 0;8 4 0 0 0 -1 0 1")
    # rhs = create_array("200 150 100 80")

    # max nvars=3
    # cj = create_array("5 -4 3  0 0 -1000")
    # A = create_array("2 1 -6 0 0 1; 6 5 10  1 0 0; 8 -3 6  0 1 0")
    # rhs = create_array("20 76 50")

    # cj = create_array("2 3  0 -1000 0 0 0 0" )
    # A = create_array("\
    # 1 1 1 0 0 0 0 0;\
    # 0 1 0 1 0 0 0 -1;\
    # 0 1 0 0 1 0 0 0;\
    # -1 1 0 0 0 1 0 0;\
    # 1 0 0 0 0 0 1 0")
    # rhs = create_array("30 3 12 0 20")

    # # max nvar=3
    # cj = create_array("3 2 5 0 0 0" )
    # A = create_array("1 2 1 1 0 0; 3 0 2 0 1 0; 1 4 0 0 0 1")
    # rhs = create_array("430 460 420")

    #max nvar=3
    A = create_array("1 1 1 1 0 0 0 0 0 0; -1 3 0 0 1 0 0 0 0 0; 1 -3 0 0 0 1 0 0 0 0; 0 2 -1 0 0 0 1 0 0 0; 0 -2 1 0 0 0 0 1 0 0; 0 1 0 0 0 0 0 0 1 0; 1 0 0 0 0 0 0 0 0 1")
    rhs = create_array("800 0 0 0 0 1000 150")
    cj = create_array("10 5 7 0 0 0 0 0 0 0")

    simplex(A, cj, rhs, nvars=3, sense=1)
