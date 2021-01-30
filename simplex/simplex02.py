# -*- coding: utf-8 -*-
#/usr/bin/env python3
# _*_ coding: utf8 _*_

import re
import numpy as np
np.set_printoptions(precision=4, suppress=True)


# def simplex(M, zfunction, b, nvars, sense=1):
def simplex(M, zfunction, b, nvars):
    """
    Simplex Method

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
    direction, coefobj = zfunction
    if direction == "max":
        sense = 1
    else:
        sense = -1

    M = M.astype(float)
    positions_basics = np.where(M[:, nvars: ] == 1)[1] + nvars  # get positions for initial  basic solutions. Only columns with number one
    solution_vector = np.zeros_like(coefobj)
    iteration = 0
    cb = coefobj[positions_basics].astype(np.float)
    zj = cb.dot(M)
    profit = coefobj - zj
    non_optimal = np.any(sense * profit > 0)

    while non_optimal:
        iteration += 1
        entering_variable = np.argmax(sense * profit)  # position of maximum value
        entering_column = M[:, entering_variable]

        # positive_values = np.where(entering_column > 0)[0]
        positive_values = entering_column > 0
        print(f"Positive Values {positive_values}")
        print(f"Column {entering_column}")
        print(f"Right-hand side {b}")

        ratios = np.full(cb.size, np.inf)  # size of Cb with infinite expressions 
        ratios[positive_values] = b[positive_values] / entering_column[positive_values]
        print(f"ratios {ratios}")

        leaving_row_index = ratios.argmin()

        cb[leaving_row_index] = coefobj[entering_variable]  # update basic variable coefficients

        leaving_variable = positions_basics[leaving_row_index]  # getting leaving_variable from position_basics

        # Gauss-Jordan
        print(f"\nIteration: {iteration}")
        print(f"Leaving: Variable {leaving_variable + 1},  Entering: Variable {entering_variable + 1}")

        pivot_element = M[leaving_row_index, entering_variable]
        if pivot_element != 1:
            M[leaving_row_index] = M[leaving_row_index] / pivot_element
            b[leaving_row_index] = b[leaving_row_index] / pivot_element

        rows = range(A.shape[0])
        for i in rows:
            if i == leaving_row_index:
                continue
            factor = -M[i, entering_variable]
            M[i, :] += factor * M[leaving_row_index]  #  M[i, :] = -M[i, entering_variable] * M[leaving_row_index] + M[i, :]
            b[i] += factor * b[leaving_row_index]  # update rhs row
        #
        # update values
        zj = cb.dot(M)
        profit = coefobj - zj
        zvalue = cb.dot(b)  # solutions is the rhs (right-hand side vector)
        #
        positions_basics[leaving_row_index] = entering_variable  # update positions of basis
        solution_vector[leaving_variable] = 0
        solution_vector[positions_basics] = b
        #
        non_optimal = np.any(sense * profit > 0)
        #
        print(f"{M}.\n zvalue: {zvalue}")
        #
    print(f"{solution_vector} {zvalue}")




def objective(expression):
    string_elements = expression.strip()
    sense = re.search("[Mm](ax|in)", string_elements)
    return (sense.group(), np.array(string_elements.split()[1:], dtype=np.float))

def create_array(data):
    "Create a matrix as numpy array from string data"
    if ";" in data:
        return np.array([row.split() for row in data.split(";")], dtype=np.float)
    return np.array(data.strip().split(), dtype=np.float64)



if __name__ == "__main__":
    #minimize nvars=2
    # cj = create_array("12 20 0 0 1000 1000" )
    # A = create_array("6 8 -1 0 1 0; 7 12 0 -1 0 1")
    # rhs = create_array("100 120")


    # cj = create_array("max 3 4 0 0 0 0 -1000 -1000")
    # A = create_array("5 4 1 0 0  0 0 0;3 5 0 1 0 0 0 0;5 4 0 0 -1 0 1 0;8 4 0 0 0 -1 0 1")
    # rhs = create_array("200 150 100 80")

    # max nvars=3
    cj = objective("max 5 -4 3  0 0 -1000")
    A = create_array("2 1 -6 0 0 1; 6 5 10  1 0 0; 8 -3 6  0 1 0")
    rhs = create_array("20 76 50")

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

    # max nvar=3
    # A = create_array("1 1 1 1 0 0 0 0 0 0; -1 3 0 0 1 0 0 0 0 0; 1 -3 0 0 0 1 0 0 0 0; 0 2 -1 0 0 0 1 0 0 0; 0 -2 1 0 0 0 0 1 0 0; 0 1 0 0 0 0 0 0 1 0; 1 0 0 0 0 0 0 0 0 1")
    # rhs = create_array("800 0 0 0 0 1000 150")
    # cj = objective("max 10 5 7 0 0 0 0 0 0 0")

    # simplex(A, c2, rhs, nvars=3, sense=1)
    simplex(A, cj, rhs, nvars=3)