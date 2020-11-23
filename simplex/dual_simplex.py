#/usr/bin/env python3
# _*_ coding: utf8 _*_

import numpy as np
np.set_printoptions(precision=4, suppress=True)

def update():
    zj = cb.dot(A)
    profit = cj - zj
    zvalue = cb.dot(r)
    print(zvalue)

def optimality_test():
    if np.all(profit <= 0) and np.all(rhs >= 0):
        print("Optimal Solution Found")
    else:
        print("Method Fails")


def feasibility_test():
    leaving = np.argmin(profit)
    
def dual_simplex(M, coefobj, b, nvars):
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

    """
    positions_basics = np.where(M[:, nvars: ] == 1)[1] + nvars  # get positions for initial  basic solutions. Only columns with number one
    solution_vector = np.zeros(coefobj.size)
    cb = coefobj[positions_basics].astype(np.float)
    iteration = 0

    while True:
        # update
        zj = cb.dot(M)
        profit = coefobj - zj
        zvalue = cb.dot(b)  # solutions is the rhs (right-hand side vector)

        # optimality_test
        if np.all(profit <= 0) and np.all(b >= 0):
            print("Optimal Solution Found")
            print(solution_vector, zvalue)
            break
        # pivot
        leaving_position = np.argmin(b)
        leaving_row = np.where(A[leaving_position] >= 0, 1e-20, A[leaving_position])  # if there are zero or positive values
        ratios = profit / leaving_row
        entering_position = np.where(leaving_row <= 0, ratios, np.infty).argmin()
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
            M[i, :] += factor * A[leaving_position]
            b[i] += factor * b[leaving_position]  # update rhs row
        leaving_variable = positions_basics[leaving_position]  # getting leaving_variable from position_basics
        positions_basics[leaving_position] = entering_position
        solution_vector[leaving_variable] = 0
        solution_vector[positions_basics] = b
        iteration += 1
        print(f"Iteration: {iteration}")
        print(f"{M}")




        
if __name__ == "__main__":
    # cj = np.array([-2,-2,-4,0,0,0], dtype=np.float)
    # A = np.array([[-2,-3,-5,1,0,0], [3,1,7,0,1,0 ], [1,4,6,0,0,1] ], dtype=np.float)
    # rhs = np.array([-2 , 3 , 5], dtype=np.float)

    cj = np.array([-3,-2,0,0,0,0], dtype=np.float)
    A = np.array([[-1,-1,1,0,0,0], [1,1,0,0,1,0 ], [-1,-2,0,0,1,0], [0,1,0,0,0,1] ], dtype=np.float)
    rhs = np.array([-1, 7 , -10, 3], dtype=np.float)


    dual_simplex(A, cj, rhs, nvars=2)
