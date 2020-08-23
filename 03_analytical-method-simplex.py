#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
import re
from itertools import combinations
import numpy as np
import pandas as pd


class Constraint:
    def __init__(self, expression):
        self.expression = expression

    def vector(self):
        'Construct vector from string constraint'
        vars, rhs =  re.split("[<>]*=", self.expression)
        return [float(element) for element in vars.split()], float(rhs)

    def slack(self):
        sense = re.findall("[<>]*=", self.expression)[0]
        if sense == '<=':
            return 1
        elif sense == '>=':
            return -1
        else:
            return 0

    def row(self, element="body"):
        if element == "body":
            return self.vector()[0]
        elif element == "slack":
            return self.slack()
        else:
            return  self.vector()[1]
        

    
class ModelLP:
    def __init__(self, objective=None, name='Model'):
        self._name = name
        self._bodycoeff = [] 
        self._rhs = []
        self._slacks = []
        self._num_const = 0
        self.body_matrix = None
        self.slack_matrix = None
        self.rhs_matrix = None
        self._objective = objective

    @property
    def name(self):
        print(self._name)

    def setObjective(self, obj):
        self._objective = np.array([float(value) for value in obj.split()])

        
    def addConstr(self, expression):
        self._num_const += 1
        c = Constraint(expression)
        self._bodycoeff.append(c.row("body"))
        self._slacks.append(c.row("slack"))
        self._rhs.append(c.row("rhs"))

    def get_rhs(self):
        if not self._rhs:
            print("There is no Constraints in the model")
        self.rhs_matrix = np.array(self._rhs).reshape(self._num_const,1)
        return self.rhs_matrix

    @property
    def num_slacks(self):
        return len(self._slacks)
    
    def set_matrix_form(self):
        body = np.array(self._bodycoeff)
        indx = np.where(np.array(self._slacks) == 0)[0]  # zero-value slacks variables
        slacks = np.diag(self._slacks)
        if indx.size > 0:  # remove slacks 0 from slack matrix
            slacks = np.delete(slacks, indx, axis=1)    
        self.body_matrix, self.slack_matrix =  body, slacks

    def standard_form(self):
        self.set_matrix_form()
        matrix = np.hstack([self.body_matrix, self.slack_matrix])
        print(matrix)
        

    def get_submatrices(self):
        'Matrix generator from original coefficient matrix'
        self.set_matrix_form()
        matrix = np.hstack([self.body_matrix, self.slack_matrix])
        numconstr, numvars = matrix.shape
        assert numconstr == self._num_const, "Number of constraints are different"
        return (matrix[:,[*columns]] for columns in combinations(range(numvars), numconstr))

    def invmats(self):
        mats = self.get_submatrices()
        return (np.linalg.inv(m) for m in mats)

    def get_basics(self):
        'Solve equations'
        b = self.get_rhs()
        inverses = self.invmats()
        return (Ainv.dot(b) for Ainv in inverses)

    def zvalues(self):
        objective = np.array(self._objective)
        objective = np.concatenate([objective, np.zeros_like(self._slacks)])
        basics = self.get_basics()  # generator
        positions = combinations(range(objective.size), self._num_const)
        zvalues = []
        solutions = []
        for basic,position in zip(basics, positions):
            solution = np.zeros((objective.size,1))
            solution[list(position)] = basic
            solutions.append(solution.flatten())
            z = objective.dot(solution)
            zvalues.append(z[0])
        z = np.array(zvalues)
        x = np.array(solutions)
        m = np.hstack([x, z.reshape(len(x), 1)])
        matrix_df = pd.DataFrame(m)
        print(matrix_df)
        print(np.hstack([x, z.reshape(len(x), 1)]))
        
        
    def solve(self):
        A = np.hstack([self.body_matrix, self.slack_matrix])
        print(list(self.get_basics(A)))
        
    def coefficient_matrix(self):
        self.set_matrix_form()
        print(np.hstack([self.body_matrix, self.slack_matrix]))
        
    def show_constr(self):
        print(self._bodycoeff)

    def num_constr(self):
        return print(f"Number Constraints: {self._num_const}")


        
        
class LP:
    def __init__(self, coefficients, rhs, objcoef):
        self.A = coefficients
        self.b = rhs
        self.c = objcoef
        
        
    def constr_num(self):
        return A.shape[0]

    def var_num(self):
        return A.shape[1]

    def submatrices(self):
        'Matrix generator from original coefficient matrix'
        return (A[:,[i,j]] for i,j in combinations(range(self.var_num()) , self.constr_num()))

    def invmats(self):
        mats = self.submatrices()
        return (np.linalg.inv(matrix) for matrix in mats)

    def basics(self):
        'Solve equations'
        inverses = self.invmats()
        return (Ainv.dot(b) for Ainv in inverses)



if __name__ == "__main__":
    model = ModelLP("Example 2.15-2")
    model.addConstr("5 4 2 1 = 100")
    model.addConstr("2 3 8 1  = 75")
    model.setObjective("12 8 14 10")
    model.name
    model.standard_form()
    model.zvalues()


