#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import re
from itertools import combinations
import numpy as np
import pandas as pd


class Constraint:
    def __init__(self, expression):
        self._expression = expression

    def vector(self):
        'Construct vector from string constraint'
        vars, rhs =  re.split("[<>]?=", self._expression)
        return [float(element) for element in vars.split()], float(rhs)

    def slack(self):
        sense = re.findall("[<>]?=", self._expression)[0]
        if sense == '<=':
            return 1
        elif sense == '>=':
            return -1
        else:
            return 0

    def component(self, element="body"):
        if element == "body":
            return self.vector()[0]  # lhs (e.g. 1 2 3 )
        elif element == "rhs":
            return  self.vector()[1]  # rhs (e.g 4)
        else:
            return self.slack()  # return 1, -1, 0
        

    
class ModelLP:
    def __init__(self, name="Model", objective=None):
        self._name = name
        self._bodycoeff = [] 
        self._rhs = []
        self._slacks = []
        self._num_const = 0
        self.body_matrix = None
        self.slack_matrix = None
        self.rhs_matrix = None
        self._objective = objective
        self._sense = ""

        
    @property
    def name(self):
        print(f"Name: {self._name} \nType: {self._sense}")

    def _setObjective(self, expression):
        obj_components = expression.strip().split()  # component of an objective function (sense, coefficients)
        self._sense = obj_components[0]  # sense in first place of the list (max or min)
        coefficients = obj_components[1: ]  # only numbers (objective function coefficients)
        self._objective = np.array(coefficients, dtype=np.float)
        if self._num_const > 0:
            self._num_const -= 1
        

    def load_instance(self, string_expressions):
        'Strings with instance semi colon separated'
        expression_list = string_expressions.split(";")
        for expression in expression_list:
            self.addExpr(expression)
                  
        
    def getObjective(self):
        return self._objective
        

    def addExpr(self, expression):
        self._num_const += 1
        is_objective = re.match("^m(ax|in)", expression.lstrip())  # match min or max 
        if is_objective:
            self._setObjective(expression.strip())
        else:
            constr = Constraint(expression.strip())
            self._bodycoeff.append(constr.component("body"))
            self._slacks.append(constr.component("slack")) # add -1,1 or 0 to list _rhs
            self._rhs.append(constr.component("rhs"))  #  add rhs to _rhs list

        
    def get_rhs(self):
        if not self._rhs:
            print("There is no Constraints in the model")
        # self.rhs_matrix = np.array(self._rhs).reshape(self._num_const,1)
        self.rhs_matrix = np.array(self._rhs)
        return self.rhs_matrix

    @property
    def num_slacks(self):
        print((np.array(self._slacks) > 0).sum())
    
    def set_matrix_form(self):
        body = np.array(self._bodycoeff)
        indx = np.where(np.array(self._slacks) == 0)[0]  # where there are zero-value slacks variables
        slacks = np.diag(self._slacks)
        if indx.size > 0:  # if thera are zero-values slackas variable then remove zero-value slacks from slack matrix
            slacks = np.delete(slacks, indx, axis=1)    
        self.body_matrix, self.slack_matrix =  body, slacks  # create body matrix and slack matrix
        if self.slack_matrix.size > 0:  # if slack matrix does not exists
            return np.hstack([self.body_matrix, self.slack_matrix])  # return matrix
        return self.body_matrix

    @property
    def standard_form(self):
        print(self.set_matrix_form())
        

        
    def get_solutions(self):
        'Matrix generator from original coefficient matrix'
        feasibles = []
        infeasibles = []
        if self.slack_matrix != None:  # zero-values slack variables
            objective = np.concatenate([self._objective, np.zeros_like(self._slacks)])  # create objective function with slacks variables
        else:
            objective = self._objective
        matrix = self.set_matrix_form()  # matrix ensamble ==> body + slacks
        numconstr, numvars = matrix.shape
        assert numconstr == self._num_const, "Number of constraints are different"
        b = self.get_rhs()  # get rhs vector
        for columns in combinations(range(numvars), numconstr):
            solution = np.zeros(objective.size)
            m = matrix[:, [*columns]] # submatrix
            try:
                basic = np.linalg.solve(m, b)
                solution[[*columns]] = basic
                if np.all(basic >= 0):   # feasible solution
                    z = objective.dot(solution)
                    feasibles.append(np.concatenate([solution, [z]]))
                else:
                    infeasibles.append(solution)
            except np.linalg.LinAlgError:
                print("Singular Matrix")
        return  np.array(feasibles),  np.array(infeasibles)
        
                
    def best_solution(self):
        feasibles, infeasibles = self.get_solutions()
        zvalues = feasibles[:, -1]
        best_vector = feasibles[np.argmax(zvalues)]
        for i,element in enumerate(best_vector[:-1], 1):
            print(f"x{i}: {element: 0.4f}")
        print(f"Z = {best_vector[-1]:0.4f}")
        
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
    model = ModelLP("Example 2.15-3")
    model.load_instance("5 4 2 1 = 100; 2 3 8 1  = 75; max 12 8 14 10")
    model.best_solution()
    
    #model2 = ModelLP("Example 2.17-1")
    #model2.load_instance("min 12 20; 6 8 >= 100; 7 12 >= 120")
    #model2.standard_form()
