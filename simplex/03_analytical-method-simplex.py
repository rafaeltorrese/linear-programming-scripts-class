#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import re
from itertools import combinations
import numpy as np

np.set_printoptions(precision=4, suppress=True)



class Constraint:
    def __init__(self, expression):
        self._expression = expression

    def vector(self):
        'Construct vector from string constraint'
        vars, rhs =  re.split("[<>]?=", self._expression)
        return [float(element) for element in vars.split()], float(rhs)

    def slack(self):
        sense = re.search("[<>]?=", self._expression)[0]
        if sense == '<=':
            return 1
        elif sense == '>=':
            return -1
        else:
            return 0

    def artificial(self):
        slack = self.slack()
        if slack <= 0:
            return 1
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
        self._artificials_list = []
        self._num_const = 0
        self.body_matrix = None
        self._slack_matrix = None
        self._artificial_matrix = None
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


    def _standard_objective(self):
        "Returns coefficients from objective function, including slack or artificial variables"
        objective = self.getObjective
        if (self.get_slack_matrix.size > 0):  # zero-values slack variables
            objective = np.concatenate([objective, np.zeros_like(self.get_slack_matrix[1])])  # create objective function with slacks variables
        if (self._artificial_matrix.size > 0) and (self._sense == "max"):  # zero-values slack variables
            objective = np.concatenate([objective, np.full(self.get_artificial_matrix.shape[1], -100000)])
        else:
            objective = np.concatenate([objective, np.full(self.get_artificial_matrix.shape[1], 100000)])
        return objective


    def load_instance(self, string_expressions):
        'Strings with an instance separated by semi-colon'
        expression_list = string_expressions.split(";")
        for expression in expression_list:
            self.addExpr(expression)

    @property
    def getObjective(self):
        "Returns only coefficients from objective function (don't include slack or artificial variables)"
        return self._objective


    def addExpr(self, expression):
        # self._num_const += 1
        is_objective = re.match("^m(ax*|in*)", expression.lstrip())  # match min or max
        if is_objective:
            self._setObjective(expression.strip())
        else:
            constr = Constraint(expression.strip())  # Constraint Class
            self._bodycoeff.append(constr.component("body"))
            self._slacks.append(constr.component("slack")) # add -1,1 or 0 to list
            self._artificials_list.append(constr.artificial()) # add 1 or 0 to list of artificial variables
            self._rhs.append(constr.component("rhs"))  #  add rhs to _rhs list
            self._num_const += 1


    def get_rhs(self):
        if not self._rhs:
            print("There is no Constraints in the model")
        # self.rhs_matrix = np.array(self._rhs).reshape(self._num_const,1)
        self.rhs_matrix = np.array(self._rhs)
        return self.rhs_matrix


    @property
    def get_slack_matrix(self):
        slacks_array = np.array(self._slacks)
        indx = np.where(slacks_array == 0)[0]  # where there are zero-value slacks variables
        self._slack_matrix = np.diag(self._slacks)
        if indx.size > 0:  # if there are slack variables with zero-values
            self._slack_matrix = np.delete(self._slack_matrix, indx, axis=1)
            return self._slack_matrix
        return self._slack_matrix


    @property
    def get_artificial_matrix(self):
        artificials_array = np.array(self._artificials_list)
        indx_zero_artificials = np.where(artificials_array == 0)[0]
        self._artificial_matrix = np.diag(self._artificials_list)
        if indx_zero_artificials.size > 0:
            self._artificial_matrix = np.delete(self._artificial_matrix, indx_zero_artificials, axis=1)
            return self._artificial_matrix
        return self._artificial_matrix


    @property
    def get_body_matrix(self):
        self.body_matrix = np.array(self._bodycoeff)
        return self.body_matrix



    def set_matrix_form(self):
        if self.get_slack_matrix.size > 0:
            matrix = np.hstack((self.get_body_matrix, self.get_slack_matrix))
        if self.get_artificial_matrix.size > 0:
            matrix = np.hstack((matrix, self.get_artificial_matrix))
        return matrix


    @property
    def show_standard_form(self):
        print(self.set_matrix_form())


    @property
    def num_slacks(self):
        return sum(np.array(self._slacks).abs())

    @property
    def num_arficials(self):
        return np.sum(np.array(self._artificials_list) > 0)


    def show_constr(self):
        print(self._bodycoeff)

    @property
    def num_constr(self):
        return print(f"Number Constraints: {self._num_const}")


    def _optimization_routine(self, M, Cb, cj, b, direction, labels, solution_vector, position_basics, iteration):
        while True:
            #update
            Zj = Cb.dot(M)
            profit = cj - Zj
            Z = Cb.dot(b)
            if np.all(direction * profit <= 0):
                print(f"\nOptimal Solution found \n Matrix:\n{M}\n\n Zvalue: {Z}\n")
                print(dict(zip(labels, solution_vector)))
                break

            # entering and leaving variables
            ratios = np.full(Cb.size, np.inf)  # size of Cb with infinite expressions 
            position_of_entering_variable = np.argmax(direction * profit)  # position of maximum value
            entering_column = M[: , position_of_entering_variable]
            positive_column_values = entering_column > 0
            ratios[positive_column_values] = b[positive_column_values] / entering_column[positive_column_values]
            position_of_leaving_variable = ratios.argmin()
            Cb[position_of_leaving_variable] = cj[position_of_entering_variable]  # update basic variable coefficients

            #Gauss-Jordan
            number_of_rows = self._num_const
            pivot_element = M[position_of_leaving_variable, position_of_entering_variable]
            if pivot_element != 1:
                M[position_of_leaving_variable] /= pivot_element
                b[position_of_leaving_variable] /=  pivot_element
            for i in range(number_of_rows):
                if i == position_of_leaving_variable: continue
                target_zero_element = -M[i, position_of_entering_variable]
                M[i] += target_zero_element * M[position_of_leaving_variable]
                b[i] += target_zero_element * b[position_of_leaving_variable]

            leaving_variable = position_basics[position_of_leaving_variable]
            position_basics[position_of_leaving_variable] = position_of_entering_variable

            solution_vector[position_basics] = b
            solution_vector[leaving_variable] = 0

            iteration += 1

            print(f"Iteration: {iteration}")
            # print(f"Entering Variable: {position_of_entering_variable + 1}, Leaving Variable: {leaving_variable + 1}")
            # print(labels[position_of_entering_variable])
            print(f"Entering Variable: {labels[position_of_entering_variable]}, Leaving Variable: {labels[leaving_variable]}")
            print(M)

    def simplex(self, twophase=False):
        if self._sense == "max":
            direction = 1
        else:
            direction = -1

        b = self.get_rhs()
        M = self.set_matrix_form()

        slacks = self._slack_matrix
        num_vars = self._objective.size
        num_slacks = slacks.shape[1]  # number of columns
        # num_slacks = self.num_slacks
        num_artificials = self.get_artificial_matrix.shape[1]  # number of columns
        label_xvars = ["".join(["x", str(x + 1)]) for x in range(num_vars)]
        label_slacks = ["".join(["s", str(s + 1)]) for s in range(num_slacks)]
        label_artificials = ["".join(["A", str(a + 1)]) for a in range(num_artificials)]

        labels = label_xvars + label_slacks + label_artificials
        print(labels)
        position_basics =np.where(M[ :,  num_vars: ] == 1)[1] + num_vars
        solution_vector = np.zeros(M.shape[1])  # number of columns

        iteration = 0
        if twophase:
            print("="*10 + "Starting Phase I" + "="*10)
            cj = np.where((self._standard_objective() == -100000) | (self._standard_objective() == 100000), 1, 0)
            Cb = cj[position_basics]  # coefficients of basics
            self._optimization_routine(M, Cb, cj, b, -1, labels, solution_vector, position_basics, iteration)
            
            print("\nStarting Phase II")
            cj = self._standard_objective()
            Cb = cj[position_basics]  # coefficients of basics
            self._optimization_routine(M, Cb, cj, b, direction, labels, solution_vector, position_basics, iteration)
        else:
            cj = self._standard_objective()
            Cb = cj[position_basics]  # coefficients of basics
            self._optimization_routine(M, Cb, cj, b, direction, labels, solution_vector, position_basics, iteration)




class AnalyticalMethod(ModelLP):
    def __init__(self, name):
        super().__init__(name)


    def get_solutions(self):
        'Matrix generator from original coefficients matrix'
        feasibles = []
        infeasibles = []
        objective = self._objective
        if self._slack_matrix.size > 0:  # zero-values slack variables
            objective = np.concatenate([objective, np.zeros_like(self._slacks)])  # create objective function with slacks variables
        matrix = self.set_matrix_form()  # matrix ensamble ==> body + slacks
        numconstr, numvars = matrix.shape
        assert numconstr == self._num_const, "Number of constraints are different"
        b = self.get_rhs()  # get rhs vector
        for columns in combinations(range(numvars), numconstr):
            solution = np.zeros_like(objective)
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


class Simplex(AnalyticalMethod):
    def _basics(self):
        return np.zeros(super().num_slacks)


    def _objective_func(self):
        # return super().getObjective()
        return np.concatenate((super().getObjective() , self._basics()) )


    def _update(self):
        rhs = super().get_rhs()
        A = super().set_matrix_form()
        basics = self._basics()
        cj = self._objective_func()
        zj = basics.dot(A)
        net_evaluation = cj - zj
        zvalue = basics.dot(rhs)
        print(zj, net_evaluation, zvalue)


if __name__ == "__main__":
    # model = ModelLP("Example 2.16-1")
    # model.load_instance("max 3 4; 1 1 <= 450; 2 1 <= 600")
    # model.simplex()

    # model = ModelLP("Example 2.16-2")
    # model.load_instance("max 2 5; 1 4 <= 24; 3 1  <= 21; 1 1 <= 9")
    # model.simplex()

    # model = ModelLP("Example 2.17-1")
    # model.load_instance("min 12 20; 6 8 >= 100; 7 12 >= 120")

    model = ModelLP("Example 2.17-7")
    model.load_instance("max 5 -4  3;2 1 -6 = 20; 6 5 10 <= 76; 8 -3 6 <= 76")

    model.show_standard_form
    # print(model.get_slack_matrix)
    model.simplex(twophase=True)



    # model = AnalyticalMethod("Example 2.15-3")
    # model.load_instance("5 4 2 1 = 100; 2 3 8 1  = 75; max 12 8 14 10")
    # model.standard_form
    # model.best_solution()

    # model2 = AnalyticalMethod("Example 2.17-1")
    # model2.load_instance("min 12 20; 6 8 >= 100; 7 12 >= 120")
    # model2.standard_form
