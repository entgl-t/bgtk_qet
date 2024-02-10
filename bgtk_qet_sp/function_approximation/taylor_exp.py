import torch
from torch.autograd import grad
import  math
from sympy import symbols, diff, factorial


class TaylorSeries():
    def __init__(self, function, order, variable=symbols('x'), center=0.0):
        self.center = torch.tensor([float(center)], requires_grad= True)  # center
        self.f = function
        self.var = variable
        self.order = order
        self.coefficients = []
        self.find_coefficients()

    def autodiff_find_coefficients(self):

        self.coefficients.append(round(float(self.f(self.center).item()), 5))
        derivative = self.f(self.center)
        for i in range(1, self.order + 1):

            derivative = grad(derivative, self.center,create_graph=True)[0]
            self.coefficients.append(round(float(derivative.item()) / math.factorial(i), 5))

            print('Derivative of order ' + str(i) + ': ', derivative.item())


    def find_coefficients(self):

        for i in range(self.order + 1):
            # Calculate the i-th derivative of the function at the given point
            derivative = diff(self.f, self.var, i)

            # Evaluate the derivative at the given point
            derivative_value = derivative.subs(self.var, self.center)

            # Calculate the i-th coefficient using the derivative and factorial
            self.coefficients.append( round(float(derivative_value / factorial(i)),10))



    def print_equation(self):
        print('\nTaylor expansion:\n')
        eqn_string = ""
        for i in range(self.order + 1):
            if self.coefficients[i] != 0:
                eqn_string += str(self.coefficients[i]) + (
                    "(x-{})^{}".format(self.center, i) if i > 0 else "") + " + "
        eqn_string = eqn_string[:-3] if eqn_string.endswith(" + ") else eqn_string
        print(eqn_string)

    def print_coefficients(self):

        print('\nCoefficients of the polynomial:\n')
        print(self.coefficients)

    def get_coefficients(self):
        """
            Returns the coefficients of the taylor series
        """
        return self.coefficients

