import numpy as np
from abc import ABC, abstractmethod
import numpy as np
#import scipy.constants as const
from scipy import integrate
#from scipy.optimize import minimize
import matplotlib.pyplot as plt





class Double_Well_Potential:

    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional Double well potential based on the given parameters.
        parameters:
          - param (list): a list of parameters representing:
          - param[0]: a (float) - controlling steepness of the wells
          - param[1]: b (float) - controlling  width and height of barrier between the wells
          - param[2]: c (float) - controlling the vertical shift of potential

        """

        # assign parameters
        self.a = param[0]
        self.b = param[1]
        self.c = param[2]

    def potential(self, x):
        """

        calculate the Double Well potential energy V(x),
        The function is given by:
        V(x) = a * x^^4 - b * x^2 + c
        parameters:
                x:position

        Returns:Double well potential for all x
        """

        return self.a * x ** 4 - self.b * x ** 2 + self.c

    def force(self, x):
        """
        Calculate the force F(x) analytically for Double Well  potential
        The force is given by:
        F(x) = - dV(x) / dx
             = - (4 * a * x ^ 3 - 2 * b * x)
        parameters:
             x: position
        Returns:numpy array: The value of the force at the given position x

        """

        return -1 * (self.a * 4 * x ** 3 - self.b * 2 * x)

    def hessian(self, x):
        """
        Calculate the Hessian matrx H(x) analytically for the 1-dimensional quadratic potential.
        Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.

        The Hessian is given by:
        H(x) = d^2 V(x) / dx^2
             = 12 * x **2 * self.a - 2 * self.b



        Parameters:
             - x (float): position


        """

        return 12 * x ** 2 * self.a - 2 * self.b

    def riemann(self, a, b, n):
        """
        calculate the Riemann integral of the potential energy function over the interval [a,b)
        using riemann central method
        parameters:
                    a(float): lower limit of integration
                    b(float):upper limit of integration
                    h(float):step size or width  of rectangle
                    s(float):midpoint of interval
        return:
               area under the curve
        """
        h = (b - a) / n
        area = 0
        for i in range(n):
            s = a + (i + 0.5) * h
            area += self.potential(s) * h

        return area

double_well = Double_Well_Potential([1, 2, -1])
n_values = np.arange(7,101,1)
a= -1.5
b= 1.5

riemann_values =[double_well.riemann(a,b,n)for n in n_values]
result = integrate.quad(double_well.potential,a,b)




plt.plot(n_values, riemann_values,label="Riemann integral", color="orange")
plt.hlines(result[0], xmin=n_values[0], xmax=n_values[-1],label="scipy integral")

plt.xlabel("n")
plt.ylabel(" Integral for  x^4-2x^2-1")
plt.legend()
plt.title("Riemann integral and scipy integral for different n")
plt.ylim(-5,-4)
plt.grid()

plt.axhline(y=-4.465, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(x=7, color='gray', linestyle='--', linewidth=0.5)
plt.show()


difference = np.abs(np.array(riemann_values) - result[0])
plt.plot(n_values, difference, color="green",linestyle="--")
plt.xlabel("n")
plt.ylabel(" Absolute Error")
plt.title("Absolute Error for Different n")
plt.grid()
plt.show()


double_well = Double_Well_Potential([1, 2, -1])
n_values = np.arange(1,101,1)
a= 0
b= 1.5

riemann_values =[double_well.riemann(a,b,n)for n in n_values]
result = integrate.quad(double_well.potential,a,b)




plt.plot(n_values, riemann_values,label="Riemann integral", color="orange")
plt.hlines(result[0], xmin=n_values[0], xmax=n_values[-1],label="scipy integral")

plt.xlabel("n")
plt.ylabel(" Integral for  x^4-2x^2-1")
plt.legend()
plt.title("Riemann integral and scipy integral for different n")
plt.ylim()
plt.grid()

#plt.axhline(y, color='grey', linestyle='--', linewidth=0.5)
#plt.axvline(x, color='grey', linestyle='--', linewidth=0.5)
plt.show()


difference = np.abs(np.array(riemann_values) - result[0])
plt.plot(n_values, difference, color="green",linestyle="--")
plt.xlabel("n")
plt.ylabel(" Absolute Error")
plt.title("Absolute Error for Different n")
plt.grid()
plt.show()