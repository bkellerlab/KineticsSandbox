import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike





class Potential:
    def __init__(self,*args):
        self.coefficients=args

    def f(self,x: ArrayLike):
        #to calculate the potential function,which will be overridden by subclasses
        raise NotImplementedError("subclass must  be implemented abstract method")

    def analytical_derivative(self, x: ArrayLike):
        raise NotImplementedError("subclass must  be implemented abstract method")


    def numerical_derivative(self, x: ArrayLike, h=0.0001):
        return (self.f(x+h)-self.f(x-h))/(2*h)


    def plot_function(self,x_values)-> None:


        y_values = self.f(x_values)
        dy_analytical = self.analytical_derivative(x_values)
        dy_numerical = self.numerical_derivative(x_values)

        #plotting subplots
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Adjust the figsize as needed

        # Plot the original function on the first subplot
        axes[0].plot(x_values, y_values, color='blue', label="f(x)", marker=".", markerfacecolor="k", markersize=4)
        axes[0].set_title('Function')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        axes[0].legend()

        # Plot the analytical derivative on the second subplot
        axes[1].plot(x_values, dy_analytical, color='red', label="f'(x) - analytical", marker=".", markerfacecolor="k",
                     markersize=4)
        axes[1].set_title('Analytical Derivative')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel("f'(x)")
        axes[1].legend()

        # Plot the numerical derivative on the third subplot
        axes[2].plot(x_values, np.round(dy_numerical, 8), color='green', label="f'(x) - numerical", marker=".", markerfacecolor="k",
                     markersize=4)
        axes[2].set_title('Numerical Derivative')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel("f'(x)")
        axes[2].legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(f"{self.__class__.__name__} and Derivatives Subplots fig.pdf")
        # Display the plot
        plt.show()

class Linear_Potential(Potential):
     def f(self, x: ArrayLike):
        m, c = self.coefficients
        return m * x + c

     def analytical_derivative(self, x):
        m, _ = self.coefficients
        return np.full_like(x, m)

class Quadratic_Potential(Potential):
     def f(self, x: ArrayLike):
         a, b, c = self.coefficients
         return a * x ** 2 + b * x + c

     def analytical_derivative(self, x):
          a, b, _ = self.coefficients
          return a * 2 * x + b

class DoubleWell_Potential(Potential):
     def f(self, x: ArrayLike):
          a, b, c = self.coefficients
          return a * x ** 4 - b * x ** 2 + c

     def analytical_derivative(self, x):
         a, b, _ = self.coefficients
         return a * 4 * x ** 3 - b * 2 * x