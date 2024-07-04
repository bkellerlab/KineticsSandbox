"""
Created on Wed July 3 13:45:56 2024

@author: arthur

This encapsulates a function to benchmark 1D potentials from the potential.D1 class

"""

# -----------------------------------------
#   I M P O R T S
# -----------------------------------------
import re
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np


def get_middle_value(values):
    return values[len(values) // 2]


def extract_potential_expression_from_docstring(cls):
    """
    Extract the potential energy function as a LaTeX expression from the docstring of the given class.

    Parameters:
        - cls: The class containing the potential function.

    Returns:
        str: The LaTeX expression of the potential energy function.
    """
    # Get the docstring of the potential method
    docstring = cls.potential.__doc__

    # Extract the mathematical expression from the docstring
    match = re.search(r"V\(x\) = (.*)", docstring)
    if not match:
        print(
            "ValueError: Could not find the potential energy function in the docstring."
        )
        return ""
    expression = match.group(1)

    # Substitute class variables with their LaTeX equivalents
    expression = expression.replace("**", "^")
    expression = expression.replace("*", " \\cdot ")
    expression = expression.replace("np.exp", "\\exp")
    expression = expression.replace("alpha", "\\alpha")

    # Convert the expression to LaTeX format
    latex_expression = f"$V(x) = {expression}$"
    return latex_expression


def extract_force_expression_from_docstring(cls):
    """
    Extract the force function as a LaTeX expression from the docstring of the given class.

    Parameters:
        - cls: The class containing the potential function.

    Returns:
        str: The LaTeX expression of the potential energy function.
    """
    # Get the docstring of the potential method
    docstring = cls.force_ana.__doc__

    # Extract the mathematical expression from the docstring
    match = re.search(r"F\(x\) = - dV\(x\) / dx\s*=\s*(.*)", docstring)
    if not match:
        print("ValueError: Could not find the force function in the docstring.")
        return ""
    expression = match.group(1)

    # Substitute class variables with their LaTeX equivalents
    expression = expression.replace("**", "^")
    expression = expression.replace("*", " \\cdot ")
    expression = expression.replace("np.exp", "\\exp")
    expression = expression.replace("alpha", "\\alpha")

    # Convert the expression to LaTeX format
    latex_expression = f"$F(x) = {expression}$"
    return latex_expression


def extract_hessian_expression_from_docstring(cls):
    """
    Extract the hessian function as a LaTeX expression from the docstring of the given class.

    Parameters:
        - cls: The class containing the potential function.

    Returns:
        str: The LaTeX expression of the potential energy function.
    """
    # Get the docstring of the potential method
    docstring = cls.hessian_ana.__doc__

    # Extract the mathematical expression from the docstring
    match = re.search(r"H\(x\) = d\^2 V\(x\) / dx\^2\s*=\s*(.*)", docstring)
    if not match:
        print("ValueError: Could not find the hessian function in the docstring.")
        return ""
    expression = match.group(1)

    # Substitute class variables with their LaTeX equivalents
    expression = expression.replace("**", "^")
    expression = expression.replace("*", " \\cdot ")
    expression = expression.replace("np.exp", "\\exp")
    expression = expression.replace("alpha", "\\alpha")

    # Convert the expression to LaTeX format
    latex_expression = f"$H(x) = {expression}$"
    return latex_expression


def benchmark_1d_potential(
    PotentialClass,
    param_ranges,
    param_names,
    x_range,
    h,
    test_V=True,
    test_F=True,
    test_H=True,
):
    """
    Generalized benchmark function for 1D potentials.

    Parameters:
        - PotentialClass (class): The class of the potential to be tested.
        - param_ranges (dict): A dictionary with parameter names as keys and lists of values to test as values.
        - param_names (list): List of parameter names in the order expected by the PotentialClass.
        - x_range (tuple): Range of x values to test the potential over x_range = (start, stop, number_of_points).
        - h (float): Spacing for finite difference approximation.
        - test_X (boolean): set False if u want to test only some of X = V (potential), F (force), H (hessian).
    """
    x = np.linspace(*x_range)

    # Test the potential function
    if test_V:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Potential for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=(12, 6))
            y_max = 0
            for param_value in param_ranges[param_name]:
                # Set parameters, varying only one at a time
                #  	•	For each parameter in param_names, use the first value from param_ranges if it is not the varied parameter.
                #   •	Use the current param_value for the parameter being varied (param_name).
                params = [
                    (
                        get_middle_value(param_ranges[name])
                        if name != param_name
                        else param_value
                    )
                    for name in param_names
                ]
                print(params)
                this_potential = PotentialClass(params)
                color = plt.cm.viridis(
                    (param_value - min(param_ranges[param_name]))
                    / (max(param_ranges[param_name]) - min(param_ranges[param_name]))
                )
                plt.plot(
                    x,
                    this_potential.potential(x),
                    color=color,
                    label=f"{param_name}={param_value:.2f}",
                )

                y_max_i = max(this_potential.potential(x))
                if y_max_i > y_max:
                    y_max = y_max_i
                else:
                    pass

            # Add text annotation
            plt.text(
                get_middle_value(x),
                y_max,
                f"{extract_potential_expression_from_docstring(PotentialClass)}",
                fontsize=12,
                ha="center",
            )
            plt.xlabel("x")
            plt.ylabel("V(x)")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            plt.title(f"Vary parameter {param_name}")
            plt.legend(loc="upper right")

    # Test the force function
    if test_F:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Force for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=(6, 6))
            y_max = 0
            plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)
            for param_value in param_ranges[param_name]:
                # Set parameters, varying only one at a time
                params = [
                    (
                        get_middle_value(param_ranges[name])
                        if name != param_name
                        else param_value
                    )
                    for name in param_names
                ]
                this_potential = PotentialClass(params)
                color = plt.cm.viridis(
                    (param_value - min(param_ranges[param_name]))
                    / (max(param_ranges[param_name]) - min(param_ranges[param_name]))
                )

                # Plot analytical force
                plt.plot(
                    x,
                    this_potential.force_ana(x),
                    color=color,
                    label=f"{param_name}={param_value:.2f}",
                )
                # Plot numerical force
                plt.plot(
                    x,
                    this_potential.force_num(x, h),
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=2,
                )

                y_max_i = max(this_potential.force_ana(x))
                if y_max_i > y_max:
                    y_max = y_max_i
                else:
                    pass

            plt.text(
                get_middle_value(x),
                y_max,
                f"{extract_force_expression_from_docstring(PotentialClass)}",
                fontsize=12,
                ha="center",
            )
            plt.xlabel("x")
            plt.ylabel("F(x)")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            plt.title(
                f"Force for various values of {param_name}, line: analytical, dots: numerical"
            )
            plt.legend(loc="upper right")

            # Plot difference between analytical and numerical force
            plt.figure(figsize=(6, 6))
            plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)
            for param_value in param_ranges[param_name]:
                # Set parameters, varying only one at a time
                params = [
                    (
                        get_middle_value(param_ranges[name])
                        if name != param_name
                        else param_value
                    )
                    for name in param_names
                ]
                this_potential = PotentialClass(params)
                color = plt.cm.viridis(
                    (param_value - min(param_ranges[param_name]))
                    / (max(param_ranges[param_name]) - min(param_ranges[param_name]))
                )

                # Plot deviation between analytical and numerical force
                plt.plot(
                    x,
                    this_potential.force_ana(x) - this_potential.force_num(x, h),
                    color=color,
                    label=f"{param_name}={param_value:.2f}",
                )
            plt.xlabel("x")
            plt.ylabel("F(x) - F_num(x)")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            plt.title(
                f"Deviation between analytical and numerical force for various values of {param_name}"
            )
            plt.legend(loc="upper right")

    # Test the Hessian function
    if test_H:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Hessian for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=(6, 6))
            y_max = 0
            plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)
            for param_value in param_ranges[param_name]:
                # Set parameters, varying only one at a time
                params = [
                    (
                        get_middle_value(param_ranges[name])
                        if name != param_name
                        else param_value
                    )
                    for name in param_names
                ]
                print(params)
                this_potential = PotentialClass(params)
                color = plt.cm.viridis(
                    (param_value - min(param_ranges[param_name]))
                    / (max(param_ranges[param_name]) - min(param_ranges[param_name]))
                )

                # Plot analytical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_ana(x),
                    color=color,
                    label=f"{param_name}={param_value:.2f}",
                )
                # Plot numerical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_num(x, h),
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=2,
                )

                y_max_i = max(this_potential.hessian_ana(x))
                if y_max_i > y_max:
                    y_max = y_max_i
                else:
                    pass

            plt.text(
                get_middle_value(x),
                y_max,
                f"{extract_hessian_expression_from_docstring(PotentialClass)}",
                fontsize=12,
                ha="center",
            )
            plt.xlabel("x")
            plt.ylabel("H(x)")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            plt.title(
                f"Hessian for various values of {param_name}, line: analytical, dots: numerical"
            )
            plt.legend(loc="upper right")

            # Plot difference between analytical and numerical Hessian
            plt.figure(figsize=(6, 6))
            plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)
            for param_value in param_ranges[param_name]:
                # Set parameters, varying only one at a time
                params = [
                    (
                        get_middle_value(param_ranges[name])
                        if name != param_name
                        else param_value
                    )
                    for name in param_names
                ]
                print(params)
                this_potential = PotentialClass(params)
                color = plt.cm.viridis(
                    (param_value - min(param_ranges[param_name]))
                    / (max(param_ranges[param_name]) - min(param_ranges[param_name]))
                )

                # Plot deviation between analytical and numerical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_ana(x) - this_potential.hessian_num(x, h),
                    color=color,
                    label=f"{param_name}={param_value:.2f}",
                )
            plt.xlabel("x")
            plt.ylabel("H(x) - H_num(x)")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            plt.title(
                f"Deviation between analytical and numerical Hessian for various values of {param_name}"
            )
            plt.legend(loc="upper right")

    plt.show()
