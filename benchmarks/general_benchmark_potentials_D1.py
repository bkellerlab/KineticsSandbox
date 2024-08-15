"""
Created on Wed July 3 13:45:56 2024

@author: arthur

This module holds functions to benchmark 1D potentials from the potential.D1 class

"""

import os

# -----------------------------------------
#   I M P O R T S
# -----------------------------------------
import re
import sys

sys.path.append("..")


import ipywidgets as widgets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import fixed, interact

# -----------------------------------------------------
# Functions that are needed for the benchmark functions
# -----------------------------------------------------

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


def _plot_potential(PotentialClass, param_ranges, param_names, x_range, param_name, param_value, h, test_V, test_F, test_H, **params):
    x = np.linspace(*x_range)

    # Calculate initial reference values for limits based on mid-point parameters
    ref_params = [
        get_middle_value(param_ranges[name])
        for name in param_names
    ]
    ref_potential = PotentialClass(ref_params)

    if test_V:
        potential_ref_data = ref_potential.potential(x)
    if test_F:
        force_ref_data = ref_potential.force_ana(x)
    if test_H:
        hessian_ref_data = ref_potential.hessian_ana(x)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Calculate global y-limits based on reference data
    def global_limits(*datasets, padding=0.1):
        min_val = min(np.concatenate(datasets))
        max_val = max(np.concatenate(datasets))
        range_val = max_val - min_val
        return (min_val - padding * range_val, max_val + padding * range_val)

    if test_V:
        ax2 = ax1.twinx()
        potential_data = PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).potential(x)
        ax2.plot(x, potential_data, 'r-', label="Potential")
        ax2.set_ylabel('Potential $V(x)$', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(global_limits(potential_ref_data))

        if test_V and not (test_F and test_H):
            fig.text(
                0.5, 1.15,
                f"{extract_potential_expression_from_docstring(PotentialClass)}",
                fontsize=12,
                ha="center",
                va="center",
            )

    if test_F and not test_H:
        force_data = PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).force_ana(x)
        ax1.plot(x, force_data, 'b-', label="Analytical Force")
        ax1.plot(x, PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).force_num(x, h), 'bo', label="Numerical Force", markersize=2)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("Force $F(x)$", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(global_limits(force_ref_data))

        fig.text(
            0.5, 1.1,
            f"{extract_force_expression_from_docstring(PotentialClass)}",
            fontsize=12,
            ha="center",
            va="center",
        )

    if test_H and not test_F:
        hessian_data = PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).hessian_ana(x)
        ax1.plot(x, hessian_data, 'g-', label="Analytical Hessian")
        ax1.plot(x, PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).hessian_num(x, h), 'go', label="Numerical Hessian", markersize=2)
        ax1.set_ylabel("Hessian $H(x)$", color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.set_ylim(global_limits(hessian_ref_data))

        fig.text(
            0.5, 1.05,
            f"{extract_hessian_expression_from_docstring(PotentialClass)}",
            fontsize=12,
            ha="center",
            va="center",
        )

    if test_H and test_F:
        force_data = PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).force_ana(x)
        hessian_data = PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).hessian_ana(x)
        ax1.plot(x, hessian_data, 'g-', label="Analytical Hessian")
        ax1.plot(x, PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).hessian_num(x, h), 'go', label="Numerical Hessian", markersize=2)
        ax1.plot(x, force_data, 'b-', label="Analytical Force")
        ax1.plot(x, PotentialClass([param_value if name == param_name else get_middle_value(param_ranges[name]) for name in param_names]).force_num(x, h), 'bo', label="Numerical Force", markersize=2)
        ax1.set_ylabel("Force $F(x)$ / Hessian $H(x)$", color='k')
        ax1.set_ylim(global_limits(force_ref_data, hessian_ref_data))
        ax1.tick_params(axis='y', labelcolor='k')

        fig.text(
            0.5, 1.15,
            f"{extract_potential_expression_from_docstring(PotentialClass)}",
            fontsize=12,
            ha="center",
            va="center",
        )

        fig.text(
            0.5, 1.1,
            f"{extract_force_expression_from_docstring(PotentialClass)}",
            fontsize=12,
            ha="center",
            va="center",
        )

        fig.text(
            0.5, 1.05,
            f"{extract_hessian_expression_from_docstring(PotentialClass)}",
            fontsize=12,
            ha="center",
            va="center",
        )

    ax1.set_xlabel("x")
    fig.tight_layout()
    fig.legend(framealpha=1, loc="upper right", facecolor='white', edgecolor='black')
    plt.title(f"Potential, Force, and Hessian for {param_name} = {np.round(param_value, 2)}")
    plt.show()

def _update_param_value_slider(change, sliders, param_ranges):
    param_name = change['new']
    sliders['param_value'].min = min(param_ranges[param_name])
    sliders['param_value'].max = max(param_ranges[param_name])
    sliders['param_value'].step = (max(param_ranges[param_name]) - min(param_ranges[param_name])) / 100.0
    sliders['param_value'].value = get_middle_value(param_ranges[param_name])

    
# ----------------------------------------
# Benchmark functions used in python files
# ----------------------------------------

def benchmark_1d_potential(
    PotentialClass,
    param_ranges,
    param_names,
    x_range,
    h,
    test_V=True,
    test_F=True,
    test_H=True,
    figsize=(12, 8),
    save=False,
    title= True
):
    """
    Generalized benchmark function for 1D potentials.
    The function tests the potential energy, force,
    and Hessian by plotting them for varying parameters
    and comparing the results of analytical force and
    Hessian to the finite difference approximation.

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
            plt.figure(figsize=figsize)
            y_max = 0
            for param_value in param_ranges[param_name]:
                # Set parameters, varying only one at a time
                #  	•	For each parameter in param_names, use the middle value from param_ranges if it is not the varied parameter.
                #   •	Use the current param_value for the parameter being varied (param_name).
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
                plt.plot(
                    x,
                    this_potential.potential(x),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )

                y_max_i = max(this_potential.potential(x))
                if y_max_i > y_max:
                    y_max = y_max_i
                else:
                    pass

            # Add text annotation
            plt.text(
                get_middle_value(x),
                y_max - 0.05 * y_max,
                f"{extract_potential_expression_from_docstring(PotentialClass)}",
                fontsize=12,
                ha="center",
            )
            plt.xlabel("$x$")
            plt.ylabel("$V(x)$")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            if title:
                plt.title(f"Vary parameter ${param_name}$")
            plt.legend(loc="upper right")
            plt.tight_layout()
            if save:
                directory = f"benchmarks/Figures/{PotentialClass.__name__}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(f"{directory}/{PotentialClass.__name__}_V_vary_param_{param_name}.png")

    # Test the force function
    if test_F:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Force for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=figsize)
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
                    label=f"${param_name}$ = {param_value:.2f}",
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
                y_max - 0.05 * y_max,
                f"{extract_force_expression_from_docstring(PotentialClass)}",
                fontsize=12,
                ha="center",
            )
            plt.xlabel("$x$")
            plt.ylabel("$F(x)$")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            if title:
                plt.title(
                    f"Force for various values of ${param_name}$, line: analytical, dots: numerical"
                )
            plt.legend(loc="upper right")
            plt.tight_layout()
            if save:
                directory = f"benchmarks/Figures/{PotentialClass.__name__}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(f"{directory}/{PotentialClass.__name__}_F_vary_param_{param_name}.png")

            # Plot difference between analytical and numerical force
            plt.figure(figsize=figsize)
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
                    label=f"${param_name}$ = {param_value:.2f}",
                )
            plt.xlabel("$x$")
            plt.ylabel("$F(x) - F_{num}(x)$")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            if title:
                plt.title(
                    f"Deviation between analytical and numerical force for various values of ${param_name}$"
                )
            plt.legend(loc="upper right")
            plt.tight_layout()
            if save:
                directory = f"benchmarks/Figures/{PotentialClass.__name__}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(f"{directory}/{PotentialClass.__name__}_F_ana_num_{param_name}.png")

    # Test the Hessian function
    if test_H:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Hessian for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=figsize)
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

                # Plot analytical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_ana(x),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
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
                y_max - 0.05 * y_max,
                f"{extract_hessian_expression_from_docstring(PotentialClass)}",
                fontsize=12,
                ha="center",
            )
            plt.xlabel("$x$")
            plt.ylabel("$H(x)$")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            if title:
                plt.title(
                   f"Hessian for various values of ${param_name}$, line: analytical, dots: numerical"
                )
            plt.legend(loc="upper right")
            plt.tight_layout()
            if save:
                directory = f"benchmarks/Figures/{PotentialClass.__name__}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(f"{directory}/{PotentialClass.__name__}_H_vary_param_{param_name}.png")

            # Plot difference between analytical and numerical Hessian
            plt.figure(figsize=figsize)
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

                # Plot deviation between analytical and numerical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_ana(x) - this_potential.hessian_num(x, h),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )
            plt.xlabel("$x$")
            plt.ylabel("$H(x) - H_{num}(x)$")
            plt.xlim(min(x), max(x) + 0.2 * (max(x) - min(x)))
            if title:
                plt.title(
                   f"Deviation between analytical and numerical Hessian for various values of ${param_name}$"
                )
            plt.legend(loc="upper right")
            plt.tight_layout()
            if save:
                directory = f"benchmarks/Figures/{PotentialClass.__name__}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(f"{directory}/{PotentialClass.__name__}_H_ana_num_{param_name}.png")

    plt.show()

def benchmark_1d_potential_subplots(
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
    The function tests the potential energy, force,
    and Hessian by plotting them for varying parameters
    and comparing the results of analytical force and
    Hessian to the finite difference approximation.

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


        fig = plt.figure(figsize=(8, 4 * len(param_names)))
        gs = gridspec.GridSpec(len(param_names), 2, width_ratios=[90, 10])

        for i, param_name in enumerate(param_names):
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])

            for param_value in param_ranges[param_name]:
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
                ax1.plot(
                    x,
                    this_potential.potential(x),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )
                ax2.hlines(
                    y = param_value,
                    xmin = min(x),
                    xmax = max(x) - 0.7 * (max(x) - min(x)),
                    colors=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )

                ax2.text(
                    max(x) - 0.2 * (max(x) - min(x)),
                    param_value,
                    f"{param_value:.2f}",
                    fontsize=9,
                    ha="center",
                )

            ax1.set_xlabel("$x$")
            ax1.set_ylabel("$V(x)$")
            ax1.set_xlim(min(x), max(x))
            ax1.set_title(f"Vary parameter ${param_name}$")
            #ax1.legend(loc="upper right")

            ax2.set_xlabel("")
            ax2.set_xlim(min(x), max(x))
            ax2.set_ylim(min(param_ranges[param_name]) - 0.2 * (max(param_ranges[param_name]) - min(param_ranges[param_name])), max(param_ranges[param_name]) + 0.2 * (max(param_ranges[param_name]) - min(param_ranges[param_name])))
            ax2.axis('off')
            ax2.set_ylabel("")
            ax2.set_title(f"${param_name} = $")

        fig.tight_layout()
        plt.show()
        plt.close()

    # Test the force function
    if test_F:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Force for varying parameters")

        fig = plt.figure(figsize=(12, 4 * len(param_names)))
        gs = gridspec.GridSpec(len(param_names), 3, width_ratios=[45, 45, 10])

        for i, param_name in enumerate(param_names):
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            ax3 = fig.add_subplot(gs[i, 2])

            for param_value in param_ranges[param_name]:
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
                ax1.plot(
                    x,
                    this_potential.force_ana(x),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )
                # Plot numerical force
                ax1.plot(
                    x,
                    this_potential.force_num(x, h),
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=2,
                )

                # Plot deviation between analytical and numerical force
                ax2.plot(
                    x,
                    this_potential.force_ana(x) - this_potential.force_num(x, h),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )
                ax3.hlines(
                    y = param_value,
                    xmin = min(x),
                    xmax = max(x) - 0.7 * (max(x) - min(x)),
                    colors=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )

                ax3.text(
                    max(x) - 0.2 * (max(x) - min(x)),
                    param_value,
                    f"{param_value:.2f}",
                    fontsize=9,
                    ha="center",
                )

            ax1.set_xlabel("$x$")
            ax1.set_ylabel("$F(x)$")
            ax1.set_xlim(min(x), max(x))
            ax1.set_title(f"Force for various values of ${param_name}$")
            #ax1.legend(loc="upper right")

            ax2.set_xlabel("$x$")
            ax2.set_ylabel("$F(x) - F_{num}(x)$")
            ax2.set_title(f"Deviation analytical and numerical Force for ${param_name}$")
            ax2.set_xlim(min(x), max(x))
            #ax2.legend(loc="upper right")

            ax3.set_xlabel("")
            ax3.set_xlim(min(x), max(x))
            ax3.set_ylim(min(param_ranges[param_name]) - 0.2 * (max(param_ranges[param_name]) - min(param_ranges[param_name])), max(param_ranges[param_name]) + 0.2 * (max(param_ranges[param_name]) - min(param_ranges[param_name])))
            ax3.axis('off')
            ax3.set_ylabel("")
            ax3.set_title(f"${param_name} = $")

        fig.tight_layout()
        plt.show()
        plt.close()

    # Test the Hessian function
    if test_H:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Hessian for varying parameters")

        fig = plt.figure(figsize=(12, 4 * len(param_names)))
        gs = gridspec.GridSpec(len(param_names), 3, width_ratios=[45, 45, 10])

        for i, param_name in enumerate(param_names):
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            ax3 = fig.add_subplot(gs[i, 2])

            for param_value in param_ranges[param_name]:
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

                # Plot analytical Hessian
                ax1.plot(
                    x,
                    this_potential.hessian_ana(x),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )
                # Plot numerical Hessian
                ax1.plot(
                    x,
                    this_potential.hessian_num(x, h),
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=2,
                )

                # Plot deviation between analytical and numerical Hessian
                ax2.plot(
                    x,
                    this_potential.hessian_ana(x) - this_potential.hessian_num(x, h),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )

                ax3.hlines(
                    y = param_value,
                    xmin = min(x),
                    xmax = max(x) - 0.7 * (max(x) - min(x)),
                    colors=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )

                ax3.text(
                    max(x) - 0.2 * (max(x) - min(x)),
                    param_value,
                    f"{param_value:.2f}",
                    fontsize=9,
                    ha="center",
                )

            ax1.set_xlabel("$x$")
            ax1.set_xlim(min(x), max(x))
            ax1.set_ylabel("$H(x)$")
            ax1.set_title(f"Hessian for various values of ${param_name}$")
            #ax1.legend(loc="upper right")

            ax2.set_xlabel("$x$")
            ax2.set_xlim(min(x), max(x))
            ax2.set_ylabel("$H(x) - H_{num}(x)$")
            ax2.set_title(f"Deviation analytical and numerical Hessian for ${param_name}$")
            #ax2.legend(loc="upper right")

            ax3.set_xlabel("")
            ax3.set_xlim(min(x), max(x))
            ax3.set_ylim(min(param_ranges[param_name]) - 0.2 * (max(param_ranges[param_name]) - min(param_ranges[param_name])), max(param_ranges[param_name]) + 0.2 * (max(param_ranges[param_name]) - min(param_ranges[param_name])))
            ax3.axis('off')
            ax3.set_ylabel("")
            ax3.set_title(f"${param_name} = $")

        fig.tight_layout()
        plt.show()
        plt.close()

# -----------------------------------------
# These two functions are manually adjusted 
# from the 'benchmark_1d_potential' to
# make better plots for a report.

def benchmark_1d_potential_morse(
    PotentialClass,
    param_ranges,
    param_names,
    x_range,
    h,
    test_V=True,
    test_F=True,
    test_H=True,
    save=False,
    title=True
):
    """
    Generalized benchmark function for 1D potentials, 
    with focus on the Morse potential in its original 
    form: V(x) = D_e (1 - e^{(-a(x - x_e))})^2 - D_e.
    The last term "- D_e" is important.
    The function tests the potential energy, force,
    and Hessian by plotting them for varying parameters
    and comparing the results of analytical force and
    Hessian to the finite difference approximation.

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
            if param_name == "D_e":
                plt.figure(figsize=(6, 8))
                y_max = 0
                plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
                for i, param_value in enumerate(param_ranges[param_name]):
                    # Set parameters, varying only one at a time
                    #  	•	For each parameter in param_names, use the middle value from param_ranges if it is not the varied parameter.
                    #   •	Use the current param_value for the parameter being varied (param_name).
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
                    plt.text(
                        max(x) + 0.09 *(max(x) - min(x)),
                        - (param_value - 0.2),
                        f"${param_name}$ = {param_value:.2f}",

                        ha="left",
                    )
                    plt.axhline(y= - param_value, color="black", linestyle="dotted", linewidth=1)
                    plt.vlines(
                        x = max(x) + 0.01 * i * (max(x) - min(x)),
                        ymin= - param_value,
                        ymax= 0,
                        colors=color,
                        linestyles="-"
                    )
                    x_new = np.append(x, max(x) + 0.01 * i * (max(x) - min(x)))
                    plt.plot(
                        x_new,
                        this_potential.potential(x_new),
                        color=color,
                        label=f"${param_name}$ = {param_value:.2f}",
                    )

                    y_max_i = max(this_potential.potential(x))
                    if y_max_i > y_max:
                        y_max = y_max_i
                    else:
                        pass

                ## Add text annotation
                #plt.text(
                #    get_middle_value(x),
                #    y_max - 0.05 * y_max,
                #    f"{extract_potential_expression_from_docstring(PotentialClass)}",
#
                #    ha="center",
                #)
                plt.xlabel("$x$")
                plt.ylabel("$V(x)$")
                plt.xlim(min(x), max(x) + 0.5 * (max(x) - min(x)))
                if title:
                    plt.title(f"Vary parameter ${param_name}$")
                #plt.legend(loc="upper right")
                plt.tight_layout()
                if save:
                    plt.savefig(f"benchmarks/Figures/Morse/morse_V_vary_param_{param_name}.png")
            else:
                plt.figure(figsize=(6, 8))
                y_max = 0
                if "D_e" in param_names:
                        plt.axhline(y= - get_middle_value(param_ranges["D_e"]), color="black", linestyle="dotted", linewidth=1)
                        plt.vlines(x = max(x), ymin = 0, ymax= - get_middle_value(param_ranges["D_e"]), colors="black", linestyles="-", linewidth=1)
                        plt.text(
                        max(x) + 0.08 *(max(x) - min(x)),
                        - get_middle_value(param_ranges["D_e"]) / 2,
                        f"$D_e$ = {get_middle_value(param_ranges["D_e"]):.2f}",

                        ha="left",
                    )
                plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
                for param_value in param_ranges[param_name]:
                    # Set parameters, varying only one at a time
                    #  	•	For each parameter in param_names, use the middle value from param_ranges if it is not the varied parameter.
                    #   •	Use the current param_value for the parameter being varied (param_name).
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
                    plt.plot(
                        x,
                        this_potential.potential(x),
                        color=color,
                        label=f"${param_name}$ = {param_value:.2f}",
                    )

                    y_max_i = max(this_potential.potential(x))
                    if y_max_i > y_max:
                        y_max = y_max_i
                    else:
                        pass

                # Add text annotation
                #plt.text(
                #    get_middle_value(x),
                #    y_max - 0.05 * y_max,
                #    f"{extract_potential_expression_from_docstring(PotentialClass)}",
#
                #    ha="center",
                #)
                plt.xlabel("$x$")
                plt.ylabel("$V(x)$")
                plt.xlim(min(x), max(x) + 0.5 * (max(x) - min(x)))
                if title:
                    plt.title(f"Vary parameter ${param_name}$")
                plt.legend(loc="upper right", frameon=False)
                plt.tight_layout()
                if save:
                    plt.savefig(f"benchmarks/Figures/Morse/morse_V_vary_param_{param_name}.png")

    # Test the force function
    if test_F:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Force for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=(6, 6))
            y_max = 0
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
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
                    label=f"${param_name}$ = {param_value:.2f}",
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

            #plt.text(
            #    get_middle_value(x),
            #    y_max - 0.05 * y_max,
            #    f"{extract_force_expression_from_docstring(PotentialClass)}",
#
            #    ha="center",
            #)
            plt.xlabel("$x$")
            plt.ylabel("$F(x)$")
            plt.xlim(min(x), max(x))
            if title:
                plt.title(
                    f"Force for various values of ${param_name}$"
                )
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()
            if save:
                plt.savefig(f"benchmarks/Figures/Morse/morse_F_vary_param_{param_name}.png")

            # Plot difference between analytical and numerical force
            plt.figure(figsize=(6, 6))
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
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
                    label=f"${param_name}$ = {param_value:.2f}",
                )
            plt.xlabel("$x$")
            plt.ylabel("$F(x) - F_{num}(x)$")
            plt.xlim(min(x), max(x))
            if title:
                plt.title(
                    f"Deviation  analytical and numerical force for ${param_name}$"
                )
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()
            if save:
                plt.savefig(f"benchmarks/Figures/Morse/morse_F_ana_num_{param_name}.png")

    # Test the Hessian function
    if test_H:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Hessian for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=(6, 6))
            y_max = 0
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
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

                # Plot analytical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_ana(x),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
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

            #plt.text(
            #    get_middle_value(x),
            #    y_max - 0.05 * y_max,
            #    f"{extract_hessian_expression_from_docstring(PotentialClass)}",
#
            #    ha="center",
            #)
            plt.xlabel("$x$")
            plt.ylabel("$H(x)$")
            plt.xlim(min(x), max(x))
            if title:
                plt.title(
                    f"Hessian for various values of ${param_name}$"
                )
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()
            if save:
                plt.savefig(f"benchmarks/Figures/Morse/morse_H_vary_param_{param_name}.png")

            # Plot difference between analytical and numerical Hessian
            plt.figure(figsize=(6, 6))
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
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

                # Plot deviation between analytical and numerical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_ana(x) - this_potential.hessian_num(x, h),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )
            plt.xlabel("$x$")
            plt.ylabel("$H(x) - H_{num}(x)$")
            plt.xlim(min(x), max(x))
            if title:
                plt.title(
                    f"Deviation analytical and numerical Hessian for ${param_name}$"
                )
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()
            if save:
                plt.savefig(f"benchmarks/Figures/Morse/morse_H_ana_num_{param_name}.png")

    plt.show()


def benchmark_1d_potential_morse_black(
    PotentialClass,
    param_ranges,
    param_names,
    x_range,
    h,
    test_V=True,
    test_F=True,
    test_H=True,
    save=False,
    title=True
):
    """
    Generalized benchmark function for 1D potentials,
    with focus on the Morse potential in its original 
    form: V(x) = D_e (1 - e^{(-a(x - x_e))})^2 - D_e.
    The last term "- D_e" is important.
    The function tests the potential energy, force,
    and Hessian by plotting them for varying parameters
    and comparing the results of analytical force and
    Hessian to the finite difference approximation.

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
            if param_name == "D_e":
                plt.figure(figsize=(6, 8))
                y_max = 0
                plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
                for param_value in param_ranges[param_name]:
                    # Set parameters, varying only one at a time
                    #  	•	For each parameter in param_names, use the middle value from param_ranges if it is not the varied parameter.
                    #   •	Use the current param_value for the parameter being varied (param_name).
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
                    plt.text(
                        max(x) + 0.09 *(max(x) - min(x)),
                        - (param_value - 0.2),
                        f"${param_name}$ = {param_value:.2f}",

                        ha="left",
                    )
                    plt.axhline(y= - param_value, color="black", linestyle="dotted", linewidth=1)
                    plt.plot(
                        x,
                        this_potential.potential(x),
                        color=color,
                        label=f"${param_name}$ = {param_value:.2f}",
                    )

                    y_max_i = max(this_potential.potential(x))
                    if y_max_i > y_max:
                        y_max = y_max_i
                    else:
                        pass

                plt.vlines(
                        x = max(x),
                        ymin= - max(param_ranges["D_e"]),
                        ymax= 0,
                        colors="k",
                        linestyles="-"
                    )
                ## Add text annotation
                #plt.text(
                #    get_middle_value(x),
                #    y_max - 0.05 * y_max,
                #    f"{extract_potential_expression_from_docstring(PotentialClass)}",
#
                #    ha="center",
                #)
                plt.xlabel("$x$")
                plt.ylabel("$V(x)$")
                plt.xlim(min(x), max(x) + 0.5 * (max(x) - min(x)))
                if title:
                    plt.title(f"Vary parameter ${param_name}$")
                #plt.legend(loc="upper right")
                plt.tight_layout()
                if save:
                    plt.savefig(f"benchmarks/Figures/Morse/morse_V_vary_param_{param_name}.png", facecolor='none', edgecolor='none', transparent=True)
            else:
                plt.figure(figsize=(6, 8))
                y_max = 0
                if "D_e" in param_names:
                        plt.axhline(y= - get_middle_value(param_ranges["D_e"]), color="black", linestyle="dotted", linewidth=1)
                        plt.vlines(x = max(x), ymin = 0, ymax= - get_middle_value(param_ranges["D_e"]), colors="black", linestyles="-", linewidth=1)
                        plt.text(
                        max(x) + 0.08 *(max(x) - min(x)),
                        - get_middle_value(param_ranges["D_e"]) / 2,
                        f"$D_e$ = {get_middle_value(param_ranges["D_e"]):.2f}",

                        ha="left",
                    )
                plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
                for param_value in param_ranges[param_name]:
                    # Set parameters, varying only one at a time
                    #  	•	For each parameter in param_names, use the middle value from param_ranges if it is not the varied parameter.
                    #   •	Use the current param_value for the parameter being varied (param_name).
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
                    plt.plot(
                        x,
                        this_potential.potential(x),
                        color=color,
                        label=f"${param_name}$ = {param_value:.2f}",
                    )

                    y_max_i = max(this_potential.potential(x))
                    if y_max_i > y_max:
                        y_max = y_max_i
                    else:
                        pass

                # Add text annotation
                #plt.text(
                #    get_middle_value(x),
                #    y_max - 0.05 * y_max,
                #    f"{extract_potential_expression_from_docstring(PotentialClass)}",
#
                #    ha="center",
                #)
                plt.xlabel("$x$")
                plt.ylabel("$V(x)$")
                plt.xlim(min(x), max(x) + 0.5 * (max(x) - min(x)))
                if title:
                    plt.title(f"Vary parameter ${param_name}$")
                plt.legend(loc="upper right", frameon=False)
                plt.tight_layout()
                if save:
                    plt.savefig(f"benchmarks/Figures/Morse/morse_V_vary_param_{param_name}.png")

    # Test the force function
    if test_F:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Force for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=(6, 6))
            y_max = 0
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
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
                    label=f"${param_name}$ = {param_value:.2f}",
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

            #plt.text(
            #    get_middle_value(x),
            #    y_max - 0.05 * y_max,
            #    f"{extract_force_expression_from_docstring(PotentialClass)}",
#
            #    ha="center",
            #)
            plt.xlabel("$x$")
            plt.ylabel("$F(x)$")
            plt.xlim(min(x), max(x))
            if title:
                plt.title(
                    f"Force for various values of ${param_name}$"
                )
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()
            if save:
                plt.savefig(f"benchmarks/Figures/Morse/morse_F_vary_param_{param_name}.png", facecolor='none', edgecolor='none', transparent=True)

            # Plot difference between analytical and numerical force
            plt.figure(figsize=(6, 6))
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
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
                    label=f"${param_name}$ = {param_value:.2f}",
                )
            plt.xlabel("$x$")
            plt.ylabel("$F(x) - F_{num}(x)$")
            plt.xlim(min(x), max(x))
            if title:
                plt.title(
                    f"Deviation  analytical and numerical force for ${param_name}$"
                )
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()
            if save:
                plt.savefig(f"benchmarks/Figures/Morse/morse_F_ana_num_{param_name}.png", facecolor='none', edgecolor='none', transparent=True)

    # Test the Hessian function
    if test_H:
        print("---------------------------------")
        print(f"Testing the {PotentialClass.__name__} Hessian for varying parameters")

        for param_name in param_names:
            plt.figure(figsize=(6, 6))
            y_max = 0
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
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

                # Plot analytical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_ana(x),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
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

            #plt.text(
            #    get_middle_value(x),
            #    y_max - 0.05 * y_max,
            #    f"{extract_hessian_expression_from_docstring(PotentialClass)}",
#
            #    ha="center",
            #)
            plt.xlabel("$x$")
            plt.ylabel("$H(x)$")
            plt.xlim(min(x), max(x))
            if title:
                plt.title(
                    f"Hessian for various values of ${param_name}$"
                )
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()
            if save:
                plt.savefig(f"benchmarks/Figures/Morse/morse_H_vary_param_{param_name}.png", facecolor='none', edgecolor='none', transparent=True)

            # Plot difference between analytical and numerical Hessian
            plt.figure(figsize=(6, 6))
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
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

                # Plot deviation between analytical and numerical Hessian
                plt.plot(
                    x,
                    this_potential.hessian_ana(x) - this_potential.hessian_num(x, h),
                    color=color,
                    label=f"${param_name}$ = {param_value:.2f}",
                )
            plt.xlabel("$x$")
            plt.ylabel("$H(x) - H_{num}(x)$")
            plt.xlim(min(x), max(x))
            if title:
                plt.title(
                    f"Deviation analytical and numerical Hessian for ${param_name}$"
                )
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()
            if save:
                plt.savefig(f"benchmarks/Figures/Morse/morse_H_ana_num_{param_name}.png", facecolor='none', edgecolor='none', transparent=True)

    plt.show()


# -------------------------------------------------------
# Benchmark function used in jupyter notebook environment
# -------------------------------------------------------

def interactive_benchmark_jupyter(PotentialClass, param_ranges, param_names, x_range, h, test_V=True, test_F=True, test_H=True):
    """
    Create an interactive benchmark for a given potential class using Jupyter widgets.

    This function generates an interactive plot that allows the user to explore the potential,
    force, and Hessian of a given potential class by adjusting its parameters using sliders.

    Parameters:
    -----------
    PotentialClass : class
        The potential class to be benchmarked. This class should have methods for calculating
        the potential, force, and Hessian.
    param_ranges : dict
        A dictionary where keys are parameter names and values are lists or arrays specifying
        the range of values for each parameter.
    param_names : list of str
        A list of parameter names to be included in the interactive widgets.
    x_range : tuple
        A tuple specifying the range of x values for plotting (start, stop, num_points).
    h : float
        The step size for numerical differentiation.
    test_V : bool, optional
        If True, the potential will be plotted (default is True).
    test_F : bool, optional
        If True, the force will be plotted (default is True).
    test_H : bool, optional
        If True, the Hessian will be plotted (default is True).

    Notes:
    ------
    This function is intended to be used in a Jupyter notebook environment. The interactive
    widgets will not render properly in a standard Python script execution environment.
    """
    sliders = {}
    param_name_widget = widgets.Dropdown(options=param_names, description='Parameter')
    param_value_widget = widgets.FloatSlider(description='Value')

    sliders['param_name'] = param_name_widget
    sliders['param_value'] = param_value_widget

    param_name_widget.observe(lambda change: _update_param_value_slider(change, sliders, param_ranges), names='value')
    _update_param_value_slider({'new': param_names[0]}, sliders, param_ranges)  # Initialize slider

    interact(
        _plot_potential,
        PotentialClass=fixed(PotentialClass),
        param_ranges=fixed(param_ranges),
        param_names=fixed(param_names),
        x_range=fixed(x_range),
        h=fixed(h),
        test_V=fixed(test_V),
        test_F=fixed(test_F),
        test_H=fixed(test_H),
        param_name=param_name_widget,
        param_value=param_value_widget
    )