"""
Created on Wed July 3 13:45:56 2024

@author: arthur

This tests the one-dimensional Morese potential 
implemented in the potential class

potential.D1(Morse)

"""

# -----------------------------------------
#   I M P O R T S
# -----------------------------------------
import matplotlib.pyplot as plt
import numpy as np

from potential import D1

# -----------------------------------------
#   Parameters
# -----------------------------------------
test_V = True
test_F = True
test_H = True


print("---------------------------------")
print(
    "Testing the Morse Potential for x = [1, 3, 4] with parameters: params = [D_e, a, x_e] = [5.0, 1.0, 1.0]"
)

my_param = [5.0, 1.0, 1.0]
my_potential = D1.Morse(my_param)

print("\nClass members: ")
print(dir(my_potential))
print("\nValues of the parameters")
print(f"D_e: {my_potential.D_e}, a: {my_potential.a}, x_e: {my_potential.x_e}")

x = np.array([1, 3, 4])
print("\nPotential at x =", x)
print(my_potential.potential(x))

# Set x-axis
x = np.linspace(0.5, 2.5, 100)
# Set distance between two points for finite difference approximation
h = 0.01

# -----------------------------------------
#   Benchmark Morse
# -----------------------------------------
if test_V:
    print("---------------------------------")
    print("Testing the Morse Potential for varying Parameters (D_e, a, x_e)")
    # ----------------------
    # Vary parameter D_e
    plt.figure(figsize=(12, 6))
    for D_e in np.linspace(1, 10, 5):
        this_param = [D_e, 1.0, 1.0]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((D_e - 1) / 9)  # normalize
        plt.plot(x, this_potential.potential(x), color=color, label=f"D_e={D_e:.2f}")
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title("Vary parameter D_e")
    plt.legend()

    # ----------------------
    # Vary parameter a
    plt.figure(figsize=(12, 6))
    for a in np.linspace(0.5, 2.0, 5):
        this_param = [5.0, a, 1.0]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((a - 0.5) / 1.5)  # normalize
        plt.plot(x, this_potential.potential(x), color=color, label=f"a={a:.2f}")
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title("Vary parameter a")
    plt.legend()

    # ----------------------
    # Vary parameter x_e
    plt.figure(figsize=(12, 6))
    for x_e in np.linspace(0.75, 1.25, 5):
        this_param = [5.0, 1.0, x_e]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((x_e - 0.75) / 0.5)  # normalize
        plt.plot(x, this_potential.potential(x), color=color, label=f"x_e={x_e:.2f}")
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title("Vary parameter x_e")
    plt.legend()

# -----------------------------------------
#   Benchmark Force
# -----------------------------------------
if test_F:
    print("---------------------------------")
    print("Testing the Force for varying Parameters (D_e, a, x_e)")

    # ----------------------
    # Vary parameter a
    plt.figure(figsize=(6, 6))
    plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)

    for a in np.linspace(0.5, 2.0, 5):
        this_param = [5.0, a, 1.0]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((a - 0.5) / 1.5)  # normalize

        # Plot analytical force
        plt.plot(x, this_potential.force_ana(x), color=color, label=f"a={a:.2f}")
        # Plot numerical force
        plt.plot(
            x,
            this_potential.force_num(x, h),
            color=color,
            marker="o",
            linestyle="None",
            markersize=3,
        )

    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Force for various values of a, line: analytical, dots: numerical")
    plt.legend()

    # ----------------------
    # Plot difference between analytical and numerical force
    plt.figure(figsize=(6, 6))
    plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)

    for a in np.linspace(0.5, 2.0, 5):
        this_param = [5.0, a, 1.0]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((a - 0.5) / 1.5)  # normalize

        # Plot deviation between analytical and numerical force
        plt.plot(
            x,
            this_potential.force_ana(x) - this_potential.force_num(x, h),
            color=color,
            label=f"a={a:.2f}",
        )

    plt.xlabel("x")
    plt.ylabel("F(x) - F_num(x)")
    plt.title(
        "Deviation between analytical force and numerical force for various values of a"
    )
    plt.legend()

# -----------------------------------------
#   Benchmark Hessian
# -----------------------------------------
if test_H:
    print("---------------------------------")
    print("Testing the Hessian for varying Parameters (D_e, a, x_e)")

    # ----------------------
    # Vary parameter D_e, plot analytical and numerical Hessian
    plt.figure(figsize=(6, 6))
    plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)

    for D_e in np.linspace(1, 10, 5):
        this_param = [D_e, 1.0, 1.0]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((D_e - 1) / 9)  # normalize

        # Plot analytical Hessian
        plt.plot(x, this_potential.hessian_ana(x), color=color, label=f"D_e={D_e:.2f}")
        # Plot numerical Hessian
        plt.plot(
            x,
            this_potential.hessian_num(x, h),
            color=color,
            marker="o",
            linestyle="None",
            markersize=3,
        )

    plt.xlabel("x")
    plt.ylabel("H(x)")
    plt.title("Hessian for various values of D_e, line: analytical, dots: numerical")
    plt.legend()

    # ----------------------
    # Plot difference between analytical and numerical Hessian
    plt.figure(figsize=(6, 6))
    plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)

    for D_e in np.linspace(1, 10, 5):
        this_param = [D_e, 1.0, 1.0]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((D_e - 1) / 9)  # normalize

        # Plot deviation between analytical and numerical Hessian
        plt.plot(
            x,
            this_potential.hessian_ana(x) - this_potential.hessian_num(x, h),
            color=color,
            label=f"D_e={D_e:.2f}",
        )

    plt.xlabel("x")
    plt.ylabel("H(x) - H_num(x)")
    plt.title(
        "Deviation between analytical Hessian and numerical Hessian for various values of D_e"
    )
    plt.legend()

    # ----------------------
    # Vary parameter a, plot analytical and numerical Hessian
    plt.figure(figsize=(6, 6))
    plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)

    for a in np.linspace(0.5, 2.0, 5):
        this_param = [5.0, a, 1.0]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((a - 0.5) / 1.5)  # normalize

        # Plot analytical Hessian
        plt.plot(x, this_potential.hessian_ana(x), color=color, label=f"a={a:.2f}")
        # Plot numerical Hessian
        plt.plot(
            x,
            this_potential.hessian_num(x, h),
            color=color,
            marker="o",
            linestyle="None",
            markersize=3,
        )

    plt.xlabel("x")
    plt.ylabel("H(x)")
    plt.title("Hessian for various values of a, line: analytical, dots: numerical")
    plt.legend()

    # ----------------------
    # Plot difference between analytical and numerical Hessian
    plt.figure(figsize=(6, 6))
    plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)

    for a in np.linspace(0.5, 2.0, 5):
        this_param = [5.0, a, 1.0]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((a - 0.5) / 1.5)  # normalize

        # Plot deviation between analytical and numerical Hessian
        plt.plot(
            x,
            this_potential.hessian_ana(x) - this_potential.hessian_num(x, h),
            color=color,
            label=f"a={a:.2f}",
        )

    plt.xlabel("x")
    plt.ylabel("H(x) - H_num(x)")
    plt.title(
        "Deviation between analytical Hessian and numerical Hessian for various values of a"
    )
    plt.legend()

    # ----------------------
    # Vary parameter x_e, plot analytical and numerical Hessian
    plt.figure(figsize=(6, 6))
    plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)

    for x_e in np.linspace(0.75, 1.25, 5):
        this_param = [5.0, 1.0, x_e]
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((x_e - 0.75) / 0.5)  # normalize

        # Plot analytical Hessian
        plt.plot(x, this_potential.hessian_ana(x), color=color, label=f"x_e={x_e:.2f}")
        # Plot numerical Hessian
        plt.plot(
            x,
            this_potential.hessian_num(x, h),
            color=color,
            marker="o",
            linestyle="None",
            markersize=3,
        )

    plt.xlabel("x")
    plt.ylabel("H(x)")
    plt.title("Hessian for various values of x_e, line: analytical, dots: numerical")
    plt.legend()

    # ----------------------
    # Plot difference between analytical and numerical Hessian
    plt.figure(figsize=(6, 6))
    plt.axhline(y=0, color="black", linestyle="dotted", linewidth=1)

    for x_e in np.linspace(0.75, 1.25, 5):
        this_param = [5.0, 0.5, x_e]
        print(this_param)
        this_potential = D1.Morse(this_param)
        color = plt.cm.viridis((x_e - 0.75) / 0.5)  # normalize

        # Plot deviation between analytical and numerical Hessian
        plt.plot(
            x,
            this_potential.hessian_ana(x) - this_potential.hessian_num(x, h),
            color=color,
            label=f"x_e={x_e:.2f}",
        )

    plt.xlabel("x")
    plt.ylabel("H(x) - H_num(x)")
    plt.title(
        "Deviation between analytical Hessian and numerical Hessian for various values of x_e"
    )
    plt.legend()


plt.show()
