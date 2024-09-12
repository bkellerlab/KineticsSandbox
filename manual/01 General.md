# Units

We follow the GROMACS conventions for units:
**https://manual.gromacs.org/documentation/2019/reference-manual/definitions.html**

For SI units, see
**https://en.wikipedia.org/wiki/International_System_of_Units**

## Base quantities

| Quantity            | Symbol | Unit                                                                                          | Unit name                         |
| ------------------- | ------ | --------------------------------------------------------------------------------------------- | --------------------------------- |
| time                | $t$    | $\text{ps}= 10^{-12}\, \text{s}$                                                              | picosecond                        |
| length              | $x$    | $\text{nm} = 10^{-9}\,\text{m}$                                                               | nanometer                         |
| mass                | $m$    | $\text{u} = 1.66054 \cdot 10^{−27}\,  \text{kg} \approx 10^{-3} \frac{\text{kg}}{\text{mol}}$ | unified atomic mass unit / Dalton |
| temperature         | $T$    | $\text{K}$                                                                                    | Kelvin                            |
| charge              | $q$    | e???                                                                                          | elementary charge ???             |
| collision frequency | $\xi$  | $\text{ps}^{-1} = 10^{12}\, \text{s}^{-1}$                                                    |                                   |

## Derived quantities

The conversion between GROMACS base units and SI base units, assumes that
$$u \cdot N_A = 0.99999999965(30)\cdot 10^{−3} \frac{\text{kg}}{\text{mol}} \approx 1$$
where $N_A$ is Avogadro's constant.

| Quantity         | Symbol and definition                            | GROMACS base units                        | SI base units                                                      | Unit                                         |
| ---------------- | ------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------- |
| velocity         | $v(t) = \frac{d}{dt}x(t)$                        | $\frac{\text{nm}}{\text{ps}}$             | $10^{3} \frac{\text{m}}{\text{s}}$                                 |                                              |
| acceleration     | $a(t) = \frac{d}{dt}v(t)$                        | $\frac{\text{nm}}{\text{ps}^2}$           | $10^{15} \frac{\text{m}}{\text{s}^2}$                              |                                              |
| linear momentum  | $p(t) = mv(t)$                                   | $\text{u}\frac{\text{nm}}{\text{ps}}$     | $\frac{\text{kg}}{\text{mol}}\frac{\text{m}}{\text{s}}$            |                                              |
| force            | $F(t) = m a(t)$                                  | $\text{u}\frac{\text{nm}}{\text{ps}^2}$   | $10^{12} \frac{\text{kg}}{\text{mol}}\frac{\text{m}}{\text{s}^2}$  | $\frac{\text{kJ}}{\text{mol}\cdot\text{nm}}$ |
| potential energy | $V(t) = F(t)\Delta x$                            | $\text{u}\frac{\text{nm}^2}{\text{ps}^2}$ | $10^{3} \frac{\text{kg}}{\text{mol}}\frac{\text{m}^2}{\text{s}^2}$ | $\frac{\text{kJ}}{\text{mol}}$               |
| kinetic energy   | $T(t) = \frac{p^2(t)}{2m} = \frac{1}{2}m v^2(t)$ | $\text{u}\frac{\text{nm}^2}{\text{ps}^2}$ | $10^{3} \frac{\text{kg}}{\text{mol}}\frac{\text{m}^2}{\text{s}^2}$ | $\frac{\text{kJ}}{\text{mol}}$               |

These units are consistent with units for the derivates of the potential energy function:

| Quantity         | Symbol and definition        | GROMACS base units                        | SI base units                                                      | Unit                                           |
| ---------------- | ---------------------------- | ----------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------- |
| potential energy | $V(x)$                       | $\text{u}\frac{\text{nm}^2}{\text{ps}^2}$ | $10^{3} \frac{\text{kg}}{\text{mol}}\frac{\text{m}^2}{\text{s}^2}$ | $\frac{\text{kJ}}{\text{mol}}$                 |
| force            | $F(x) = - \frac{d}{dx}V(x)$  | $\text{u}\frac{\text{nm}}{\text{ps}^2}$   | $10^{12}\frac{\text{kg}}{\text{mol}}\frac{\text{m}}{\text{s}^2}$   | $\frac{\text{kJ}}{\text{mol}\cdot\text{nm}}$   |
| hessian          | $h(x)= \frac{d^2}{dx^2}V(x)$ | $\text{u}\frac{1}{\text{ps}^2}$           | $10^{21}\frac{\text{kg}}{\text{mol}}\frac{1}{\text{s}^2}$          | $\frac{\text{kJ}}{\text{mol}\cdot\text{nm}^2}$ |

## Natural constants

| Symbol | Name               | Value                                                                           | Implementation                                            |
| ------ | ------------------ | ------------------------------------------------------------------------------- | --------------------------------------------------------- |
| $N_A$  | Avogadro constant  | $6.02214076\cdot 10^{23} \, \mathrm{mol}^{-1}$                                  | scipy.constants.Avogadro                                  |
| $R$    | Ideal gas constant | $8.314462618 \cdot 10^{-3}\, \mathrm{kJ}\, \mathrm{mol}^{-1}\, \mathrm{K}^{-1}$ | scipy.constants.R \* 0.001                                |
| $h$    | Planck constant    | $0.399031271\,\mathrm{kJ}\,\mathrm{mol}^{−1}\,\mathrm{ps}$                      | h = scipy.constants.h _ scipy.constants.h _ 0.001 \* 1e12 |
|        |                    |                                                                                 |                                                           |

## System

As system is implemented by the system class, which holds information on

- particle masses
- state of the system, i.e. positions and momenta
- thermodynamic state of the system, i.e. temperature
- parameters for the interaction with the thermal bath, i.e. collision frequency and simulation time step

## One-dimensional systems
