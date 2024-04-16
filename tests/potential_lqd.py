import sys
sys.path.append("..")
from potential.D1 import Linear_Potential
from potential.D1 import Quadratic_Potential
from potential.D1 import Double_Well_Potential

linear_potential = Linear_Potential((1,-1))
assert  linear_potential.potential(7) == -6, "should be -6"
linear_potential = Linear_Potential((1,1))
assert  linear_potential.potential(1.1) == 2.1, "should be 2.1"
linear_potential = Linear_Potential((-0.1,1.2))
assert  linear_potential.potential(2) == 2.3, "should be 2.3"


quadratic_potential = Quadratic_Potential((1,2,1))
assert quadratic_potential.potential(2) == 9
quadratic_potential = Quadratic_Potential((-1,4,5))
assert quadratic_potential.potential(10) == -55


double_well_dotential = Double_Well_Potential((2,3,4))
assert double_well_dotential.potential(1) == 9


