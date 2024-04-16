import numpy as np
import sys

sys.path.append("..")
from potential.D1 import Linear_Potential
from potential.D1 import Quadratic_Potential
from potential.D1 import Double_Well_Potential

def test_linear_pot():

     linear_potential_1= Linear_Potential((1,-1))
     assert  linear_potential_1.potential(7) == -6, "should be -6"
     assert  linear_potential_1.force(1) == 1, "should be 1"
     assert np.isclose(linear_potential_1.force_num(1,0.001),1)

     linear_potential_2 = Linear_Potential((2,1))
     assert  linear_potential_2.potential(1.1) == 3.1, "should be 3.1"
     assert  linear_potential_2.force(0) == -1, "should be -1"
     assert np.isclose(linear_potential_2.force_num(0,0.001),-1)

     linear_potential_3 = Linear_Potential((-1,3.2))
     assert  np.isclose(linear_potential_3.potential(1),2.2)
     assert  np.isclose(linear_potential_3.force(1),-3.2)
     assert np.isclose(linear_potential_3.force_num(1,0.001),-3.2)


def test_quadratic_pot():
     quadratic_potential_1 = Quadratic_Potential((1, 2, 1))
     assert quadratic_potential_1.potential(2) == 9
     assert quadratic_potential_1.force(2) == -6
     assert np.isclose(quadratic_potential_1.force_num(2, 0.001), -6)

     quadratic_potential_2 = Quadratic_Potential((-1,4,5))
     assert quadratic_potential_2.potential(10) == -55
     assert quadratic_potential_2.force(10) == 16
     assert np.isclose(quadratic_potential_2.force_num(10, 0.001), 16)

     quadratic_potential_3 = Quadratic_Potential((1.1,1.2,1))
     assert quadratic_potential_3.potential(0) == 1
     assert np.isclose(quadratic_potential_3.force(0),-1.2)
     assert np.isclose(quadratic_potential_3.force_num(0, 0.001), -1.2)



def test_double_well_pot():
    double_well_potential_1 = Double_Well_Potential((2,3,4))
    assert double_well_potential_1.potential(2) == 24
    assert double_well_potential_1.force(2) == -52
    assert np.isclose(double_well_potential_1.force_num(2, 0.001), -52)

    double_well_potential_2 = Double_Well_Potential((12,-2,10))
    assert double_well_potential_2.potential(1) == 24
    assert double_well_potential_2.force(1) == -52
    assert np.isclose(double_well_potential_2.force_num(1, 0.001), -52)

    double_well_potential_3 = Double_Well_Potential((1,0.2,0.2))
    assert double_well_potential_3.potential(1) == 1
    assert double_well_potential_3.force(1) == -3.6
    assert np.isclose(double_well_potential_3.force_num(1, 0.001), -3.6)

if __name__ == "__main__" :
    test_linear_pot()
    test_quadratic_pot()
    test_double_well_pot()


