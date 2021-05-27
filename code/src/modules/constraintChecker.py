from src.modules.state import State
from src.modules.decision import Decision
from src.modules import constants as con

import math

def checkState(t, b_l, v_ta, d) -> bool:
    # Check termination nodes - vehicle B_L must be equal to beta_t - no demand for trips at t=T - vehicle must be at home
    if (t == con.T) & ( ~(b_l >= con.beta_T) | ~(d == 0) | ~(v_ta == 0)):
        return False
    
    return True

def checkDecision(s: State, x: Decision) -> bool:
    # TODO integrate return codes as reason for excluding => testings
    # Maximum G2V amount limited by residual capacity and charging rate
    if not (con.eta*x.get_x_G2V() <= min(con.beta_max-s.get_B_L(), con.my)):
        print(1)
        return False
    
    # Maximum V2G amount limited by residual energy and charging rate
    if not (con.eta*x.get_x_V2G() <= min(s.get_B_L()-con.beta_min, con.my)):
        print(2)
        return False

    # Either G2V or V2G amounts transmitted
    if not (x.get_x_V2G()*x.get_x_G2V() == 0):
        print(3)
        return False

    # G2V and V2G only possible if vehicle is onsite and no trip starts
    if not( x.get_x_G2V() + x.get_x_V2G() <= con.phi*(1-s.getY())*(1-x.get_x_t())):
        print(4)
        return False
    
    # Starting trip only if vehicle is onsite
    if not( x.get_x_t() <= con.phi*(1-s.getY())):
        print(5)
        return False

    # Starting trip only if demand for trip
    if not( x.get_x_t()  <= con.phi*s.get_D()):
        print(6)
        return False
    
    # Starting trip only if enough energy given
    if not( con.ny*s.get_D() <=  s.get_B_L() + con.phi*(1-x.get_x_t())):
        print(7)
        return False

    # Target energy must be met if end of time horizon reached
    if (con.T-1 == s.get_t()) & ~(s.B_L + (x.get_x_G2V() - x.get_x_V2G()) >= con.beta_T):
        print(8)
        return False

    # Vehicle cannot start trips if wont be home until end of horizon
    if not(s.t + x.get_x_t()*math.ceil(s.get_D()/con.gamma/con.tau) <= con.T):
        print(9)
        return False

    return True
    
