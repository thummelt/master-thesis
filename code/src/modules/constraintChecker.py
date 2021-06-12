from src.modules.state import State
from src.modules.decision import Decision
from src.modules import constants as con

import math

def checkState(t: int, b_l: float, v_ta: float, d: float) -> bool:
    # Check termination nodes - vehicle B_L must be equal to beta_t - no demand for trips at t=T - vehicle must be at home
    if (t == con.T) & ( ~(b_l >= con.beta_T) | ~(d == 0.0) | ~(v_ta == 0.0)):
        return False

    # Check starting nodes - vehicles must have initial battery level - must be at home
    if (t == 0) & ( ~(b_l >= con.beta_0) | ~(v_ta == 0.0)):
        return False
    
    return True

def checkDecision(s: State, x: Decision) -> bool:
    code = [0]
    # Maximum G2V amount limited by residual capacity and charging rate
    if not (con.eta*x.get_x_G2V() <= min(con.beta_max-s.get_B_L(), con.my)):
        code += [1]
    
    # Maximum V2G amount limited by residual energy and charging rate
    if not (x.get_x_V2G() <= min(s.get_B_L()-con.beta_min, con.my)):
         code += [2]

    # Either G2V or V2G amounts transmitted
    if not (x.get_x_V2G()*x.get_x_G2V() == 0):
         code += [3]

    # G2V and V2G only possible if vehicle is onsite and no trip starts
    if not( x.get_x_G2V() + x.get_x_V2G() <= con.phi*(1-s.getY())*(1-x.get_x_t())):
         code += [4]
    
    # Starting trip only if vehicle is onsite
    if not( x.get_x_t() <= con.phi*(1-s.getY())):
         code += [5]

    # Starting trip only if demand for trip
    if not( x.get_x_t()  <= con.phi*s.get_D()):
         code += [6]
    
    # Starting trip only if enough energy given (including min level)
    if not( con.ny*s.get_D() <=  s.get_B_L()-con.beta_min + con.phi*(1-x.get_x_t())):
        code += [7]

    # Target energy must be reachable by end of time horizon
    # Current decision considered and for future periods where vehicle is at home max charge rate to fill up battery
    if not(s.B_L + ( con.eta*x.get_x_G2V() + con.my*(max(0,con.T-s.get_t()-(1-x.get_x_t())*1-x.get_x_t()*math.ceil(s.get_D()/con.gamma/con.tau))) - x.get_x_V2G() - con.ny*s.get_D()*x.get_x_t()) >= con.beta_T):
        code += [8]

    # Vehicle cannot start trips if wont be home until end of horizon
    # V_TA is set for t+1 and vehicles drives first time at t
    # => 19.5km decided to start in t=0 -> V_TA = 9.5 in t=1 -> V_TA = 0 in t=2 
    if not(s.t + x.get_x_t()*math.ceil(s.get_D()/con.gamma/con.tau) <= con.T):
        code += [9]

    # if s.getKey() == '(0,5.0,0.0,9.5,0.1,0.1)':
    #     print(x.getKey() + str(list(filter(lambda c: c > 0, code))))

    return code == [0]
    
