from src.modules.state import State
from src.modules.decision import Decision
from src.modules import constants as con

from typing import List

import pandas as pd
import math

## Represents transition object

def performTransition(s_s: State, x: Decision, trpln: float, prc_b: float, prc_s: float) -> State:
    
    # t
    t = s_s.get_t()+1

    # Battery Load
    b_l = round(s_s.get_B_L() + con.eta*x.get_x_G2V() - x.get_x_V2G() - con.ny*x.get_x_t()*min(s_s.get_D(), con.gamma*con.tau) - con.ny*min(s_s.get_V_TA(),con.gamma*con.tau)*s_s.getY(),1)

    # Kilometre until arrival
    v_ta = max(0.0,s_s.getY()*(s_s.get_V_TA()-con.gamma*con.tau) + x.get_x_t()*(s_s.get_D()-con.gamma*con.tau))

    # Copy exogenous information from exInfo
    d = float(trpln)
    p_b = prc_b
    p_s = prc_s

    return State(t,b_l,v_ta,d,p_b,p_s)

class Transition:

    # Variables
    p : float
    s_s : State
    s_d : State
    x : Decision
    # Information that is known after making decision => integrated in destination state
    trpln: float
    prc_b: float
    prc_s: float


    def __init__(self, s_s: State, x: Decision, p: float, trpln: float, prc_b: float, prc_s: float):
        self.s_s = s_s
        self.x = x
        self.p = p
        self.trpln = trpln
        self.prc_b = prc_b
        self.prc_s = prc_s

        # Calculate transition to to destination state
        self.s_d = self.performTransition(self.s_s,  self.x, self.trpln, self.prc_b, self.prc_s)

 

    def performTransition(self, s_s: State, x: Decision, trpln: float, prc_b: float, prc_s: float) -> State:
        
        # t
        t = s_s.get_t()+1

        # Battery Load
        b_l = round(s_s.get_B_L() + con.eta*x.get_x_G2V() - x.get_x_V2G() - round(con.ny*x.get_x_t()*min(s_s.get_D(), con.gamma*con.tau),1) - round(con.ny*min(s_s.get_V_TA(),con.gamma*con.tau)*s_s.getY(),1),1)

        # Kilometre until arrival
        v_ta = max(0.0,s_s.getY()*(s_s.get_V_TA()-con.gamma*con.tau) + x.get_x_t()*(s_s.get_D()-con.gamma*con.tau))

        # Copy exogenous information from exInfo
        d = float(trpln)
        p_b = prc_b
        p_s = prc_s

        return State(t,b_l,v_ta,d,p_b,p_s)

    


    def __str__(self) -> str:
        return "Transition [(%s) -> (%s) & (p=%f | %s, %s, %s) => (%s)]" % ( self.s_s.__str__(), self.x.__str__(), self.p, self.trpln, self.prc_b, self.prc_s, self.s_d.__str__())

    def getKey(self) -> str:
        return "(%s,%s,%f,%f,%f,%f)" % ( self.s_s.getKey(), self.x.getKey(), self.p, self.trpln, self.prc_b, self.prc_s) 


    #######################################################################################
    # Getter and setter                                                                   #
    #######################################################################################


    def get_s_s(self) -> State:
        return self.s_s

    def get_s_d(self) -> State:
        return self.s_d

    def get_x(self) -> Decision:
        return self.x
    
    def get_trpln(self) -> float:
        return self.trpln

    def get_prc_b(self) -> float:
        return self.prc_b

    def get_prc_s(self) -> float:
        return self.prc_s


    def __eq__ (self, t) -> bool:
        return ((self.get_s_d().__eq__(t.get_s_d())) and (self.get_s_s().__eq__(t.get_s_s())) and (self.get_x().__eq__(t.get_x())) and (self.trpln == t.trpln) and (self.prc_b == t.prc_b)  and (self.prc_s == t.prc_s))