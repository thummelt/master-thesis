from src.modules.state import State
from src.modules.decision import Decision
from src.modules import constants as con

from typing import List

import pandas as pd
import math

## Represents transition object

class Transition:

    # Variables
    p : float
    s_s : State
    s_d : State
    x : Decision
    # Information that is known after making decision => integrated in destination state
    trpln: float
    prc: float


    def __init__(self, s_s: State, x: Decision, p: float, trpln: float, prc: float):
        self.s_s = s_s
        self.x = x
        self.p = p
        self.trpln = trpln
        self.prc = prc

        # Calculate transition to to destination state
        self.s_d = self.performTransition(self.s_s,  self.x, self.p, self.trpln, self.prc)

 

    def performTransition(self, s_s: State, x: Decision, p: float, trpln: float, prc: float) -> State:
        
        # t
        t = s_s.get_t()+1

        # Battery Load
        b_l = round(s_s.get_B_L() + con.eta*(x.get_x_G2V()-x.get_x_V2G()) - con.ny*con.gamma*con.tau*s_s.getY(),2)

        # Time until arrival
        v_ta = s_s.getY()*(s_s.get_V_TA()-1) + x.get_x_t()*math.ceil(s_s.get_D()/con.gamma/con.tau)

        # Copy exogenous information from exInfo
        d = trpln
        p_b = prc
        p_s = prc # TODO later distinguish buy and sell

        return State(t,b_l,v_ta,d,p_b,p_s)

    


    def __str__(self) -> str:
        return "Transition [(%s) -> (%s) & (p=%f | %s, %s) => (%s)]" % ( self.s_s.__str__(), self.x.__str__(), self.p, self.trpln, self.prc, self.s_d.__str__())

    def getKey(self) -> str:
        return "(%s,%s,%f,%f,%f)" % ( self.s_s.getKey(), self.x.getKey(), self.p, self.trpln, self.prc) 


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

    def get_prc(self) -> float:
        return self.prc


    def __eq__ (self, t) -> bool:
        return ((self.get_s_d().__eq__(t.get_s_d())) and (self.get_s_s().__eq__(t.get_s_s())) and (self.get_x().__eq__(t.get_x())) and (self.trpln == t.trpln) and (self.prc == t.prc))