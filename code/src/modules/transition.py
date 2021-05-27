from src.modules.state import State
from src.modules.decision import Decision
from src.modules.app import App
from src.modules import constants as con

import pandas as pd
import math

## Represents transition object

class Transition:

    # Const
    App : App

    # Variables
    p : float
    s_s : State
    s_d : State
    x : Decision
    # Information that is known after making decision => integrated in destination state
    ex_info : pd.DataFrame()


    def __init__(self, app: App, s_s: State, x: Decision, p: float, exInfo : pd.DataFrame):
        self.App = app
        self.s_s = s_s
        self.x = x
        self.p = p
        self.ex_info = exInfo

        # Calculate transition to to destination state
        self.performTransition()

        # Check if valid destination state is reached - if not set s_d to None
        #if self.checkDestState() == False:
        #    self.s_d = None



    def performTransition(self):
        
        # t
        t = self.s_s.get_t()+1

        # Battery Load
        b_l = self.s_s.get_B_L() + con.eta*(self.x.get_x_G2V()-self.x.get_x_V2G()) - con.ny*con.gamma*con.tau*self.s_s.getY()
        print((self.x.get_x_V2G()))

        # Time until arrival
        v_ta = self.s_s.getY()*(self.s_s.get_V_TA()-1) + self.x.get_x_t()*math.ceil(self.s_s.get_D()/con.gamma/con.tau)

        # Copy exogenous information from exInfo
        d = self.ex_info["Length"]
        p_b = self.ex_info["Price"]
        p_s = self.ex_info["Price"] # Todo later distinguish buy and sell

        self.s_d = State(t,b_l,v_ta,d,p_b,p_s)

    
    def checkDestState(self) -> bool:
        return self.App.getStateByKey(self.s_d.getKey())

    def __str__(self) -> str:
        return "Transition [(%s) -> (%s) & (p=%f | %s) => (%s)]" % ( self.s_s.__str__(), self.x.__str__(), self.p, self.ex_info.iloc[0,:].to_string(),  self.s_d.__str__())



    #######################################################################################
    # Getter and setter                                                                   #
    #######################################################################################


    def get_s_s(self) -> State:
        return self.s_s

    def get_s_d(self) -> State:
        return self.s_d

    def get_x(self) -> Decision:
        return self.x
    
    def get_ex_info(self) -> pd.DataFrame:
        return self.ex_info


    def __eq__ (self, t) -> bool:
        return ((self.get_s_d().__eq__(t.get_s_d())) and (self.get_s_s().__eq__(t.get_s_s())) and (self.get_x().__eq__(t.get_x())) and (self.get_ex_info().equals(t.get_ex_info())))