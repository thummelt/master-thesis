from modules.state import State
from modules.decision import Decision
from modules.app import App

import pandas as pd

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

        # Check if valid destination state is reached
        if self.checkDestState():
            pass
        else:
            pass


    def performTransition(self):
        # t

        # Battery Load

        # Time until arrival

        # Copy exogenous information from exInfo

        pass

    
    def checkDestState(self) -> bool:
        return self.App.getStateByKey(self.s_d.getKey())

    def __str__(self) -> str:
        return "Transition [(%s) -> (%s) -> (%s)]" % ( self.s_s.__str__(), self.x.__str__(), self.s_d.__str__())



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