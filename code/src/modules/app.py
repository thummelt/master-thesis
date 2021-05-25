from modules.state import State
from modules.decision import Decision
from modules import constants as con
from modules.constraintChecker import checkDecision

import pandas as pd
import numpy as np
from typing import List, Dict



## Sets algorithm up. Called by Jupyter NB

class App:

    # Variables
    states = pd.DataFrame(columns=["key", "t", "B_L", "D", "P_B", "P_S", "terminal", "v_n"])

    states: List[State] = []
    decisions : Dict[State, List[Decision]] = {}

    def __init__(self):
        # States
        self.states = self.constructStates()

        # For each state all decisions        
        for s in self.states:
            self.decisions[s] = self.constructDecisions(s)            


        # Call VI with states, state-decision pairs (transitions are derived automatically just to get successor state)

    def constructStates(self) -> List[State]:
        ## {10..50}*{0..max_trip/speed}*{0..max_trip}*{0..max_preis_b}*{0..max_preis_s}
        return [State(1, 2, 25, 10, 5, 5), State(1, 0, 25, 10, 5, 5), State(1, 0, 25, 0, 5, 5)]
    

    def constructDecisions(self, s:State) -> List[Decision]:
        ## {0..my}*{0..my}*{0,1} -> {0}*{0..my}*{0,1} and {0..my}*{0}*{0,1}

        df1 = pd.DataFrame({"x_G2V": list(np.arange(0.0,con.my, con.step_en))})
        df2 = pd.DataFrame({"x_V2G": list(np.arange(0.0,con.my, con.step_en))})
        df3 = pd.DataFrame({"x_trip": [0,1]}) if s.getY() == 0 else pd.DataFrame({"x_trip": [0]})

        df1["x_V2G"] = 0
        df2["x_G2V"] = 0

        df1["key"] = 0
        df2["key"] = 0
        df3["key"] = 0
        
        df = pd.concat([pd.merge(df1, df3, on=["key"]).copy(), pd.merge(df2, df3, on=["key"]).copy()], ignore_index=True)
        df.drop_duplicates(ignore_index=True, inplace=True)

        # Create decision objects and filter out invalid decisions
        ls = list(filter(lambda d: checkDecision(s,d), [Decision(a.x_G2V, a.x_V2G, a.x_trip) for a in df.itertuples()]))

        return ls


    def getStateByKey(self, key : str) -> bool:
        return str in self.states["key"]


