from src.modules.state import State
from src.modules.decision import Decision
from src.modules import generator as g

import pandas as pd
from typing import List, Dict



## Sets algorithm up. Called by Jupyter NB

class App:

    # Variables

    ## States as dataframe to speed up lookup
    dfStates = pd.DataFrame(columns=["key", "t", "B_L", "V_TA", "D", "P_B", "P_S", "terminal", "V_N", "obj"])

    decisions : Dict[State, List[Decision]] = {}

    def __init__(self):
        # States
        self.dfStates = g.constructStates()

        # For each state all decisions        
        for s in self.dfStates["obj"]:
            self.decisions[s] = g.constructDecisions(s)  

        # Filter out states without transitions          


        # Call VI with states, state-decision pairs (transitions are derived automatically just to get successor state)



    def getStateByKey(self, key : str) -> bool:
        return self.dfStates["key"].isin(key)


