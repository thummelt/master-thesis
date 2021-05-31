from src.modules.state import State
from src.modules.decision import Decision
from src.modules import generator as g
from src.modules.solutionAlgorithms import SolutionAlgorithms 
from src.modules.probabilities import Probabilities
from src.modules import constants as con

import pandas as pd
from typing import List, Dict
from itertools import product, chain
import numpy as np



## Sets algorithm up. Called by Jupyter NB

class App:

    # Variables
    sol: SolutionAlgorithms

    ## States as dataframe to speed up lookup
    dfStates = pd.DataFrame(columns=["key", "t", "B_L", "V_TA", "D", "P_B", "P_S", "terminal", "V_N", "obj"])

    decisions : Dict[State, List[Decision]] = {}

    def __init__(self):
        # Solver
        self.sol = SolutionAlgorithms()

        # States
        self.dfStates = g.constructStates()

        # For each state all decisions        
        for s in self.dfStates["obj"]:
            self.decisions[s] = g.constructDecisions(s)  


        # Prune data frame 
        ## States without decisions
        ## states with infeasible Transitions          

    
    def valueIteration(self) -> bool:
        # Prepare data 
        
        p = Probabilities()
        
        # Construct data frame for performance [state, decision, t, key, Trip, Length, Price, p]
        df = pd.DataFrame(chain.from_iterable([list(product([k], v)) for k,v in self.decisions.items()]), columns=["s", "d"])
        df["t"] = df["s"].apply(lambda x: x.get_t())
        df["key"] = df["s"].apply(lambda x: x.get_t())

        # Construct ex_info data frame for all ts
        df_p = pd.DataFrame()

        for t in np.arange(0, con.T, con.tau*60):
            df_p = df_p.append(p.getProbabilities(t))
                        
        df = pd.merge(df,df_p, on=["t"]).copy()
        
        # Call VI with states, state-decision pairs (transitions are derived automatically just to get successor state)
        return df # TODO
        # return self.sol.performStandardVI(df)




    def getStateByKey(self, key : str) -> bool:
        return self.dfStates["key"].isin(key)


