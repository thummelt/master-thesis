from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition
from src.modules.analysis import Analysis
from src.modules.probabilities import Probabilities
from src.modules import constants as con

from typing import List, Dict

import pandas as pd
import numpy as np
import logging

# Represents VI algorithm


class SolutionAlgorithms:

    # Variables
    conv: dict
    epsilon: float

    def __init__(self):
        pass



    def performStandardVI(self, df: pd.DataFrame, states: Dict[str,State]) -> bool:
        
        # Calculate contribution of state-decision pairs
        cont = df[["s_key", "d_key", "s_obj","d_obj"]].drop_duplicates().copy()

        cont["cont"] = cont["s_obj"].apply(lambda s: s.get_P_S())*con.eta*cont["d_obj"].apply(lambda d: d.get_x_v2g())
        - cont["s"].apply(lambda s: s.get_P_B())*con.eta * cont["d_obj"].apply(lambda d: d.get_x_g2v())
        - con.epsilon * cont["s_obj"].apply(lambda s: s.get_D()) * cont["d_obj"].apply(lambda d: d.get_x_t())

        
        iterationCounter: int = 0

        # Loop until epsilon-convergence has been achieved
        while not all([s.hasConverged(0.1) for s in states.values()]):
                iterationCounter += 1

                # For each state-decision-ex_info derive expected future state contribution
                df["ex_cont"] = df["p"]*states[df["s_d_key"]].get_V_N()
                
                # Aggregate values per stat-decision pair 
                grp_sum = df[["s_key","d_key","ex_cont"]].groupby(["s_key","d_key"])["ex_cont"].sum().reset_index()

                # Merge with contribution dataframe
                grp_sum = pd.merge(grp_sum, cont, on=["s_key","d_key"])

                # Add contribution by decision to contribution of future states
                grp_sum["cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_con"]

                # Select max aggregate per state (decision with best contribution)
                max_con = grp_sum.groupby("s_key").max()

                # TODO later store best decision
                # st_best_dec =grp_sum.groupby("s").idxmax()

                # Update state values as sum of total_contributions
                for s, v in max_con.iteritems():
                        states[s].set_V_N(v)


        def performApproxVI(self) -> bool:
                pass

        def performMyopicOpti(self) -> bool:
                pass

        def performRandomDeicison(self) -> bool:
                pass