from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition
from src.modules.analysis import Analysis
from src.modules.probabilities import Probabilities
from src.modules import constants as con

from typing import List, Dict

import pandas as pd
import numpy as np

# Represents VI algorithm


class SolutionAlgorithms:

    # Variables
    conv: dict
    epsilon: float

    def __init__(self):
        pass



    def performStandardVI(self, df: pd.DataFrame) -> bool:
        
        # States 
        states = df["s"].drop_duplicates.copy()

        # Calculate contribution of state-decision pairs
        cont = df[["s","d"]].drop_duplicates().copy()
        cont["cont"] = cont["s"].apply(lambda s: s.get_P_S())*con.eta*cont["d"].apply(lambda d: d.get_x_v2g())
        - cont["s"].apply(lambda s: s.get_P_B())*con.eta * cont["d"].apply(lambda d: d.get_x_g2v())
        - con.epsilon * cont["s"].apply(lambda s: s.get_D()) * cont["d"].apply(lambda d: d.get_x_t())

        cont["s_key"] = cont["s"].apply(lambda x: x.getKey())



        iterationCounter: int = 0

        # Loop until epsilon-convergence has been achieved
        while not all([s.hasConverged(0.1) for s in states]):
                iterationCounter += 1

                # For each state-decision-ex_info perform transition
                for row in df.iterrows():
                        t = Transition(row["s"], row["d"], row["p"], row[["Trip", "Length", "Price"]])
                        s_d = t.get_s_d()

                        # Check if valid state reached by transition
                        if not s_d.getKey() in df["s_key"]:
                                print("%s has been reached by %s but is not valid" %(s_d.__str__(), t.__str__() ))
                                break

                        # Calculate total contribution
                        row["ex_cont"] = row["p"]*s_d.get_V_N()
                
                # Aggregate values per stat-decision pair 
                grp_sum = df[["s","d","ex_cont"]].groupby(["s","d"])["ex_cont"].sum().reset_index()

                # Merge with contribution dataframe
                grp_sum = pd.merge(grp_sum, cont, on=["s","d"])

                # Add contribution by decision to contribution of future states
                grp_sum["cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_con"]

                # Select max aggregate per state (decision with best contribution)
                max_con = grp_sum.groupby("s").max()

                # TODO later store best decision
                # st_best_dec =grp_sum.groupby("s").idxmax()

                # Update state values as sum of total_contributions
                for s, v in max_con.iteritems():
                        s.set_V_N(v)


