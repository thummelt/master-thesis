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
import datetime
import operator
import openpyxl

# Represents VI algorithm


class SolutionAlgorithms:

    # Variables
    conv: dict
    epsilon: float

    def __init__(self):
        logging.basicConfig(filename="logs" + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '_app.log',
                        filemode='w+', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')




    def performStandardVI(self, df: pd.DataFrame, states: Dict[str,State]) -> bool:
        
        # Calculate contribution of state-decision pairs
        cont = df[["s_key", "d_key", "s_obj","d_obj"]].drop_duplicates(subset=["s_key", "d_key"]).copy()

        cont["cont"] = cont["s_obj"].apply(lambda s: s.get_P_S())*con.eta*cont["d_obj"].apply(lambda d: d.get_x_V2G()) \
        - cont["s_obj"].apply(lambda s: s.get_P_B())*con.eta * cont["d_obj"].apply(lambda d: d.get_x_G2V()) \
        - con.epsilon * cont["s_obj"].apply(lambda s: s.get_D()) * cont["d_obj"].apply(lambda d: 1-d.get_x_t())

        cont["cont"] = cont["cont"].astype(float)

        
        iterationCounter: int = 0

        best_dec = pd.DataFrame(columns=["s_key", "d_key", "value"])

        # Loop until epsilon-convergence has been achieved
        while not all([states[s].hasConverged(0.1) for s in df["s_key"]]):
                
                iterationCounter += 1

                #logging.info("Current iteration %d" % iterationCounter)
                logging.debug("Current iteration %d" % iterationCounter)
                logging.debug(sum([states[s].hasConverged(0.1) for s in df["s_key"].unique()]))
                logging.debug("Max diff %s" % str(max({s: abs(states[s].get_V_N()-states[s].get_V_N_1()) for s in df["s_key"].unique()}.items(), key=operator.itemgetter(1))))


                # For each state-decision-ex_info derive expected future state contribution
                df["ex_cont"] = df["p"]*df["s_d_key"].apply(lambda s: states.get(s).get_V_N())
                
                # Aggregate values per stat-decision pair 
                grp_sum = df[["s_key","d_key","ex_cont"]].groupby(["s_key","d_key"])["ex_cont"].sum().reset_index()

                # Merge with contribution dataframe
                grp_sum = pd.merge(grp_sum, cont[["s_key","d_key", "cont"]], on=["s_key","d_key"])

                # Add contribution by decision to contribution of future states
                grp_sum["tot_cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_cont"]

                # Select max aggregate per state (decision with best contribution)
                max_con = grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].max()

                # Store best decision
                best_dec = grp_sum.loc[grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].idxmax(),["s_key", "d_key"]]

                # Update state values as sum of total_contributions
                for s, v in max_con.items():
                        states[s].set_V_N(v)
        
        best_dec["value"] = best_dec["s_key"].apply(lambda s: states[s].get_V_N())
        best_dec = best_dec.append(pd.DataFrame({ \
                                                        "s_key": list(map(lambda s: s.getKey(), filter(lambda s: s.get_isTerminal(), states.values()))), \
                                                        "value": list(map(lambda s: s.get_V_N(), filter(lambda s: s.get_isTerminal(), states.values())))}, \
                        ), ignore_index=True)
        best_dec.to_excel("/usr/app/output/vi_best_decisions.xlsx")


        def performApproxVI(self) -> bool:
                pass

        def performMyopicOpti(self) -> bool:
                pass

        def performRandomDeicison(self) -> bool:
                pass