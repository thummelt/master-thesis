from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition
from src.modules.analysis import Analysis
from src.modules.probabilities import Probabilities
from src.modules import constants as con
from src.modules import generator as g

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

    def performStandardVI(self, df: pd.DataFrame, states: Dict[str, State]) -> bool:
        """Perform standard value iteration

        Args:
            df (pd.DataFrame): state-decision-transition combinations 
            states (Dict[str, State]): Key-Value pairs of state key and state obkect

        """        
        # Calculate contribution of state-decision pairs
        cont = df[["s_key", "d_key", "s_obj", "d_obj"]].drop_duplicates(subset=["s_key", "d_key"]).copy()

        cont["cont"] = cont["s_obj"].apply(lambda s: s.get_P_S())*con.eta*cont["d_obj"].apply(lambda d: d.get_x_V2G()) \
            - cont["s_obj"].apply(lambda s: s.get_P_B())*con.eta * cont["d_obj"].apply(lambda d: d.get_x_G2V()) \
            - con.epsilon * \
            cont["s_obj"].apply(lambda s: s.get_D()) * \
            cont["d_obj"].apply(lambda d: 1-d.get_x_t())

        cont["cont"] = cont["cont"].astype(float)

        iterationCounter: int = 0

        best_dec = pd.DataFrame(columns=["s_key", "d_key", "value"])

        # Loop until epsilon-convergence has been achieved
        while not all([states[s].hasConverged(con.convergence) for s in df["s_key"]]):
            iterationCounter += 1

            logging.debug("Current iteration %d" % iterationCounter)
            logging.debug(sum([states[s].hasConverged(0.1)
                               for s in df["s_key"].unique()]))
            logging.debug("Max diff %s" % str(max({s: abs(states[s].get_V_N(
            )-states[s].get_V_N_1()) for s in df["s_key"].unique()}.items(), key=operator.itemgetter(1))))

            # For each state-decision-ex_info derive expected future state contribution
            df["ex_cont"] = df["p"] * df["s_d_key"].apply(lambda s: states.get(s).get_V_N())
            # df.to_excel("/usr/app/output/vi_ex_cont_%s.xlsx" % str(iterationCounter))

            # Aggregate values per state-decision pair
            grp_sum = df[["s_key", "d_key", "ex_cont"]].groupby(["s_key", "d_key"])["ex_cont"].sum().reset_index()
            # grp_sum.to_excel("/usr/app/output/vi_grp_sum_%s.xlsx" % str(iterationCounter))

            # Merge with contribution dataframe
            grp_sum = pd.merge(grp_sum, cont[["s_key", "d_key", "cont"]], on=["s_key", "d_key"])

            # Add contribution by decision to contribution of future states
            grp_sum["tot_cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_cont"]
            # grp_sum.to_excel("/usr/app/output/vi_grp_sum_total_%s.xlsx" % str(iterationCounter))

            # Select max aggregate per state (decision with best contribution)
            max_con = grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].max()

            # Store best decision
            best_dec = grp_sum.loc[grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].idxmax(), ["s_key", "d_key"]]

            # Update state values as sum of total_contributions
            for s, v in max_con.items():
                states[s].set_V_N(v)

        best_dec["value"] = best_dec["s_key"].apply(lambda s: states[s].get_V_N())
        best_dec = best_dec.append(pd.DataFrame({
            "s_key": list(map(lambda s: s.getKey(), filter(lambda s: s.get_isTerminal(), states.values()))),
            "value": list(map(lambda s: s.get_V_N(), filter(lambda s: s.get_isTerminal(), states.values())))},
        ), ignore_index=True)

        best_dec.to_excel("/usr/app/output/xlsx/vi_best_decisions.xlsx")


    def performMyopic(self, df: pd.DataFrame, states: Dict[str, State]):
        """Myopic optimization by maximizing contribution of current decision

        # Construct all states 
        # Construct all decisions
        # Choose decision leading to highest contribution

        Args:
            df (pd.DataFrame): DF with columns ["s_key", "d_key", "s_obj", "d_obj"]
        """    

        # Calculate contribution of state-decision pairs
        cont = df.copy()

        cont["cont"] = cont["s_obj"].apply(lambda s: s.get_P_S())*con.eta*cont["d_obj"].apply(lambda d: d.get_x_V2G()) \
            - cont["s_obj"].apply(lambda s: s.get_P_B())*con.eta * cont["d_obj"].apply(lambda d: d.get_x_G2V()) \
            - con.epsilon * \
            cont["s_obj"].apply(lambda s: s.get_D()) * \
            cont["d_obj"].apply(lambda d: 1-d.get_x_t())

        cont["cont"] = cont["cont"].astype(float)

        best_dec = pd.DataFrame(columns=["s_key", "d_key", "value"])

        # Select max contrivution per state (decision with best contribution)
        max_con = cont.groupby(["s_key"])["cont"].max()

        # Store best decision
        best_dec = cont.loc[cont.groupby(["s_key"])["cont"].idxmax(), ["s_key", "d_key"]]

        # Update state values as sum of total_contributions
        for s, v in max_con.items():
            states[s].set_V_N(v)

        best_dec["value"] = best_dec["s_key"].apply(lambda s: states[s].get_V_N())
        best_dec = best_dec.append(pd.DataFrame({
            "s_key": list(map(lambda s: s.getKey(), filter(lambda s: s.get_isTerminal(), states.values()))),
            "value": list(map(lambda s: s.get_V_N(), filter(lambda s: s.get_isTerminal(), states.values())))},
        ), ignore_index=True)

        best_dec.to_excel("/usr/app/output/xlsx/myopic_best_decisions.xlsx")
        

    def performApproxVI(self, df: pd.DataFrame, states: Dict[str, State], p: Probabilities, iniState: str, n: int, samples: int = 20):
        """Value Iteration with different data frame (only partial transitions given) and different state value update function

        Args:
            df (pd.DataFrame): state-decision combinations (transitations are sampled in algorithm)
            states (Dict[str, State]): Key-Value pairs of state-key and state-object (all states)
            iniState (State): Initial state
            n (int): Number of iterations (= number of sample paths created. )
        """        

        # Initalize values of states by their max contribution of decision
        ## Calculate contribution for state decision pairs
        logging.debug("Calculating max possible contribution by decisions per state")
        cont = df[["s_key", "d_key", "s_obj", "d_obj"]].drop_duplicates(subset=["s_key", "d_key"]).copy()

        cont["cont"] = cont["s_obj"].apply(lambda s: s.get_P_S())*con.eta*cont["d_obj"].apply(lambda d: d.get_x_V2G()) \
        - cont["s_obj"].apply(lambda s: s.get_P_B())*con.eta * cont["d_obj"].apply(lambda d: d.get_x_G2V()) \
        - con.epsilon * \
        cont["s_obj"].apply(lambda s: s.get_D()) * \
        cont["d_obj"].apply(lambda d: 1-d.get_x_t())

        cont["cont"] = cont["cont"].astype(float)

        #cont.to_excel("/usr/app/output/xlsx/avi_cont.xlsx")
        
        ## Store new value
        logging.debug("Updating state values with calculated contribution")
        for s, v in cont[["s_key", "d_key", "cont"]].groupby(["s_key"])["cont"].max().items():
            states[s].set_V_N(v)

    	
        # Storing key! not object
        cState = iniState

        iterationCounter: int = 0

        best_dec = pd.DataFrame(columns=["s_key", "d_key", "value"])

        logging.debug("Starting algorithm loop")

        # Loop until epsilon-convergence has been achieved
        while iterationCounter <= n:

            iterationCounter += 1
            logging.debug("Current iteration %d" % iterationCounter)

            # Create sample of exog information at t for current state
            df_exo = g.constructExogInfoT(df.loc[df.s_key == cState,:], p, iterationCounter-1, samples) # TODO evt abaenerung auf zusaetzlichhge schleife für t
            df_exo.to_excel("/usr/app/output/xlsx/avi_df1_%s.xlsx" % str(iterationCounter))

            # Perform transitions
            df_trans = g.constructTransitions(df_exo, states)
            df_trans.to_excel("/usr/app/output/xlsx/avi_df2_%s.xlsx" % str(iterationCounter))
            
            # Caclulate expected contribution
            df_trans["ex_cont"] = df_trans["p"] * df_trans["s_d_key"].apply(lambda s: states.get(s).get_V_N())

            # Calculate expected total contribution
            grp_sum = df_trans[["s_key", "d_key", "ex_cont"]].groupby(["s_key", "d_key"])["ex_cont"].sum().reset_index()
            grp_sum = pd.merge(grp_sum, cont[["s_key", "d_key", "cont"]], on=["s_key", "d_key"])
            grp_sum["tot_cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_cont"]
            grp_sum.to_excel("/usr/app/output/xlsx/avi_grp_%s.xlsx" % str(iterationCounter))
            
            # Select decision maximizing expected total contribution
            max_con = grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].max()

            # Store best decision
            best_dec = best_dec.append(grp_sum.loc[grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].idxmax(), ["s_key", "d_key"]], ignore_index=True)
            
            # Update state value
            states[cState].set_V_N_AVI(max_con.loc[cState])

            # Perform transition to next state
            logging.debug("Best decision for state %s is  %s" % (cState, best_dec.loc[best_dec.s_key == cState, "d_key"].iloc[0]))

            df_best_dec = df_trans.loc[(df_trans.s_key == cState) & (df_trans.d_key == best_dec.loc[best_dec.s_key == cState, "d_key"].iloc[0])].reset_index(drop=True)
            df_best_dec.to_excel("/usr/app/output/xlsx/avi_df3_%s.xlsx" % str(iterationCounter))
            trans = df_best_dec.loc[np.random.random_integers(0, len(df_best_dec.index)),:]
            logging.debug("Choosing randomly the following transition %s with probability %f" % (trans[["trpstrt", "trpln", "prc_b", "prc_s"]].to_string(), trans["p"]))
            cState = trans["s_d_key"]
            logging.debug("Next state is %s" % cState)


        logging.debug("Finished. Starting to spool out best decisions")

        best_dec["value"] = best_dec["s_key"].apply(lambda s: states[s].get_V_N())
        best_dec = best_dec.append(pd.DataFrame({
            "s_key": list(map(lambda s: s.getKey(), filter(lambda s: s.get_isTerminal(), states.values()))),
            "value": list(map(lambda s: s.get_V_N(), filter(lambda s: s.get_isTerminal(), states.values())))},
        ), ignore_index=True)


        best_dec.to_excel("/usr/app/output/xlsx/avi_best_decisions.xlsx")

    
    def performApproximateDP():
        pass
        
        # Assign starting state

        # For n in 1...N:
        ## For t in 0...T:
        ### Get decisions for state
        ### Get transitions for t
        ### Choose decision maximizing current and expected contribution
        ### Calculate value for decision and expected contribution
        ### Store value for current state
        ### Compute next state by sampling from possible transitions


