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
from pathlib import Path

# Represents VI algorithm


class SolutionAlgorithms:

    # Variables
    conv: dict
    epsilon: float

    def __init__(self):
        logging.basicConfig(filename="logs" + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '_app.log',
                            filemode='w+', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    def performStandardVI(self, df: pd.DataFrame, states: Dict[str, State], key: str  ="") -> bool:
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
            # df.to_excel("/usr/app/output/xlsx/01_VI/vi_ex_cont_%s.xlsx" % str(iterationCounter))

            # Aggregate values per state-decision pair
            grp_sum = df[["s_key", "d_key", "ex_cont"]].groupby(["s_key", "d_key"])["ex_cont"].sum().reset_index() 
            # grp_sum.to_excel("/usr/app/output/xlsx/01_VI/vi_grp_sum_%s.xlsx" % str(iterationCounter))

            # Merge with contribution dataframe
            grp_sum = pd.merge(grp_sum, cont[["s_key", "d_key", "cont"]], on=["s_key", "d_key"])

            # Add contribution by decision to contribution of future states
            grp_sum["tot_cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_cont"]
            # grp_sum.to_excel("/usr/app/output/xlsx/01_VI/vi_grp_sum_total_%s.xlsx" % str(iterationCounter))

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

        best_dec.to_excel("/usr/app/output/xlsx/01_VI/%s-best_decisions.xlsx" % key)
        best_dec.to_pickle("/usr/app/output/df/%s-best_decisions.xlsx" % key)


    def performMyopic(self, df: pd.DataFrame, states: Dict[str, State], key: str = ""):
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

        best_dec.to_excel("/usr/app/output/xlsx/02_MO/%s-best_decisions.xlsx" % key) 
        best_dec.to_pickle("/usr/app/output/df/%s-best_decisions.pkl" % key)
        

    def performApproxVI(self, df: pd.DataFrame, states: Dict[str, State], p: Probabilities, iniState: str, n: int, samples: int = 20, key: str = ""):
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

        #cont.to_excel("/usr/app/output/xlsx/03_AVI/avi_cont.xlsx")
        
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
        while iterationCounter < n:

            iterationCounter += 1
            logging.debug("Current iteration %d of %d" % (iterationCounter, n))

            # Create sample of exog information at t for current state
            df_exo = g.constructExogInfoT(df.loc[df.s_key == cState,:], p, iterationCounter-1, samples) # TODO evt abaenerung auf zusaetzlichhge schleife für t -> nicht mehr iteration counter nutzen
            #df_exo.to_excel("/usr/app/output/xlsx/03_AVI/avi_df1_%s.xlsx" % str(iterationCounter))

            # Perform transitions
            df_trans = g.constructTransitions(df_exo, states)
            #df_trans.to_excel("/usr/app/output/xlsx/03_AVI/avi_df2_%s.xlsx" % str(iterationCounter))
            
            # Caclulate expected contribution
            df_trans["ex_cont"] = df_trans["p"] * df_trans["s_d_key"].apply(lambda s: states.get(s).get_V_N())

            # Calculate expected total contribution
            grp_sum = df_trans[["s_key", "d_key", "ex_cont"]].groupby(["s_key", "d_key"])["ex_cont"].sum().reset_index()
            grp_sum = pd.merge(grp_sum, cont[["s_key", "d_key", "cont"]], on=["s_key", "d_key"])
            grp_sum["tot_cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_cont"]
            #grp_sum.to_excel("/usr/app/output/xlsx/03_AVI/avi_grp_%s.xlsx" % str(iterationCounter))
            
            # Select decision maximizing expected total contribution
            max_con = grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].max()

            # Store best decision
            best_dec = best_dec.append(grp_sum.loc[grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].idxmax(), ["s_key", "d_key"]], ignore_index=True)
            
            # Update state value
            states[cState].set_V_N_AVI(max_con.loc[cState])

            # Perform transition to next state
            logging.debug("Best decision for state %s is  %s" % (cState, best_dec.loc[best_dec.s_key == cState, "d_key"].iloc[0]))

            df_best_dec = df_trans.loc[(df_trans.s_key == cState) & (df_trans.d_key == best_dec.loc[best_dec.s_key == cState, "d_key"].iloc[0])].reset_index(drop=True)
            #df_best_dec.to_excel("/usr/app/output/xlsx/03_AVI/avi_df3_%s.xlsx" % str(iterationCounter))
            trans = df_best_dec.loc[np.random.random_integers(0, len(df_best_dec.index)),:]
            logging.debug("Choosing randomly the following transition %s with probability %f" % (trans[["trpstrt", "trpln", "prc_b", "prc_s"]].to_string(), trans["p"]))
            cState = trans["s_d_key"]
            logging.debug("Next state is %s" % cState)


        logging.debug("Finished. Starting to spool out best decisions")        
        
        best_dec = best_dec.append(cont.loc[cont.loc[~cont.s_key.isin(best_dec.s_key)].groupby(["s_key"])["cont"].idxmax(), ["s_key", "d_key"]])
        best_dec["value"] = best_dec["s_key"].apply(lambda s: states[s].get_V_N())

        best_dec.to_excel("/usr/app/output/xlsx/03_AVI/%s-best_decisions.xlsx" % key)
        best_dec.to_pickle("/usr/app/output/df/%s-best_decisions.pkl" % key)

    
    def performApproximateDP(self, df_st: pd.DataFrame, df_dec: pd.DataFrame, states: Dict[str, State], p: Probabilities, iniState: str, n: int, samples: int = None, key: str = ""):
        """Perform approximate dynamic programming

        # Assign starting state
       
        # For n in 1...N:
        ## For t in 0...T:
        ### Get decisions for state
        ### Get transitions for t
        ### Choose decision maximizing current and expected contribution
        ### Calculate value for decision and expected contribution
        ### Store value for current state
        ### Compute next state by sampling from possible transitions

        Args:
            df_st (pd.DataFrame): State DF
            df_dec (pd.DataFrame): Decision space
            states (Dict[str, State]): Key-Value Pair of State key and state object
            p (Probabilities): Probability class
            iniState (str): Initial State
            n (int): Number iterations
            samples (int): Samples to draw exog information. If None all exog. inf. are created
        """

        iterationCounter: int = 0
        best_dec = pd.DataFrame(columns=["s_key", "d_key", "value"])

        logging.debug("Starting algorithm loop")

        # Loop until reached target iteration number
        while iterationCounter < n:

            cState = iniState

            iterationCounter += 1
            logging.debug("Current iteration %d of %d" % (iterationCounter, n))
            print("%d/%d" % (iterationCounter, n))

            # Loop over all time slices
            for t in np.arange(0, con.T):

                # Construct all decisions for current state
                df = pd.merge(df_st, g.constructDecisions(states[cState], df_dec), on=["s_key"])
                                
                # Calculate contribution of decisions
                cont = df[["s_key", "d_key", "s_obj", "d_obj"]].drop_duplicates(subset=["s_key", "d_key"]).copy()
                cont["cont"] = cont["s_obj"].apply(lambda s: s.get_P_S())*con.eta*cont["d_obj"].apply(lambda d: d.get_x_V2G()) \
                - cont["s_obj"].apply(lambda s: s.get_P_B())*con.eta * cont["d_obj"].apply(lambda d: d.get_x_G2V()) \
                - con.epsilon * \
                cont["s_obj"].apply(lambda s: s.get_D()) * \
                cont["d_obj"].apply(lambda d: 1-d.get_x_t())

                cont["cont"] = cont["cont"].astype(float)


                # Create sample of exog information at t for current state
                df_exo = g.constructExogInfoT(df, p, t, samples) 
                #df_exo.to_excel("/usr/app/output/xlsx/04_ADP/adp_df1_%s.xlsx" % str(iterationCounter))

                # Perform transitions
                df_trans = g.constructTransitions(df_exo, states)
                #df_trans.to_excel("/usr/app/output/xlsx/04_ADP/adp_df2_%s.xlsx" % str(iterationCounter))
                
                # Caclulate expected contribution
                df_trans["ex_cont"] = df_trans["p"] * df_trans["s_d_key"].apply(lambda s: states.get(s).get_V_N())
                
                # Calculate expected total contribution
                grp_sum = df_trans[["s_key", "d_key", "ex_cont"]].groupby(["s_key", "d_key"])["ex_cont"].sum().reset_index()
                grp_sum = pd.merge(grp_sum, cont[["s_key", "d_key", "cont"]], on=["s_key", "d_key"])
                grp_sum["tot_cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_cont"]
                #grp_sum.to_excel("/usr/app/output/xlsx/04_ADP/adp_grp_%s.xlsx" % str(iterationCounter))
                
                # Select decision maximizing expected total contribution
                max_con = grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].max()

                # Store/update best decision
                tmp = grp_sum.loc[grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].idxmax(), ["s_key", "d_key"]]
                best_dec = pd.concat([best_dec[~best_dec.s_key.isin(tmp.s_key)], tmp])

                # Update state value
                states[cState].set_V_N(max_con.loc[cState])

                # Perform transition to next state
                logging.debug("Best decision for state %s is  %s" % (cState, best_dec.loc[best_dec.s_key == cState, "d_key"].iloc[0]))

                df_best_dec = df_trans.loc[(df_trans.s_key == cState) & (df_trans.d_key == best_dec.loc[best_dec.s_key == cState, "d_key"].iloc[0])].reset_index(drop=True)
                #df_best_dec.to_excel("/usr/app/output/xlsx/04_ADP/adp_df3_%s.xlsx" % str(iterationCounter))

                # Sample transition randomly according to their probabilities
                ## Need to be scaled for np random choice to be qual to 1 in sum (some transitions filtered out because invalid)
                probs = np.divide(df_best_dec["p"], df_best_dec["p"].sum())

                trans = df_best_dec.loc[np.random.choice(np.arange(0, len(df_best_dec.index)), size = 1, p = probs),:]
                cState = trans["s_d_key"].iloc[0]

                logging.debug("Choosing randomly the following transition %s with probability %f" % (trans[["trpstrt", "trpln", "prc_b", "prc_s"]].to_string(), trans["p"]))
                logging.debug("Next state is %s" % cState)

                # End inner t loop
            
            # End outer iteration loop

        logging.debug("Finished. Starting to spool out best decisions after running MO for remaining states")

        # Add all non-visited states to df with myopic decision
        df_st.reset_index(inplace=True)
        
        

        dec_space = Path("/usr/app/output/df/[%d-%d]-decisionspace.pkl" %  (con.T, con.trip_max))
        if dec_space.is_file():
            logging.debug("Reading decision space from disk.")
            df_missing_dec = pd.read_pickle(dec_space.absolute())
            # Need to prune loaded decisions. No decision constructed for t = 0 as we have initial state - not for terminal nodes - only for states without decision
            df_missing_dec = df_missing_dec.loc[(~df_missing_dec.s_key.isin(best_dec.s_key)) & (df_missing_dec.s_key.isin(df_st.s_key))]
            df_missing_dec = df_missing_dec.loc[(df_missing_dec.s_key.apply(lambda s: (states[s].get_t() != 0) & (not states[s].get_isTerminal())))]
        else:
            logging.debug("Constructing decision space freshly.")
            ls_mising_dec = [g.constructDecisions(states[s], df_dec) for s in df_st.loc[(~df_st.s_key.isin(best_dec.s_key)) & (df_st.s_obj.apply(lambda s: not s.get_isTerminal())), "s_key"]]
            df_missing_dec = pd.concat(ls_mising_dec)
            df_missing_dec.to_pickle("/usr/app/output/df/[%d-%d]-decisionspace.pkl" %  (con.T, con.trip_max))

        logging.info("Start to run myopic optimization for %d state with valid decisions of %d missing of %d total states" % (len(df_missing_dec.s_key.unique()), len(df_st.loc[~df_st.s_key.isin(best_dec.s_key)]), len(df_st.index)))

        # Begin myopic decision making 
        cont_missing_dec = df_missing_dec.copy().reset_index()
        cont_missing_dec["s_obj"] =  cont_missing_dec["s_key"].apply(lambda s: states[s])
        cont_missing_dec["cont"] = cont_missing_dec["s_obj"].apply(lambda s: s.get_P_S())*con.eta*cont_missing_dec["d_obj"].apply(lambda d: d.get_x_V2G()) \
            - cont_missing_dec["s_obj"].apply(lambda s: s.get_P_B())*con.eta * cont_missing_dec["d_obj"].apply(lambda d: d.get_x_G2V()) \
            - con.epsilon * \
            cont_missing_dec["s_obj"].apply(lambda s: s.get_D()) * \
            cont_missing_dec["d_obj"].apply(lambda d: 1-d.get_x_t())

        cont_missing_dec["cont"] = cont_missing_dec["cont"].astype(float)

        # Select max contrivution per state (decision with best contribution)
        max_con_missing_dec = cont_missing_dec.groupby(["s_key"])["cont"].max()

        # Store best decision
        df_missing_dec = cont_missing_dec.loc[cont_missing_dec.groupby(["s_key"])["cont"].idxmax(), ["s_key", "d_key"]]

        # Update state values as sum of total_contributions
        for s, v in max_con_missing_dec.items():
            states[s].set_V_N(v)

        # End myopic decision making

        # Add all remaining (terminal) states
        best_dec = pd.concat([best_dec, df_missing_dec])
        best_dec = pd.concat([best_dec, pd.DataFrame({"s_key": df_st.loc[df_st.s_obj.apply(lambda s: s.get_isTerminal()), "s_key"]})]).reset_index(drop=True)

        # Values
        best_dec["value"] = best_dec["s_key"].apply(lambda s: states.get(s).get_V_N())

        best_dec.to_excel("/usr/app/output/xlsx/04_ADP/%s-best_decisions.xlsx" % key)
        best_dec.to_pickle("/usr/app/output/df/%s-best_decisions.pkl" % key)




        
        


