
from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition, performTransition
from src.modules.probabilities import Probabilities
from src.modules import generator as g
from src.modules import constants as con


import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import logging

# Stores measures and provide evaluation functionality
 
logging.basicConfig(filename="logs" + '/' + datetime.now().strftime('%Y%m%d_%H%M') + '_app.log',
                        filemode='w+', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')



def analyseValue(algo_lookup: Dict[str, pd.DataFrame], iniState: State, prob: Probabilities, states: Dict[str, State], param):
    results = pd.DataFrame(columns=["Algorithm", "Scenario", "Value"])
    scenario_runs = []

    con.T = param["T"]
    con.trip_max  = param["trip_max"]

    np.random.seed(1997)

    # Derive scenarios (Dict[int, pd.DataFrame (Exog info)])
    scen_exo = createScenarios(prob, param["T"], param["trip_max"])

    for algo, lookup in algo_lookup.items():
        scenario_runs = []

        for scen, exo in scen_exo.items():
            logging.info("%s - %d" % (algo, scen))
            res = runScen(algo, scen, lookup, exo.copy(), iniState, states, param)
            results.loc[len(results.index)] = [algo, scen, res[0]]
            scenario_runs += [res[1]]

        
        cont = pd.concat(scenario_runs)
        cont.to_excel("/usr/app/output/xlsx/[%s-%s]-%s_scenarios.xlsx" % (param["T"], param["trip_max"], algo))
        cont.to_pickle("/usr/app/output/xlsx/[%s-%s]-%s_scenarios.pkl" % (param["T"], param["trip_max"], algo))

    # Generate pkl and excel and store away
    results.to_excel("/usr/app/output/xlsx/[%s-%s]-value_comp.xlsx" % (param["T"], param["trip_max"]))
    results.to_pickle("/usr/app/output/xlsx/[%s-%s]-value_comp.pkl" % (param["T"], param["trip_max"]))

    return results

def runScen(algo: str, scen: int, lookup: pd.DataFrame, exog: pd.DataFrame, iniState: str, states: Dict[str, State], param) -> Tuple[float, pd.DataFrame]:
    details = pd.DataFrame(columns=["Algorithm", "Scenario", "t", "smpl", "State",  "B_L",
                            "V_TA ", "D", "P_B",  "P_S", "Decision", "xG2V", "xV2G", "xTrip", "Contribution"])
    cState = iniState
    cStateObj = states[iniState]
    
    # Construct decision space
    dec_space = g.decisionSpace()

    iterations = exog["smpl"].max()

    # Start simulation
    for i in np.arange(1, iterations+1):

        cState = iniState
        cStateObj = states[iniState]

        # Start looping over all time slices
        for t in np.arange(0,param["T"]):
            logging.info("%d - %d" % (i, t))
            # Slice exog on smpls
            exo = exog.loc[exog.smpl == i, :].copy()

            # Derive all valid decisions for current state
            decs = g.constructDecisions(states[cState],  dec_space.copy())
            decs["s_obj"] = states[cState]
            decs["t"] = decs["s_obj"].apply(lambda s: s.get_t())

            # Caclulate contribution of state-decision pair
            decs["cont"] = decs["s_obj"].apply(lambda s: s.get_P_S())*con.eta*decs["d_obj"].apply(lambda d: d.get_x_V2G()) \
                - decs["s_obj"].apply(lambda s: s.get_P_B())*con.eta * decs["d_obj"].apply(lambda d: d.get_x_G2V()) \
                - con.epsilon * \
                decs["s_obj"].apply(lambda s: s.get_D()) * \
                decs["d_obj"].apply(lambda d: 1-d.get_x_t())

            decs["cont"] = decs["cont"].astype(float)




            # Get decision according to policy
            if algo == "mo":
                # Get decision freshly for current state by maximizing current contribution over possible decisions                      
                best_dec = str(decs.loc[decs.groupby(["s_key"])["cont"].idxmax(), "d_key"].values[0])
                dec_ls = best_dec.split(",")

            else: # ADP
                # Slice exo info
                df_exo = exo.loc[(exo.t == t), :]
                #print(df_exo)

                if t == param["T"] - 1:
                    df_exo.loc[df_exo.t == t, "trpln"] = 0.0
                #print(df_exo)

                df_exo = pd.merge(decs,df_exo, on=["t"])
               
 
                # Perform transitions
                df_trans = g.constructTransitions(df_exo, states.keys(), param["T"])
                #print(df_trans.head(5))
                
                # Caclulate expected contribution by using lookup table
                df_trans["ex_cont"] = df_trans["p"] * df_trans["s_d_key"].apply(lambda s: lookup.loc[lookup.s_key == s,"value"].values[0])
                #print(df_trans.head(5))
                
                # Calculate expected total contribution
                grp_sum = df_trans[["s_key", "d_key", "ex_cont"]].groupby(["s_key", "d_key"])["ex_cont"].sum().reset_index()
                #print(grp_sum.head(5))
                grp_sum = pd.merge(grp_sum, decs[["s_key", "d_key", "cont"]], on=["s_key", "d_key"])
                #print(grp_sum.head(5))
                grp_sum["tot_cont"] = grp_sum["cont"] + con.expec*grp_sum["ex_cont"]
                #print(grp_sum.head(5))
                
                # Store/update best decision
                best_dec = grp_sum.loc[grp_sum[["s_key", "tot_cont"]].groupby(["s_key"])["tot_cont"].idxmax(), ["d_key"]].values[0][0]
                #print(best_dec)
                dec_ls = best_dec.split(",")
                #print(dec_ls)

            
            # Create decision obkect
            dec = Decision(float(dec_ls[0]), float(dec_ls[1]), int(dec_ls[2]))

            # Derive and store contributon
            cont = cStateObj.get_P_S()*con.eta*dec.get_x_V2G() - cStateObj.get_P_B()*con.eta * dec.get_x_G2V() \
            - con.epsilon * cStateObj.get_D() * (1-dec.get_x_t())

            # Store information
            details.loc[len(details.index)] = [algo, scen, t, i, cState, cStateObj.get_B_L(), cStateObj.get_V_TA(), cStateObj.get_D(), cStateObj.get_P_B(), cStateObj.get_P_S(), \
                dec.getKey(), dec.get_x_G2V(), dec.get_x_V2G(), dec.get_x_t(), cont]

            # Perform transition to new state
            cState = performTransition(cStateObj, dec, exo.loc[(exo.t == t), "trpln"].iloc[0], exo.loc[(exo.t == t), "prc_b"].iloc[0], exo.loc[(exo.t == t), "prc_s"].iloc[0]).getKey()
            
            if t < param["T"] - 1:
                cStateObj = states[cState]

            

    return (details.loc[:, "Contribution"].sum()/iterations, details)

def createScenarios(prob: Probabilities, t_horizon, trip_max) -> Dict[int, pd.DataFrame]:
    s1 = pd.DataFrame(columns = ["t", "smpl", "trpln", "prc_b", "prc_s", "p"])
    s2 = pd.DataFrame(columns = ["t", "smpl", "trpln", "prc_b", "prc_s", "p"])
    s3 = pd.DataFrame(columns = ["t", "smpl", "trpln", "prc_b", "prc_s", "p"])

    for t in np.arange(t_horizon):
        p = prob.getProbabilities(t*con.tau*60)
        p = p.loc[(p.p > 0.0) & (p.trpln <= trip_max)].reset_index()
        p.p = p.p.astype(float)

        # Most likely
        s1.loc[len(s1.index)] = p.iloc[p.p.idxmax()][['t', 'trpln', 'prc_b', 'prc_s', 'p']].copy()

        # Most unlikely (of the ones that could actually be observed)
        s2.loc[len(s2.index)] = p.iloc[p.p.idxmin()][['t', 'trpln', 'prc_b', 'prc_s', 'p']].copy()

        # 1000 random smpls
        tmp = p.iloc[np.random.choice(np.arange(0, len(p.index)), size=1000, replace=True, p=np.divide(p["p"], p["p"].sum()))][['t', 'trpln', 'prc_b', 'prc_s','p']].copy()
        tmp["smpl"] = np.arange(tmp.shape[0])+1
        s3 = pd.concat([s3, tmp])

    s1["smpl"] = 1
    s2["smpl"] = 1

    # Transform t to match index 
    s1.t = s1.t/int(60*con.tau)
    s2.t = s2.t/int(60*con.tau)
    s3.t = s3.t/int(60*con.tau)

    return {0: s2, 1: s1, 2: s3}



class Analysis:

    # Variables to store measures
    runtime = None

    # Only for value iteration
    splitRuntime = None
    stateSpace = None
    decisionSpace = None
    totalSpace = None


    allVar = []

    directory : str

    def __init__(self, algo):      
        # Variables to store measures
        self.runtime = pd.DataFrame(columns = ["T", "trip_max", "time"])

        # Only for value iteration
        self.splitRuntime = pd.DataFrame(
            columns=["T", "trip_max", "t_state", "d_state", "tr_state", "vi"])
        self.stateSpace = pd.DataFrame(columns=["T", "trip_max","amount"])
        self.decisionSpace = pd.DataFrame(columns=["T", "trip_max","amount"])
        self.totalSpace = pd.DataFrame(columns=["T", "trip_max","amount"]) 

        if algo in ["avi", "adp"]:
            self.runtime = pd.DataFrame(columns = ["T", "trip_max", "iterations", "samples", "time"])
            
        if algo == "vi":
            self.allVar = [("rt", self.runtime), ("splitrt", self.splitRuntime), ("sspace",
                        self.stateSpace), ("dspace", self.decisionSpace), ("tspace", self.totalSpace)]
        else:
            self.allVar = [("rt", self.runtime)]

    def addMeasure(self, key, value, m):
        if m == "rt":
            self.runtime.loc[key]=value
            return
        if m == "splitrt":
            self.splitRuntime.loc[key]=value
            return
        if m == "sspace":
            self.stateSpace.loc[key]=value
            return
        if m == "dspace":
            self.decisionSpace.loc[key]=value
            return
        if m == "tspace":
            self.totalSpace.loc[key]=value
            return

    def setDir(self, direc: str):
        self.directory = direc


    def putout(self, algo):
        # Spool out dataframe pkl
        for var in self.allVar:
            if len(var[1].index) != 0:
                var[1].to_pickle("%s/%s.pkl" % (self.directory, var[0]))

        # Spool out excel
        with pd.ExcelWriter('%s/%s-analysis.xlsx' % (self.directory, algo)) as writer:
            for var in self.allVar:
                if len(var[1].index) != 0:
                    var[1].to_excel(writer, sheet_name = var[0])
