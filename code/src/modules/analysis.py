
from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition, performTransition
from src.modules.probabilities import Probabilities

from src.modules import constants as con


import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import logging

# Stores measures and provide evaluation functionality


def analyseValue(algo_dec: Dict[str, pd.DataFrame], iniState: State, prob: Probabilities, states: Dict[str, State]):
    results = pd.DataFrame(columns=["Algorithm", "Scenario", "Value"])
    scenario_runs = []

    np.random.seed(1997)

    # Derive scenarios (Dict[int, pd.DataFrame (Exog info)])
    scen_exo = createScenarios(prob)


    for algo, dec in algo_dec.items():
        scenario_runs = []

        for scen, exo in scen_exo.items():
            logging.info("%s - %d" % (algo, scen))
            res = runScen(algo, scen, dec, exo.copy(), iniState, states)
            results.loc[len(results.index)] = [algo, scen, res[0]]
            scenario_runs += [res[1]]
        
        con = pd.concat(scenario_runs)
        con.to_excel("/usr/app/output/xlsx/%s_scenarios.xlsx" % algo)
        con.to_pickle("/usr/app/output/df/%s_scenarios.pkl" % algo)


    # Generate pkl and excel and store away
    results.to_excel("/usr/app/output/xlsx/value_comp.xlsx")
    results.to_pickle("/usr/app/output/df/value_comp.pkl")

    return results

def runScen(algo: str, scen: int, decisions: pd.DataFrame, exog: pd.DataFrame, iniState: str, states: Dict[str, State]) -> Tuple[float, pd.DataFrame]:
    details = pd.DataFrame(columns=["Algorithm", "Scenario", "t", "smpl", "State",  "B_L",
                            "V_TA ", "D", "P_B",  "P_S", "Decision", "xG2V", "xV2G", "xTrip", "Contribution"])
    cState = iniState
    cStateObj = states[iniState]

    iterations = exog["smpl"].max()

    for i in np.arange(1, iterations+1):

        cState = iniState
        cStateObj = states[iniState]

        for t in np.arange(0,con.T):
            logging.info("%d - %d" % (i, t))
            # Slice exog on smpls
            exo = exog.loc[exog.smpl == i, :].copy()

            # Get decision
            dec_ls = decisions.loc[decisions.s_key == cState, "d_key"].values[0].split(",")
            dec = Decision(float(dec_ls[0]), float(dec_ls[1]), int(dec_ls[2]))

            # Derive and store contributon
            cont = cStateObj.get_P_S()*con.eta*dec.get_x_V2G() - cStateObj.get_P_B()*con.eta * dec.get_x_G2V() \
            - con.epsilon * cStateObj.get_D() * (1-dec.get_x_t())

            # Store information
            details.loc[len(details.index)] = [algo, scen, t, i, cState, cStateObj.get_B_L(), cStateObj.get_V_TA(), cStateObj.get_D(), cStateObj.get_P_B(), cStateObj.get_P_S(), \
                dec.getKey(), dec.get_x_G2V(), dec.get_x_V2G(), dec.get_x_t(), cont]

            # Perform transition to new state
            cState = performTransition(cStateObj, dec, exo.loc[(exo.t == t), "trpln"].iloc[0], exo.loc[(exo.t == t), "prc_b"].iloc[0], exo.loc[(exo.t == t), "prc_s"].iloc[0]).getKey()
            if t < con.T - 1:
                cStateObj = states[cState]

            

    return (details.loc[:, "Contribution"].sum()/iterations, details)

def createScenarios(prob: Probabilities) -> Dict[int, pd.DataFrame]:
    s1 = pd.DataFrame(columns = ["t", "smpl", "trpln", "prc_b", "prc_s"])
    s2 = pd.DataFrame(columns = ["t", "smpl", "trpln", "prc_b", "prc_s"])
    s3 = pd.DataFrame(columns = ["t", "smpl", "trpln", "prc_b", "prc_s"])

    print(con.trip_max)
    for t in np.arange(con.T):
        p = prob.getProbabilities(t*con.tau*60)
        p = p.loc[(p.p > 0.0) & (p.trpln <= con.trip_max)].reset_index()
        p.p = p.p.astype(float)

        # Most likely
        s1.loc[len(s1.index)] = p.iloc[p.p.idxmax()][['t', 'trpln', 'prc_b', 'prc_s']].copy()

        # Most unlikely (of the ones that could actually be observed)
        s2.loc[len(s2.index)] = p.iloc[p.p.idxmin()][['t', 'trpln', 'prc_b', 'prc_s']].copy()

        # 1000 random smpls
        tmp = p.iloc[np.random.choice(np.arange(0, len(p.index)), size=1000, replace=True, p=np.divide(p["p"], p["p"].sum()))][['t', 'trpln', 'prc_b', 'prc_s']].copy()
        tmp["smpl"] = np.arange(tmp.shape[0])+1
        s3 = pd.concat([s3, tmp])

    s1["smpl"] = 1
    s2["smpl"] = 1

    # Transform t to match index 
    s1.t = s1.t/int(60*con.tau)
    s2.t = s2.t/int(60*con.tau)
    s3.t = s3.t/int(60*con.tau)

    return {0: s1, 1: s2, 2: s3}



class Analysis:

    # Variables to store measures
    runtime = pd.DataFrame(columns=["time"])

    # Only for value iteration
    splitRuntime = pd.DataFrame(
        columns=["t_state", "d_state", "tr_state", "vi"])
    stateSpace = pd.DataFrame(columns=["amount"])
    decisionSpace = pd.DataFrame(columns=["amount"])
    totalSpace = pd.DataFrame(columns=["amount"])

    allVar = []

    def __init__(self):
        self.allVar = [("rt", self.runtime), ("splitrt", self.splitRuntime), ("sspace",
                        self.stateSpace), ("dspace", self.decisionSpace), ("tspace", self.totalSpace)]

    def addMeasure(self, key, value, m):
        if m == "rt":
            self.runtime.loc[key]=[value]
            return
        if m == "splitrt":
            self.splitRuntime.loc[key]=value
            return
        if m == "sspace":
            self.stateSpace.loc[key]=[value]
            return
        if m == "dspace":
            self.decisionSpace.loc[key]=[value]
            return
        if m == "tspace":
            self.totalSpace.loc[key]=[value]
            return


    def putout(self, key):
        # Spool out dataframe pkl
        for var in self.allVar:
            if len(var[1].index) != 0:
                var[1].to_pickle("/usr/app/data/tmp/%s_%s.pkl" % (key, var[0]))

        # Spool out excel
        with pd.ExcelWriter('/usr/app/output/xlsx/%s-analysis.xlsx' % key) as writer:
            for var in self.allVar:
                if len(var[1].index) != 0:
                    var[1].to_excel(writer, sheet_name = var[0])
