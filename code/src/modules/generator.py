from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition
from src.modules import constants as con
from src.modules.constraintChecker import checkDecision, checkState

import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import List
import math

import logging


num_cores = multiprocessing.cpu_count()


def constructStates() -> pd.DataFrame:
    ## {\beta_min..\beta_max}*{0..max_trip/speed}*{0..max_trip}*{0..max_preis_b}*{0..max_preis_s}

    df0 = pd.DataFrame({"key" : 0, "t": list(np.arange(0,con.T+1, 1))})
    df1 = pd.DataFrame({"key" : 0, "B_L": [round(x,2) for x in np.arange(con.beta_min,con.beta_max+con.step_en, con.step_en)]})
    df2 = pd.DataFrame({"key" : 0, "V_TA": list(np.arange(0,math.ceil(con.trip_max/con.gamma/con.tau)+1, 1))})
    df3 = pd.DataFrame({"key" : 0, "D": list(np.arange(0,con.trip_max+1, 1))})
    df4 = pd.DataFrame({"key" : 0, "P_B": [round(x,2) for x in np.arange(0.0,con.price_b_max+con.step_pr, con.step_pr)]})
    df5 = pd.DataFrame({"key" : 0, "P_S": [round(x,2) for x in np.arange(0.0,con.price_s_max+con.step_pr, con.step_pr)]})

    df = (df0
            .pipe(pd.merge, right=df1, on=["key"])
            .pipe(pd.merge, right=df2, on=["key"])
            .pipe(pd.merge, right=df3, on=["key"])
            .pipe(pd.merge, right=df4, on=["key"])
            .pipe(pd.merge, right=df5, on=["key"])
    )


    df.drop_duplicates(ignore_index=True, inplace=True)


    # Filter out invalid states
    df.drop(df.index[[not checkState(s.t, s.B_L, s.V_TA, s.D) for s in df.itertuples()]], inplace=True)


    # Create state objects
    df["s_obj"] = [State(s.t, s.B_L, s.V_TA, s.D, s.P_B, s.P_S) for s in tqdm(df.itertuples())]

    df["s_key"] = df["s_obj"].apply(lambda x: x.getKey())
 
    return df[["s_key", "s_obj"]].copy()


def constructDecisions(s:State) -> pd.DataFrame:
    ## {0..my}*{0..my}*{0,1} -> {0}*{0..my}*{0,1} and {0..my}*{0}*{0,1}

    df1 = pd.DataFrame({"x_G2V": [round(x,2) for x in np.arange(0.0,con.my+0.1, con.step_en)]})
    df2 = pd.DataFrame({"x_V2G": [round(x,2) for x in np.arange(0.0,con.my+0.1, con.step_en)]})
    df3 = pd.DataFrame({"x_trip": [0,1]}) if s.getY() == 0 else pd.DataFrame({"x_trip": [0]})

    df1["x_V2G"] = 0
    df2["x_G2V"] = 0

    df1["key"] = 0
    df2["key"] = 0
    df3["key"] = 0
    
    df = pd.concat([pd.merge(df1, df3, on=["key"]).copy(), pd.merge(df2, df3, on=["key"]).copy()], ignore_index=True)
    df.drop_duplicates(ignore_index=True, inplace=True)

    # Create decision objects
    df["d_obj"] = [Decision(a.x_G2V, a.x_V2G, a.x_trip) for a in df.itertuples()]

    # Filter out invalid decisions
    df.drop(df.index[[not checkDecision(s, d.d_obj) for d in df.itertuples()]], inplace=True)

    df["d_key"] = df["d_obj"].apply(lambda x: x.getKey())

    df["s_key"] = s.getKey()

    return df[["d_key", "s_key", "d_obj"]].copy()


def constructTransitions(df:pd.DataFrame, states: List) -> pd.DataFrame:
    df["tr_obj"] = [Transition(x.s_obj, x.d_obj, x.p, x.trpln, x.prc) for x in df.itertuples()]
    

    df["s_d_key"] = df["tr_obj"].apply(lambda tr: tr.get_s_d().getKey())

    logging.debug("DataFrame has %d rows before transition pruning." % len(df))

    df.drop(df.index[[not checkState(s.t, s.B_L, s.V_TA, s.D) for s in df["tr_obj"].apply(lambda tr: tr.get_s_d())]], inplace=True)

    logging.debug("DataFrame has %d rows after filtering invalid states." % len(df))
    
    # Filter out rows where transitions are invalid
    # Long form
    #for i, row in tqdm(df.iterrows()):
    #    if not row["s_d_key"] in states:
    #        logging.warn("%s has been reached by %s but is not valid" %(row["tr_obj"].get_s_d().__str__(), row["tr_obj"].__str__() ))
    #        df.drop(index=i, inplace=True)
    
    # Filter out invalid => short form
    df = df[(df["s_d_key"].isin(states)) | (df["s_obj"].apply(lambda s: s.get_isTerminal()))]
    
    logging.debug("DataFrame has %d rows after transition pruning." % len(df))

    return df.copy()