from src.modules.state import State
from src.modules.decision import Decision
from src.modules import constants as con
from src.modules.constraintChecker import checkDecision, checkState

import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import List
import math


num_cores = multiprocessing.cpu_count()


def constructStates() -> pd.DataFrame:
    ## {\beta_min..\beta_max}*{0..max_trip/speed}*{0..max_trip}*{0..max_preis_b}*{0..max_preis_s}

    df0 = pd.DataFrame({"key" : 0, "t": list(np.arange(0.0,con.T+1, 1))})
    df1 = pd.DataFrame({"key" : 0, "B_L": list(np.arange(con.beta_min,con.beta_max+con.step_en, con.step_en))})
    df2 = pd.DataFrame({"key" : 0, "V_TA": list(np.arange(0.0,math.ceil(con.trip_max/con.gamma/con.tau), 1))})
    df3 = pd.DataFrame({"key" : 0, "D": list(np.arange(0.0,con.trip_max+1, 1))})
    df4 = pd.DataFrame({"key" : 0, "P_B": list(np.arange(0.0,con.price_b_max+con.step_pr, con.step_pr))})
    df5 = pd.DataFrame({"key" : 0, "P_S": list(np.arange(0.0,con.price_s_max+con.step_pr, con.step_pr))})

    df = (df0
            .pipe(pd.merge, right=df1, on=["key"])
            .pipe(pd.merge, right=df2, on=["key"])
            .pipe(pd.merge, right=df3, on=["key"])
            .pipe(pd.merge, right=df4, on=["key"])
            .pipe(pd.merge, right=df5, on=["key"])
    )


    df.drop_duplicates(ignore_index=True, inplace=True)

    df["V_N"] = 0

    # Filter out invalid states
    df.drop(df.index[[not checkState(s.t, s.B_L, s.V_TA, s.D) for s in df.itertuples()]], inplace=True)
    print(len(df))

    # Create state objects
    df["obj"] = [State(s.t, s.B_L, s.V_TA, s.D, s.P_B, s.P_S) for s in tqdm(df.itertuples())]


    df["terminal"] = df["obj"].apply(lambda x: x.get_isTerminal())

    df["key"] = df["obj"].apply(lambda x: x.getKey())
 
    return df.copy()


def constructDecisions(s:State) -> List[Decision]:
    ## {0..my}*{0..my}*{0,1} -> {0}*{0..my}*{0,1} and {0..my}*{0}*{0,1}

    df1 = pd.DataFrame({"x_G2V": list(np.arange(0.0,con.my+con.step_en, con.step_en))})
    df2 = pd.DataFrame({"x_V2G": list(np.arange(0.0,con.my+con.step_en, con.step_en))})
    df3 = pd.DataFrame({"x_trip": [0,1]}) if s.getY() == 0 else pd.DataFrame({"x_trip": [0]})

    df1["x_V2G"] = 0
    df2["x_G2V"] = 0

    df1["key"] = 0
    df2["key"] = 0
    df3["key"] = 0
    
    df = pd.concat([pd.merge(df1, df3, on=["key"]).copy(), pd.merge(df2, df3, on=["key"]).copy()], ignore_index=True)
    df.drop_duplicates(ignore_index=True, inplace=True)

    # Create decision objects and filter out invalid decisions
    ls = list(filter(lambda d: checkDecision(s,d), [Decision(a.x_G2V, a.x_V2G, a.x_trip) for a in df.itertuples()]))

    return ls

def generateTransitions(s: (State, List[Decision])) -> pd.DataFrame:
    
    # TODO [state, decision, transition] => transition has prob and destination state assigned
    return pd.DataFrame()
