from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition
from src.modules.transition import performTransition
from src.modules.probabilities import Probabilities
from src.modules import constants as con
from src.modules.constraintChecker import checkDecision, checkState

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import math
from joblib import Parallel, delayed
import multiprocessing as mp

import logging



def constructStates(params: Tuple = None) -> pd.DataFrame:
    ## {\beta_min..\beta_max}*{0..max_trip/speed}*{0..max_trip}*{prc_b}*{prc_b|prc_s}

    df0 = pd.DataFrame({"key" : 0, "t": list(np.arange(0 if params is None else 1,con.T+1, 1))})
    df1 = pd.DataFrame({"key" : 0, "B_L": [round(x,1) for x in np.arange(con.beta_min,con.beta_max+con.step_en, con.step_en)]})
    df2 = pd.DataFrame({"key" : 0, "V_TA": [0.0]+list(np.arange(0.5,con.trip_max+1, 1))})
    df3 = pd.DataFrame({"key" : 0, "D": [0.0]+list(np.arange(0.5,con.trip_max+1, 1))}) # Might be adjusted to match prob distribution

    # Prices can be considered as specific ones for each time to keep state space smaller
    df_prc_b = pd.read_pickle("/usr/app/data/probabilities/d_prc_b.pkl")
    df_prc_s = pd.read_pickle("/usr/app/data/probabilities/d_prc_s.pkl")
    df4 = pd.merge(df_prc_b,df_prc_s, on=["t"], suffixes=("_b", "_s")).loc[:,["t","prc_b", "prc_s"]]
    df4.rename(columns={"prc_b":"P_B", "prc_s":"P_S"}, inplace=True)

    df4["t"] = df4["t"]/int(60*con.tau) # given in 15*t for t = 0, ..., T - therefore calculation required


    df = (df0
            .pipe(pd.merge, right=df1, on=["key"])
            .pipe(pd.merge, right=df2, on=["key"])
            .pipe(pd.merge, right=df3, on=["key"])
            .pipe(pd.merge, right=df4, on=["t"]) # Prices only merged inner on t and not crossjoin!
    )

     ## Add initial state
    if params is not None:
        ls = [0] + list(map(float, params[0].split(",")))
        df.loc[len(df.index)] = ls

    df["t"] = df["t"].astype(int)
    df.drop(columns = ["key"], inplace = True)

    df.drop_duplicates(ignore_index=True, inplace=True)

    # Filter out invalid states for t== 0 and t==T
    logging.debug("Created Cross Product. Filtering state space")
    df_check = df.loc[(df["t"] > 0) & (df["t"] < con.T),:].reset_index(drop=True)
    df_filter = df.loc[(df["t"]  == 0) | (df["t"]  == con.T),:].reset_index(drop=True)
    df_filter.drop(df_filter.index[[not checkState(s.t, s.B_L, s.V_TA, s.D) for s in df_filter.itertuples()]], inplace=True)

    df = pd.concat([df_check, df_filter])

    # Create state objects
    # ERROR parallel execution here does not set isTerminal flag somehow :o
    #df["s_obj"] = Parallel(n_jobs=mp.cpu_count())(delayed(lambda s: State(s.t, s.B_L, s.V_TA, s.D, s.P_B, s.P_S))(s) for s in tqdm(df.itertuples()))
    #logging.debug("Create state objects in chunks for %d states" % len(df))
    dfs = []
    for k,g in df.groupby(np.arange(len(df))//10000000):
        #logging.debug("Chunk progress %d/%d" % (k*10000000, len(df)))
        g["s_obj"] = [State(s.t, s.B_L, s.V_TA, s.D, s.P_B, s.P_S) for s in g.itertuples()]
        dfs += [g]
    df = pd.concat(dfs)
    #logging.debug("Finished state objects in chunks resulting in %d states" % len(df))

    #df["s_obj"] = [State(s.t, s.B_L, s.V_TA, s.D, s.P_B, s.P_S) for s in df.itertuples()]
                    
    df["s_key"] = df["s_obj"].apply(lambda x: x.getKey())
 
    return df[["s_key", "s_obj"]].copy()


def decisionSpace() -> pd.DataFrame:
    ## {0..my}*{0..my}*{0,1} -> {0}*{0..my}*{0,1} and {0..my}*{0}*{0,1}

    df1 = pd.DataFrame({"x_G2V": [round(x,1) for x in np.arange(0.0,round(con.my,1)+0.1, con.step_en)]})
    df2 = pd.DataFrame({"x_V2G": [round(x,1) for x in np.arange(0.0,round(con.my,1)+0.1, con.step_en)]})
    df3 = pd.DataFrame({"x_trip": [0,1]})

    df1["x_V2G"] = 0
    df2["x_G2V"] = 0

    df1["key"] = 0
    df2["key"] = 0
    df3["key"] = 0
    
    df = pd.concat([pd.merge(df1, df3, on=["key"]).copy(), pd.merge(df2, df3, on=["key"]).copy()], ignore_index=True)
    df.drop_duplicates(ignore_index=True, inplace=True)

    # Create decision objects
    #df["d_obj"] = Parallel(n_jobs=mp.cpu_count())(delayed(lambda a: Decision(a.x_G2V, a.x_V2G, a.x_trip))(a) for a in df.itertuples())
    #[Decision(a.x_G2V, a.x_V2G, a.x_trip) for a in tqdm(df.itertuples())] 
    #logging.debug("Create dec objects in chunks for %d decisions" % len(df))
    dfs = []
    for k,g in df.groupby(np.arange(len(df))//10000000):
        #logging.debug("Chunk progress %d/%d" % (k*1000000, len(df)))
        g["d_obj"] = [Decision(a.x_G2V, a.x_V2G, a.x_trip) for a in g.itertuples()]
        dfs += [g]
    df = pd.concat(dfs)
    #logging.debug("Finished decision objects in chunks resulting in %d decisions" % len(df))
    
    df["d_key"] = df["d_obj"].apply(lambda x: x.getKey())

    return df

def constructDecisions(s:State, df_c: pd.DataFrame) -> pd.DataFrame:

    # Filter on y_t
    df = df_c[df_c["x_trip"].apply(lambda x_t: False if (s.getY() == 1) & (x_t == 1) else True)].copy().reset_index()
    
    # Filter out invalid decisions
    #df.drop(df.index[[not checkDecision(s, d.d_obj) for d in df.itertuples()]], inplace=True)
    #logging.debug("Filtering dec objects in chunks for %d decisions" % len(df))
    dfs = []
    for k,g in df.groupby(np.arange(len(df))//10000000):
        #logging.debug("Chunk progress %d/%d" % (k*1000000, len(df)))
        g.drop(g.index[[not checkDecision(s, d.d_obj) for d in g.itertuples()]], inplace=True)
        dfs += [g]
    df = pd.concat(dfs)
    #logging.debug("Finished filtering decision objects in chunks resulting in %d decisions" % len(df))

    df["s_key"] = s.getKey()

    return df[["d_key", "s_key", "d_obj"]].copy()


def constructExogInfo(df: pd.DataFrame, p: Probabilities) -> pd.DataFrame:
    # Construct ex_info data frame for all t
    df_p = pd.DataFrame()

    processed_list = Parallel(n_jobs=mp.cpu_count())(delayed(p.getProbabilities)(t*con.tau*60) for t in np.arange(0, con.T+1, 1))
    df_p = pd.concat(processed_list)

    # Need to match time index
    df_p["t"] = df_p["t"]/int(con.tau*60)
    df_p.reset_index(inplace=True,drop=True)
            
    return pd.merge(df,df_p, on=["t"])


def constructExogInfoT(df: pd.DataFrame, p: Probabilities, t: int, samples: int = None) -> pd.DataFrame:
    # Construct ex_info data frame for given t
    if samples is not None:
        df_p = p.getProbabilitiesSampled(t*con.tau*60, samples)
    else:
        df_p = p.getProbabilities(t*con.tau*60)

    # Need to match time index
    df_p["t"] = df_p["t"]/int(con.tau*60)
    df_p.reset_index(inplace=True,drop=True)
            
    return pd.merge(df,df_p, on=["t"])


def constructTransition(df: pd.DataFrame) -> str:
    return performTransition(df.loc[0, "s_obj"], df.loc[0, "d_obj"], df.loc[0, "trpln"], df.loc[0, "s_prc_b"], df.loc[0, "prc_s"]).getKey()


def constructTransitions(df:pd.DataFrame, states: List) -> pd.DataFrame:
    # Construct transition objects and get key of destination state
    #df["s_d_key"] = Parallel(n_jobs=mp.cpu_count())(delayed(lambda t: performTransition(t.s_obj, t.d_obj, t.trpln, t.prc_b, t.prc_s).getKey())(t) for t in df.itertuples())
    dfs = []
    for k,g in df.groupby(np.arange(len(df))//100000000):
        #logging.debug("Chunk progress %d/%d" % (k*1000000, len(df)))
        g["s_d_key"] = Parallel(n_jobs=mp.cpu_count())(delayed(lambda t: performTransition(t.s_obj, t.d_obj, t.trpln, t.prc_b, t.prc_s).getKey())(t) for t in g.itertuples())
        dfs += [g]
    df = pd.concat(dfs)

    logging.debug("DataFrame has %d rows before transition pruning." % len(df))

    #df_chunks = np.array_split(raw_df ,8)
#
    #with multiprocessing.Pool(8) as pool:
    #    processed_df = pd.concat(pool.map(process_df_function, df_chunks), ignore_index=True)
#
    ## Filter out rows leading to invalid states
    #df = df.drop(df.index[[not checkState(int(s[0]), float(s[1]), float(s[2]), float(s[3])) for s in df["s_d_key"].apply(lambda s: s.split(","))]])
#
    #logging.debug("DataFrame has %d rows after filtering invalid states." % len(df))

    # Filter out rows leading to unknown (=> invalid) destination states
    df = df[df["s_d_key"].isin(states)]
    
    logging.debug("DataFrame has %d rows after transition pruning." % len(df))

    return df.copy()