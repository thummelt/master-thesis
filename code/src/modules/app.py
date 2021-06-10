from src.modules.state import State
from src.modules.decision import Decision
from src.modules import generator as g
from src.modules.solutionAlgorithms import SolutionAlgorithms 
from src.modules.probabilities import Probabilities
from src.modules.analysis import Analysis
from src.modules import constants as con

import pandas as pd
from typing import List, Dict
from itertools import product, chain
import numpy as np
import logging
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import time


from fnmatch import fnmatch



## Sets algorithm up. Called by Jupyter NB

class App:

    # Variables
    sol: SolutionAlgorithms
    p: Probabilities

    starttime : float
    key : str
    splittime = []

    ## States as dataframe to speed up lookup
    df_states = pd.DataFrame(columns=["s_key", "s_obj"])

    ## Decisions as dataframe to speed up lookup
    df_decisions = pd.DataFrame(columns=["d_key", "s_key", "d_obj"])


    def __init__(self, T, trip_max, an: Analysis):

        logging.basicConfig(filename="logs" + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '_app.log',
                        filemode='w+', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

            
        # Solver
        self.sol = SolutionAlgorithms()

        # Probabilities
        self.p = Probabilities()

        self.key = "[%d-%d]" % (T if T != None else con.T, trip_max if trip_max != None else con.trip_max)
        
        # Override time and trpln parameters if given
        if T != None:
            con.T = T
        
        if trip_max != None:
            con.trip_max = trip_max

        self.splittime = []

        # Starttime
        self.starttime = time.time()

        # States
        self.df_states = g.constructStates()
        self.df_states.set_index("s_key", inplace=True, drop=True)
        self.splittime += [time.time()-self.starttime]
        an.addMeasure(self.key, len(self.df_states.index), "sspace")

        logging.info("Finished creation of %d states" % len(self.df_states))

        # For each state (which are not terminal) construct all decisions  
        df_dec = g.decisionSpace()
        processed_list = Parallel(n_jobs=mp.cpu_count())(delayed(g.constructDecisions)(i, df_dec) for i in tqdm(self.df_states.loc[self.df_states["s_obj"].apply(lambda s: not s.get_isTerminal()),"s_obj"]))
        self.df_decisions = pd.concat(processed_list)
        self.splittime += [time.time()-self.starttime]
        an.addMeasure(self.key, len(self.df_decisions.index), "dspace")
        
        logging.info("Finished creation of %d decisions" % len(self.df_decisions))

        self.valueIteration(an)
        
        self.splittime += [time.time()-self.starttime]
        an.addMeasure(self.key, self.splittime, "splitrt")
        an.addMeasure(self.key, time.time()-self.starttime, "rt")
        
    
    def valueIteration(self, an: Analysis) -> bool:
        
        # Construct data frame for performance ["s_key", "s_obj", "d_key", "d_obj", "t",  "trpln", "prc", "p", "tr_obj"]
        df = pd.merge(self.df_states, self.df_decisions, on=["s_key"])
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        logging.info("Finished joining states and decisions. Shape of df is %s"  % str(df.shape))
        
        # Construct exog information
        df = g.constructExogInfo(df, self.p)
        
        logging.info("Finished creation of exogenous information. Shape of df is %s"  % str(df.shape))

        # Construct transitions
        df = g.constructTransitions(df, self.df_states.index.values.tolist())
        df.reset_index(inplace=True, drop=True)
        self.splittime += [time.time()-self.starttime]
        an.addMeasure(self.key, len(df.index), "tspace")

        logging.info("Finished creation of transitions. Shape of df is %s" % str(df.shape))

        #df.to_pickle("/usr/app/data/tmp/viInputDf.pkl") 
        #self.df_states.to_pickle("/usr/app/data/tmp/viDFStates.pkl") 

        
        # Call VI with state-decision-transition tuples
        return self.sol.performStandardVI(df, self.df_states["s_obj"].to_dict())


    def approxValueIteration(self) -> bool:
        # Construct sample path

        # Call solution algorithm
        pass


    def trivialPolicy(self) -> bool:
        pass

