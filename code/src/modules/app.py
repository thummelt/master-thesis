from src.modules.state import State
from src.modules.decision import Decision
from src.modules import generator as g
from src.modules.solutionAlgorithms import SolutionAlgorithms
from src.modules.probabilities import Probabilities
from src.modules.analysis import Analysis
from src.modules import constants as con

import pandas as pd
from typing import List, Dict, Tuple
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
    an: Analysis

    starttime : float
    key : str
    splittime = []
    algo : str

    ## States as dataframe to speed up lookup
    df_states = pd.DataFrame(columns=["s_key", "s_obj"])

    ## Decisions as dataframe to speed up lookup
    df_decisions = pd.DataFrame(columns=["d_key", "s_key", "d_obj"])


    def __init__(self):
        logging.basicConfig(filename="logs" + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '_app.log',
                        filemode='w+', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Solver
        self.sol = SolutionAlgorithms()

        # Probabilities
        self.p = Probabilities()

        # Analysis 
        self.an = Analysis()


    def putout(self):
        self.an.putout()
         
    def run(self, T = None, trip_max = None, algo: int = 0, params: Tuple = None):

        self.key = "[%d-%d]" % (T if T != None else con.T, trip_max if trip_max != None else con.trip_max)
        
        # Override time and trpln parameters if given
        if T is not None:
            con.T = T
        
        if trip_max is not None:
            con.trip_max = trip_max

        self.splittime = []

        # Starttime
        self.starttime = time.time()

        # States
        print("States")
        self.df_states = g.constructStates()
        self.df_states.set_index("s_key", inplace=True, drop=True)
        self.splittime += [time.time()-self.starttime]
        self.an.addMeasure(self.key, len(self.df_states.index), "sspace")

        self.df_states.to_pickle("/usr/app/output/df/allstates_%s.pkl" % self.key)

        logging.info("Finished creation of %d states" % len(self.df_states))

        # For each state (which are not terminal) construct all decisions  
        print("Decisions")
        df_dec = g.decisionSpace()

        # Not constructing decisions for adp
        if algo != 3:
            # ERROR  parallel execution is errenerous
            #processed_list = Parallel(n_jobs=mp.cpu_count())(delayed(g.constructDecisions)(i, df_dec) for i in tqdm(self.df_states.loc[self.df_states["s_obj"].apply(lambda s: not s.get_isTerminal()),"s_obj"]))
            processed_list = [g.constructDecisions(i, df_dec) for i in tqdm(self.df_states.loc[self.df_states["s_obj"].apply(lambda s: not s.get_isTerminal()),"s_obj"])]
            self.df_decisions = pd.concat(processed_list)
            self.splittime += [time.time()-self.starttime]
            self.an.addMeasure(self.key, len(self.df_decisions.index), "dspace")

        #self.df_decisions.to_excel("/usr/app/output/xlsx/alldecisions.xlsx")
        
        logging.info("Finished creation of %d decisions" % len(self.df_decisions))

        if algo == 0:
            self.valueIteration()
            self.algo = "Value Iteration"
        elif algo == 1:
            self.myopicOptimization()
            self.algo = "Myopic Optimization"
        elif algo == 2:
            self.approxValueIteration(params)
            self.algo = "Approximate Value Iteration"
        elif algo == 3:
            self.approxDynamicProgramming(params, df_dec)
            self.algo = "Approxmiate Dynamic Programming"
        else:
            logging.error("No solution algorithm could be associated")

        logging.info("Finished algorithm execution.")
        logging.info("Peace out.")
        
        self.splittime += [time.time()-self.starttime]
        if algo == 0:
            self.an.addMeasure(self.key, self.splittime, "splitrt")
        self.an.addMeasure(self.key, time.time()-self.starttime, "rt")
        
    
    def valueIteration(self) -> bool:
        
        df = pd.merge(self.df_states, self.df_decisions, on=["s_key"])
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        logging.info("Finished joining states and decisions. Shape of df is %s"  % str(df.shape))
        
        # Construct exog information
        print("Exo")
        df = g.constructExogInfo(df, self.p)
        
        logging.info("Finished creation of exogenous information. Shape of df is %s"  % str(df.shape))

        # Construct transitions
        print("Trans")
        df = g.constructTransitions(df, self.df_states.index.values.tolist())
        df.reset_index(inplace=True, drop=True)
        self.splittime += [time.time()-self.starttime]
        self.an.addMeasure(self.key, len(df.index), "tspace")

        logging.info("Finished creation of transitions. Shape of df is %s" % str(df.shape))

        #df.to_pickle("/usr/app/data/tmp/viInputDf.pkl") 

        
        # Call VI with state-decision-transition tuples
        print("Algo")
        return self.sol.performStandardVI(df.copy(), self.df_states["s_obj"].to_dict())


    def approxValueIteration(self, params) -> bool:

        df = pd.merge(self.df_states, self.df_decisions, on=["s_key"])
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        logging.info("Finished joining states and decisions. Shape of df is %s"  % str(df.shape))
        
        # Call approx VI with state-decision tuples
        print("Algo")
        return self.sol.performApproxVI(df.copy(), self.df_states["s_obj"].to_dict(), self.p, params[0], params[1], params[2])


    def myopicOptimization(self) -> bool:
        df = pd.merge(self.df_states, self.df_decisions, on=["s_key"])
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        logging.info("Finished joining states and decisions. Shape of df is %s"  % str(df.shape))
        
        # Call myopic optimization algo with state-decision tuples
        print("Algo")
        return self.sol.performMyopic(df.copy(), self.df_states["s_obj"].to_dict())


    def approxDynamicProgramming(self, params, df_dec) -> bool:
        df = self.df_states.copy()
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        # Call adp algo with state-decision tuples
        print("Algo")
        return self.sol.performApproximateDP(df.copy(), df_dec, self.df_states["s_obj"].to_dict(), self.p, params[0], params[1])

