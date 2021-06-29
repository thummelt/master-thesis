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
from pathlib import Path
import os


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

    directory : str

    ## States as dataframe to speed up lookup
    df_states = pd.DataFrame(columns=["s_key", "s_obj"])

    ## Decisions as dataframe to speed up lookup
    df_decisions = pd.DataFrame(columns=["d_key", "s_key", "d_obj"])


    def __init__(self, algo):
        logging.basicConfig(filename="logs" + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '_app.log',
                        filemode='w+', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Probabilities
        self.p = Probabilities()

        # Analysis
        self.an = Analysis(algo)


    def putout(self):
        self.an.putout(self.algo)
         
    def run(self, T = None, trip_max = None, algo: str = "adp", params: Tuple = None, run_nr = 0):

        # Create directory
        self.directory = "/usr/app/output/xlsx/%s/%s-%d/" % (algo, datetime.datetime.now().strftime('%Y%m%d'), run_nr)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Solver
        self.sol = SolutionAlgorithms(self.directory)
        
        # Analysis 
        self.an.setDir(self.directory)
        
        self.algo = algo
        # Override time and trpln parameters if given
        if T is not None:
            con.T = T
        
        if trip_max is not None:
            con.trip_max = trip_max

        self.key = "[%d-%d]" % (con.T, con.trip_max)

        self.splittime = []

        # Starttime
        self.starttime = time.time()

        # States
        print("States")
        self.df_states = g.constructStates(params)
        self.df_states.set_index("s_key", inplace=True, drop=True)
        if algo == "vi":
            self.splittime += [time.time()-self.starttime]
            self.an.addMeasure(self.key, [con.T, con.trip_max, len(self.df_states.index)], "sspace")

        self.df_states.to_pickle("/usr/app/output/df/%s-statespace.pkl" % self.key)

        logging.info("Finished creation of %d states" % len(self.df_states))

        # For each state (which are not terminal) construct all decisions  
        print("Decisions")
        df_dec = g.decisionSpace()

        # Not constructing decisions for adp
        if algo != "adp":
            # ERROR  parallel execution is errenerous
            #processed_list = Parallel(n_jobs=mp.cpu_count())(delayed(g.constructDecisions)(i, df_dec) for i in tqdm(self.df_states.loc[self.df_states["s_obj"].apply(lambda s: not s.get_isTerminal()),"s_obj"]))
            
            dec_space = Path("/usr/app/output/df/[%d-%d]-decisionspace.pkl" %  (con.T, con.trip_max))
            if (dec_space.is_file()) & (False):
                logging.debug("Reading decision space from disk.")
                self.df_decisions = pd.read_pickle(dec_space.absolute())
            else:
                logging.debug("Constructing decision space freshly.")
                ls_mising_dec = [g.constructDecisions(i, df_dec) for i in self.df_states.loc[self.df_states["s_obj"].apply(lambda s: not s.get_isTerminal()),"s_obj"]]
                self.df_decisions = pd.concat(ls_mising_dec)
                self.df_decisions.to_pickle("/usr/app/output/df/%s-decisionspace.pkl" % self.key)
            

            if algo == "vi":
                self.splittime += [time.time()-self.starttime]
                self.an.addMeasure(self.key, [con.T, con.trip_max, len(self.df_decisions.index)], "dspace")            

            logging.info("Finished creation of %d decisions" % len(self.df_decisions))        
        

         # Adapting key to also integrate algorithm
        self.key = "[%s%s]-" % (algo, "-"+ str(params[1:]).replace(",","") if params is not None else "") + self.key

        logging.info("Starting algorithm execution for key %s." % self.key)


        if algo == "vi":
            self.valueIteration()
            
        elif algo == "mo":
            self.myopicOptimization()
            
        elif algo == "avi":
            self.approxValueIteration(params)

        elif algo == "adp":
            self.approxDynamicProgramming(params, df_dec)

        else:
            logging.error("No solution algorithm could be associated")

        logging.info("Finished algorithm execution.")
        logging.info("Peace out.")
        
        
        if algo == "vi":
            self.splittime += [time.time()-self.starttime]
            self.an.addMeasure(self.key, [con.T, con.trip_max] + self.splittime, "splitrt")
        self.an.addMeasure(self.key, [con.T, con.trip_max, time.time()-self.starttime] if params is None else [con.T, con.trip_max,params[1], params[2], time.time()-self.starttime], "rt")
        
    
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
        if self.algo == "vi":
            self.splittime += [time.time()-self.starttime]
            self.an.addMeasure(self.key, [con.T, con.trip_max, len(df.index)], "tspace")

        logging.info("Finished creation of transitions. Shape of df is %s" % str(df.shape))

        #df.to_pickle("/usr/app/data/tmp/viInputDf.pkl") 

        # Call VI with state-decision-transition tuples
        print("Algo")
        return self.sol.performStandardVI(df.copy(), self.df_states["s_obj"].to_dict(), self.key)


    def approxValueIteration(self, params) -> bool:

        df = pd.merge(self.df_states, self.df_decisions, on=["s_key"])
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        logging.info("Finished joining states and decisions. Shape of df is %s"  % str(df.shape))
        
        # Call approx VI with state-decision tuples
        print("Algo")
        return self.sol.performApproxVI(df.copy(), self.df_states["s_obj"].to_dict(), self.p, params[0], params[1], params[2], self.key)


    def myopicOptimization(self) -> bool:
        df = pd.merge(self.df_states, self.df_decisions, on=["s_key"])
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        logging.info("Finished joining states and decisions. Shape of df is %s"  % str(df.shape))
        
        # Call myopic optimization algo with state-decision tuples
        print("Algo")
        return self.sol.performMyopic(df.copy(), self.df_states["s_obj"].to_dict(), self.key)


    def approxDynamicProgramming(self, params, df_dec) -> bool:
        df = self.df_states.copy()
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        # Call adp algo with state-decision tuples
        print("Algo")
        return self.sol.performApproximateDP(df.copy(), df_dec, self.df_states["s_obj"].to_dict(), self.p, params[0], params[1], params[2], self.key)

