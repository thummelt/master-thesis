from src.modules.state import State
from src.modules.decision import Decision
from src.modules import generator as g
from src.modules.solutionAlgorithms import SolutionAlgorithms 
from src.modules.probabilities import Probabilities
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

from fnmatch import fnmatch



## Sets algorithm up. Called by Jupyter NB

class App:

    # Variables
    sol: SolutionAlgorithms
    p: Probabilities

    ## States as dataframe to speed up lookup
    df_states = pd.DataFrame(columns=["s_key", "s_obj"])

    ## Decisions as dataframe to speed up lookup
    df_decisions = pd.DataFrame(columns=["d_key", "s_key", "d_obj"])

    decisions : Dict[State, List[Decision]] = {}

    def __init__(self):

        logging.basicConfig(filename="logs" + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '_app.log',
                        filemode='w+', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

            
        # Solver
        self.sol = SolutionAlgorithms()

        # Probabilities
        self.p = Probabilities()

        # States
        self.df_states = g.constructStates()
        self.df_states.set_index("s_key", inplace=True, drop=True)

        logging.info("Finished creation of %d states" % len(self.df_states))

        # For each state (which are not terminal) construct all decisions        
        processed_list = Parallel(n_jobs=mp.cpu_count()*4)(delayed(g.constructDecisions)(i) for i in tqdm(self.df_states.loc[self.df_states["s_obj"].apply(lambda s: not s.get_isTerminal()),"s_obj"]))
        self.df_decisions = pd.concat(processed_list)
        self.df_decisions.reset_index(inplace=True,drop=True)

        logging.info("Finished creation of %d decisions" % len(self.df_decisions))

        #print(self.df_decisions.loc[(self.df_decisions["s_key"].apply(lambda k: fnmatch(k, "(1,1.?,0,0.0,0.0,0.0)")))])
        



    
    def valueIteration(self) -> bool:
        
        # Construct data frame for performance ["s_key", "s_obj", "d_key", "d_obj", "t",  "trpln", "prc", "p", "tr_obj"]
        df = pd.merge(self.df_states, self.df_decisions, on=["s_key"])
        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        #print(df.head())
        #print(df.loc[(df["s_key"].apply(lambda k: fnmatch(k, "(1,1.?,0,0.0,0.0,0.0)")))])

        logging.info("Finished joining states and decisions. Shape of df is %s"  % str(df.shape))
        
        # Construct exog information
        df = g.constructExogInfo(df, self.p)
        #print(df.loc[df["s_key"]=='(1,1.0,0,0.0,0.0,0.0)'])
        
        logging.info("Finished creation of exogenous information. Shape of df is %s"  % str(df.shape))

        # Construct transitions
        df = g.constructTransitions(df, self.df_states.index.values.tolist())
        df.reset_index(inplace=True, drop=True)
        #print(df.loc[df["s_key"]=='(1,1.0,0,0.0,0.0,0.0)'])

        logging.info("Finished creation of transitions. Shape of df is %s" % str(df.shape))


        ### TODO
        df.to_pickle("/usr/app/data/tmp/viInputDf.pkl") 
        self.df_states.to_pickle("/usr/app/data/tmp/viDFStates.pkl") 

        
        # Call VI with state-decision-transition tuples
        return self.sol.performStandardVI(df, self.df_states["s_obj"].to_dict())


    def approxValueIteration(self) -> bool:
        pass

    def myopicOptimization(self) -> bool:
        pass

    def randomDecision(self) -> bool:
        pass

