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
        self.df_states.reset_index(inplace=True,drop=True)
        logging.info("Finished creation of %d states" % len(self.df_states))

        # For each state all decisions        
        processed_list = Parallel(n_jobs=mp.cpu_count())(delayed(g.constructDecisions)(i) for i in tqdm(self.df_states["s_obj"]))
        self.df_decisions = pd.concat(processed_list)
        self.df_decisions.reset_index(inplace=True,drop=True)
        
        logging.info("Finished creation of %d decisions" % len(self.df_decisions))


        # Prune data frame 
        ## States without decisions, but not terminal

    
    def valueIteration(self) -> bool:
        # Prepare data    
        
        # Construct data frame for performance ["s_key", "s_obj", "d_key", "d_obj", "t",  "trpln", "prc", "p", "tr_obj"]
        df = pd.merge(self.df_states, self.df_decisions, on=["s_key"])

        df["t"] = df["s_obj"].apply(lambda s: s.get_t())

        # Construct ex_info data frame for all t
        df_p = pd.DataFrame()

        processed_list = Parallel(n_jobs=mp.cpu_count())(delayed(self.p.getProbabilities)(t*con.tau*60) for t in np.arange(0, con.T+1, 1))
        df_p = pd.concat(processed_list)

        # Need to match time index
        df_p["t"] = df_p["t"]/(con.tau*60)
        df_p.reset_index(inplace=True,drop=True)
              
        df = pd.merge(df,df_p, on=["t"]).copy()

        # Construct transitions
        df = g.constructTransitions(df, self.df_states["s_key"].tolist())
        
        # Call VI with state-decision-transition tuples
        return df # TODO
        # return self.sol.performStandardVI(df, self.df_states["s_obj"].tolist())


    def approxValueIteration(self) -> bool:
        pass

    def myopicOptimization(self) -> bool:
        pass

    def randomDecision(self) -> bool:
        pass

