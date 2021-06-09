
from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition
from src.modules import constants as con

import pandas as pd


from datetime import datetime

## Stores measures and provide evaluation functionality

class Analysis:

    # Variables to store measures
    runtime = pd.DataFrame(columns=["time"])
    splitRuntime = pd.DataFrame(columns=["t_state", "d_state", "tr_state", "vi"])
    stateSpace = pd.DataFrame(columns=["amount"])
    decisionSpace = pd.DataFrame(columns=["amount"])
    totalSpace = pd.DataFrame(columns=["amount"])

    allVar = []

    def __init__(self):
        self.allVar = [("rt", self.runtime), ("splitrt", self.splitRuntime), ("sspace", self.stateSpace), ("dspec", self.decisionSpace), ("tspace", self.totalSpace)]

    def addMeasure(self, key, value, m):
        if m == "rt":
            self.runtime.loc[key] = [value]
            return
        if m == "splitrt":
            self.splitRuntime.loc[key] = value
            return
        if m == "sspace":
            self.stateSpace.loc[key] = [value]
            return
        if m == "dspace":
            self.decisionSpace.loc[key] = [value]
            return
        if m == "tspace":
            self.totalSpace.loc[key] = [value]
            return

        
    def putout(self):
        # Spool out dataframe pkl
        for var in self.allVar:
            var[1].to_pickle("/usr/app/data/tmp/%s.pkl" % var[0])

        # Spool out excel 
        with pd.ExcelWriter('/usr/app/output/analysis%s.xlsx' % datetime.now().strftime("%Y%m%d_%H%M%S")) as writer:  
            for var in self.allVar:
                var[1].to_excel(writer, sheet_name=var[0])
            


    



