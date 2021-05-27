
from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition
from src.modules import constants as con

import pandas as pd

## Stores measures and provide evaluation functionality

class Analysis:

    # Variables to store measures

    def __init__(self):
        pass

    def extractPolicy(self, t: pd.Dataframe ):
        # Extract for each state transition data (decision, ex_info)
        for s in t["s"]:
            pass

        
        # Plot time - state of charge

        # Plot time - (charging/discharging & price_b/price_s)

        # Plot time - (start trip & trip length)
