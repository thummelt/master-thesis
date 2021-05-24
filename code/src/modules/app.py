from modules.state import State
from modules.decision import Decision

import pandas as pd


## Sets algorithm up. Called by Jupyter NB

class App:

    # Variables
    states = pd.DataFrame(columns=["key", "t", "B_L", "D", "P_B", "P_S", "terminal", "v_n"])

    states: list(State) = []
    decisions : dict(State, list(Decision)) = {}

    def __init__(self):
        # States

        # For each state all decisions

        # For each state and decision pair perform transition

        # Call VI with states, state-decision pairs and transitions
        pass
    
    def setScenario(self):
        pass

    def prepare(self):
        pass

    def getStateByKey(self, key : str) -> bool:
        return str in self.states["key"]


