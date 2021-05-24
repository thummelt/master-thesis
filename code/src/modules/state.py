## Represents state object

class State:

    # Variables

    # Classification
    isTerm : bool = False
    visited : bool = False

    # Value
    v_n : float
    v_n_1 : float

    def __init__(self):
        v_n, v_n_1 = 0

    