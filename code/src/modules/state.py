from src.modules import constants as con

## Represents state object

class State:

    # Variables
    t : float
    B_L : float
    V_TA : int
    D : float
    P_B : float
    P_S : float

    # Classification
    isTerminal : bool = False
    visited : bool = False

    # Value
    v_n : float
    v_n_1 : float

    def __init__(self, t:int, b_l:float, v_ta: int,  d:float, p_b:float, p_s:float):
        self.t = t
        self.B_L = b_l
        self.V_TA = v_ta

        # Exogenous information that has arrived until t => incorporate for decision making
        self.D = d
        self.P_B = p_b
        self.P_S = p_s

        # Terminal state if at end of horizon and target energy level is met and vehicle is at home
        self.isTerminal = ((self.B_L >= con.beta_T) & (self.V_TA == 0) & (self.t == con.T))
                          

        # Initialize values to 0 as goal is to maximize
        self.v_n = 0
        self.v_n_1 = 0

    def __str__(self) -> str:
        return "State t=%d[(%s,%s, %s,%s,%s) - %s - (%f, %f)]" % ( self.t, self.B_L, self.V_TA, self.D, self.P_B, self.P_S, self.isTerminal, self.v_n, self.v_n_1)

    def hasConverged(self, eps: float) -> bool:
        return (self.v_n-self.v_n_1) < eps

    def getY(self) -> int:
        """Return helping variable y_t describing whether vehicle is driving (= 1) or not (= 0)

        Returns:
            int: Driving status in dependence of V_TA
        """        
        return 1 if self.V_TA > 0 else 0

    
    #######################################################################################
    # Getter and setter                                                                   #
    #######################################################################################

    def getPDRepresentation(self) -> list:
        return [self.getKey(), self.t, self.B_L, self.V_TA, self.D, self.P_B, self.P_S, self.isTerminal, self.v_n, self]

    def getKey(self) -> str:
        return "(%d,%d,%s,%s,%s,%s)" % ( self.t, self.B_L, self.V_TA, self.D, self.P_B, self.P_S ) 

    def get_t(self) -> int:
        return self.t

    def get_B_L(self) -> float:
        return self.B_L
    
    def get_V_TA(self) -> int:
        return self.V_TA

    def get_D(self) -> float:
        return self.D
    
    def get_P_B(self) -> float:
        return self.P_B
    
    def get_P_S(self) -> float:
        return self.P_S

    def get_isTerminal(self) -> bool:
        return self.isTerminal
    
    def get_V_N(self) -> float:
        return self.v_n

    def get_V_N_1(self) -> float:
        return self.v_n_1

    ##

    def set_B_L(self, x : float):
        self.B_L = x
    
    def set_V_TA(self, x : int):
        self.V_TA = x

    def set_D(self, x : float):
        self.D = x
    
    def set_P_B(self, x : float):
        self.P_B = x
    
    def set_P_S(self, x : float):
        self.P_S = x
    
    def set_isTerminal(self, x : bool):
        self.isTerminal = x
    
    def set_V_N(self, x : float):
        self.v_n_1 = self.v_n_1
        self.v_n = x

    def __eq__ (self, s) -> bool:
        return self.getKey() == s.getKey()

    

    