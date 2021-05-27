
## Represents decision object

class Decision:

    # Variables
    x_G2V : float
    x_V2G : float
    x_t : int

    def __init__(self, x_g2v:float, x_v2g:float, x_t:int):
        self.x_G2V = x_g2v
        self.x_V2G = x_v2g
        self.x_t = x_t
        

    def __str__(self) -> str:
        return "Decision [(%f,%f,%d)]" % ( self.x_G2V, self.x_V2G, self.x_t)

    
    
    #######################################################################################
    # Getter and setter                                                                   #
    #######################################################################################

    def get_x_G2V(self) -> float:
        return self.x_G2V

    def get_x_V2G(self) -> float:
        return self.x_V2G
    
    def get_x_t(self) -> int:
        return self.x_t

    ##

    def set_x_G2V(self, x : float):
        self.x_G2V = x

    def set_x_V2G(self, x : float):
        self.x_V2G = x
    
    def set_x_t(self, x : int):
        self.x_t = x


    def __eq__ (self, d) -> bool:
        return (self.get_x_G2V() == d.get_x_G2V()) and (self.get_x_V2G() == d.get_x_V2G()) and (self.get_x_t() == d.get_x_t())
       