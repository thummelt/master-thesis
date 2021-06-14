#######################################################################################
# General                                                                             #
#######################################################################################

# Time intervals in h
tau: int = 0.25

# Time horizon
T: float = 2# 24/tau

# Convergence epsilon
convergence: float = 0.1

# Approximate VI smoothing factor
alpha: float = 0.25


#######################################################################################
# Vehicle                                                                             #
#######################################################################################

# Charging/Discharging efficiency
eta: float = 1#0.9

# Charging rate in kW/tau*h
my: float = 9*tau

# Energy consumption in kWh/km (actually 0.147)
ny: float = 0.2

# Battery capacity in kWh
beta: float = 5#0

# Max battery level
beta_max: float = 1*beta

# Min battery level
beta_min: float = 0.2*beta

# Energy level at end of time horizon
beta_T: float = 0.8*beta

# Initla energy level
beta_0: float = 0.8*beta

# Avg driving speed in km/h
gamma: float = 40

#######################################################################################
# Other                                                                               #
#######################################################################################

# Large constant
phi: int = 9999999

# Penalty costs for not attending trip
epsilon: float = 100 

#######################################################################################
# Model                                                                               #
#######################################################################################

# Stepsize energy amounts (battery, charging, discharging)
step_en: float = 0.1

# Stepsize price
step_pr: float = 0.1

# Max trip length
trip_max: float = 11#40

# Max price buy
price_b_max: float = 0.1

# Max price sell
price_s_max: float = 0.1

# Influence of expectation of future values
expec : float = 1.0