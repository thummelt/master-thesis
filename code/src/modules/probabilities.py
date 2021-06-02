from src.modules import constants as con

import pandas as pd
import numpy as np


## Stores probability functions for transitions

class Probabilities:

    # Const
    uncertainties = ["trpstrt", "trpln", "prc"]
    probabilities = ["p_t(t)", "p_l(t)", "p_p(t)"]

    # Variables to store measures
    d_trpstrt : pd.DataFrame
    d_len : pd.DataFrame
    d_price: pd.DataFrame


    def __init__(self):
        self.d_trpstrt = pd.read_pickle("/usr/app/data/probabilities/trpstrt.pkl")
        self.d_len = pd.read_pickle("/usr/app/data/probabilities/trplen.pkl")
        #self.d_price = pd.read_pickle("/usr/app/data/probabilities/price.pkl")

        # TODO later per real data
        ran = [np.random.random(size=None) for x in np.arange(0, 1441/con.tau)]
        self.d_price = pd.DataFrame({"t": np.arange(0, 1441/con.tau), "0.1": ran, "0.0": [1-r for r in ran]})
        


    def getProbabilities(self, t: int) -> pd.DataFrame:

        dfs_y = []
        dfs_n = []

        # Trip start
        p = self.d_trpstrt.loc[self.d_trpstrt["t"] == t, "p(t)"].values[0]
        p_t_y = pd.DataFrame({self.uncertainties[0]: ["y"], self.probabilities[0]: [p]}, index=[0])
        p_t_n = pd.DataFrame({self.uncertainties[0]: ["n"], self.probabilities[0]: [1-p]}, index=[0])

        # Trip Length
        p_l_y = pd.melt(self.d_len.loc[self.d_len["t"] == t,self.d_len.columns[1:]], id_vars = [], var_name = self.uncertainties[1], value_name = self.probabilities[1]).copy()
        p_l_n = pd.DataFrame({self.uncertainties[1]: ["0"], self.probabilities[1]: [1]}, index=[0])

        # Electricity price
        p_p = pd.melt(self.d_price.loc[self.d_price["t"] == t,self.d_price.columns[1:]], id_vars = [], var_name = self.uncertainties[2], value_name = self.probabilities[2]).copy()

        # Construct possibilities
        dfs_y = [p_t_y, p_l_y, p_p]
        dfs_n = [p_t_n, p_l_n, p_p]

        return self._getProbDF(dfs_y, t)[self.uncertainties+["p"]+["t"]].append(self._getProbDF(dfs_n, t)[self.uncertainties+["p"]+["t"]], ignore_index=True)


    def _getProbDF(self, dfs: list, t: int) -> pd.DataFrame:
        res = pd.DataFrame({"key": 0}, index=[0])

        for df in dfs:
            df["key"] = 0
            res = pd.merge(res,df, on=['key'],  suffixes=('', '_del')).copy()
        
        res.drop(columns=["key"], inplace=True)
        res["p"] = 1
        for p in self.probabilities:
            res["p"] = res["p"]*res[p]
        
        res["t"] = t
        return res.copy()





    
