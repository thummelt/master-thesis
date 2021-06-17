from src.modules import constants as con

import pandas as pd
import numpy as np


## Stores probability functions for transitions

class Probabilities:

    # Const
    uncertainties = ["trpstrt", "trpln", "prc_b", "prc_s"]
    probabilities = ["p_t(t)", "p_l(t)", "p_p_b(t)", "p_p_s(t)"]

    # Variables to store measures
    d_trpstrt : pd.DataFrame
    d_len : pd.DataFrame
    d_price_b: pd.DataFrame
    d_price_s: pd.DataFrame


    def __init__(self):
        self.d_trpstrt = pd.read_pickle("/usr/app/data/probabilities/trpstrt.pkl")
        self.d_len = pd.read_pickle("/usr/app/data/probabilities/trplen.pkl")
        # Only integrating trip until given trip_max val
        self.d_len = self.d_len.loc[:,[True] + [float(c) <= con.trip_max for c in self.d_len.columns[1:]]]
        self.d_price_b = pd.read_pickle("/usr/app/data/probabilities/d_prc_b.pkl")
        self.d_price_s = pd.read_pickle("/usr/app/data/probabilities/d_prc_s.pkl")

        np.random.seed(1997)

       


    def getProbabilities(self, t: int) -> pd.DataFrame:

        dfs_y = []
        dfs_n = []

        # Trip start
        p = self.d_trpstrt.loc[self.d_trpstrt["t"] == t, "p(t)"].values[0]
        p_t_y = pd.DataFrame({self.uncertainties[0]: ["y"], self.probabilities[0]: [p]}, index=[0])
        p_t_n = pd.DataFrame({self.uncertainties[0]: ["n"], self.probabilities[0]: [1-p]}, index=[0])

        # Trip Length
        p_l_y = pd.melt(self.d_len.loc[self.d_len["t"] == t,self.d_len.columns[1:]], id_vars = [], var_name = self.uncertainties[1], value_name = self.probabilities[1]).copy()
        p_l_n = pd.DataFrame({self.uncertainties[1]: [0], self.probabilities[1]: [1]}, index=[0])

        # Electricity price (given to valid within t. therefore for transition from t to t+1 return prices from t+1 )
        p_p_b = self.d_price_b.loc[self.d_price_b["t"] == t+int(con.tau*60), ["prc", "p","t"]].copy()
        p_p_b.rename(columns={"p": self.probabilities[2], "prc": self.uncertainties[2]}, inplace=True)

        p_p_s = self.d_price_s.loc[self.d_price_s["t"] == t+int(con.tau*60), ["prc", "p","t"]].copy()
        p_p_s.rename(columns={"p": self.probabilities[3], "prc": self.uncertainties[3]}, inplace=True)

        # Construct possibilities
        dfs_y = [p_t_y, p_l_y, p_p_b, p_p_s] 
        dfs_n = [p_t_n, p_l_n, p_p_b, p_p_s] 

        return self._getProbDF(dfs_y, t)[self.uncertainties+["p"]+["t"]].append(self._getProbDF(dfs_n, t)[self.uncertainties+["p"]+["t"]], ignore_index=True)


    def getProbabilitiesSampled(self, t: int, samples: int) -> pd.DataFrame:

        dfs_y = []
        dfs_n = []

        # Trip start
        p = self.d_trpstrt.loc[self.d_trpstrt["t"] == t, "p(t)"].values[0]
        p_t_y = pd.DataFrame({self.uncertainties[0]: ["y"], self.probabilities[0]: [p]}, index=[0])
        p_t_n = pd.DataFrame({self.uncertainties[0]: ["n"], self.probabilities[0]: [1-p]}, index=[0])

        # Trip Length
        p_l_y = pd.melt(self.d_len.loc[self.d_len["t"] == t,self.d_len.columns[1:]], id_vars = [], var_name = self.uncertainties[1], value_name = self.probabilities[1]).copy()
        p_l_n = pd.DataFrame({self.uncertainties[1]: [0], self.probabilities[1]: [1]}, index=[0])

        # Electricity price (given to valid within t. therefore for transition from t to t+1 return prices from t+1 )
        p_p_b = self.d_price_b.loc[self.d_price_b["t"] == t+int(con.tau*60), ["prc", "p","t"]].copy()
        p_p_b.rename(columns={"p": self.probabilities[2], "prc": self.uncertainties[2]}, inplace=True)

        p_p_s = self.d_price_s.loc[self.d_price_s["t"] == t+int(con.tau*60), ["prc", "p","t"]].copy()
        p_p_s.rename(columns={"p": self.probabilities[3], "prc": self.uncertainties[3]}, inplace=True)

        # Construct possibilities
        dfs_y = [p_t_y, p_l_y, p_p_b, p_p_s] 
        dfs_n = [p_t_n, p_l_n, p_p_b, p_p_s] 

        # All probabilities
        df_p = self._getProbDF(dfs_y, t)[self.uncertainties+["p"]+["t"]].append(self._getProbDF(dfs_n, t)[self.uncertainties+["p"]+["t"]], ignore_index=True)

        # Sample random numbers
        df_p = df_p.loc[np.random.choice(np.arange(0, len(df_p.index)), size=samples, replace=True, p= np.divide(df_p["p"], df_p["p"].sum()))]
        df_p["p"] = 1/samples

        return df_p


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
        res["p"] = res["p"].astype(float)
        return res.copy()





    
