# Filename: q1d.py
# Date Created: 13-Mar-2019 10:04:58 pm
# Description:
import pandas as pd
from math import erf, sqrt, exp, log, pow, pi
import matplotlib.pyplot as plt
import numpy as np
from functions import *
pd.set_option('display.max_columns', 20)

# Load in the outputs from q1b.py
# Processed actual option prices
df_calls = pd.read_pickle("./outputs/df_calls.pkl")
df_puts = pd.read_pickle("./outputs/df_puts.pkl")
FTSE100 = pd.read_pickle("./outputs/FTSE100.pkl")

# Estimated volatility for call
est_vol_df = pd.read_pickle("./outputs/est_vol_df.pkl")

# Computed BS option prices
bs_call_df = pd.read_pickle("./outputs/bs_call_df.pkl")
bs_put_df = pd.read_pickle("./outputs/bs_put_df.pkl")

# Find the

# Strike price
strike_price = '7300'
K = int(strike_price)

# date - 06/07/2018
t = 135
ttm = (277 - t ) / 365 # expressed in years
ttm
# Estimated volatility
annual_vol = est_vol_df[strike_price][t]

# Risk free interest rate
r = FTSE100.iloc[t-1,2] / 100

# Stock price on that date
S = FTSE100.iloc[t-1,1]

# Find the Call option price
d_1 = d1(S,K,r,annual_vol,ttm)
d_2 = d2(d_1,annual_vol,ttm)

# Compute the call option price
c_price = ComputeCallOptionPrice(S,K,d_1,d_2,ttm,r)

c_price
S
annual_vol
r
FTSE100.iloc[t-1,1]

df_calls[strike_price][t]
FTSE100.iloc[t-1,0]
