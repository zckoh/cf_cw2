# Filename: q2transfer.py
# Date Created: 23-Mar-2019 4:08:23 pm
# Description:

import pandas as pd
from math import erf, sqrt, exp, log, pow
import matplotlib.pyplot as plt
import numpy as np
from functions import *
pd.set_option('display.max_columns', 20)

"""
Using risk-free interest rate
Obtained from UK 10-Year Bond Yield
https://uk.investing.com/rates-bonds/uk-10-year-bond-yield
"""

# load in the Call and FTSE Index spreadsheets
df_calls = pd.read_excel('FTSEOptionsData.xlsx', sheet_name='Calls')
df_puts = pd.read_excel('FTSEOptionsData.xlsx', sheet_name='Puts')
FTSE100 = pd.read_excel ('FTSEOptionsData.xlsx', sheet_name='FTSE Index')

# We process the spreadsheets
# Remove Code row
df_calls = df_calls.drop(0, axis=0)
df_puts = df_puts.drop(0, axis=0)
FTSE100 = FTSE100.drop(0,axis=0)

# Change Name column to Date
df_calls = df_calls.rename(columns={"Name": "Date"})
df_puts = df_puts.rename(columns={"Name": "Date"})
FTSE100 = FTSE100.rename(columns={"Name": "Date"})

# Change Date column to datetime
df_calls['Date'] = pd.to_datetime(df_calls['Date'])
df_puts['Date'] = pd.to_datetime(df_puts['Date'])
FTSE100['Date'] = pd.to_datetime(FTSE100['Date'])

# Remove unnecessary words in the column headers
df_calls.columns = df_calls.columns.str.replace('CALL ESX JAN19 ', '')
df_puts.columns = df_puts.columns.str.replace('PUT ESX JAN19 ', '')
# FTSE100.columns = FTSE100.columns.str.replace('FTSE 100 - PRICE INDEX', 'FTSE100')
FTSE100 = FTSE100.rename(index=str, columns = {"TR UK GVT BMK BID YLD 10Y (£) - RED. YIELD":"r",
                                                'FTSE 100 - PRICE INDEX':'FTSE100'})


# Dataframes below are used for question 2
ttm_df = pd.DataFrame(df_calls['Date'])
s_div_x = pd.DataFrame(df_calls['Date'])
c_div_x = pd.DataFrame(df_calls['Date'])
k_df = pd.DataFrame(df_calls['Date'])
dcds = pd.DataFrame(df_calls['Date'])
bs_call_df = pd.DataFrame(df_calls['Date'])

for K in range(5000,9200,20):
    ttm_df[str(K)] = np.nan
    s_div_x[str(K)] = np.nan
    bs_call_df[str(K)] = np.nan
    c_div_x[str(K)] = np.nan
    k_df[str(K)] = np.nan
    dcds[str(K)] = np.nan

total_len = 275
start_idx = 1

# For each column in Calls spreadsheet
for K in range(5000,9200,20):
    # For each row starting from T/4 + 1
    for t in range(start_idx + int(total_len/4) + 1, total_len + 1):
        # Compute the annualized volatility of the underlying asset (FTSE100)
        annual_vol = FTSE100['FTSE100'][t-int(total_len/4):t-1].pct_change(1).std() * sqrt(252)

        # Get stock price at time t
        S = FTSE100.iloc[t-1,1]

        # Get risk free interest rate at time t
        r = FTSE100.iloc[t-1,2] / 100

        # Get the ttm value (Asumming last trading day is on the third friday
        # in delivery month - 18th Jan 2019, index=277)
        ttm = (277 - t) / 365 # expressed in years

        # Find d1 and d2
        d_1 = d1(S,K,r,annual_vol,ttm)
        d_2 = d2(d_1,annual_vol,ttm)

        # Compute the call option price
        c_price = ComputeCallOptionPrice(S,K,d_1,d_2,ttm,r)

        # Store the prices for plotting later
        bs_call_df.at[t,K] = c_price

        # Used for question 2
        # Store the ttm values
        ttm_df.at[t,K] = ttm

        # Store the S/X values
        s_div_x.at[t,K]  = S/K

        # Store the C/X prices
        c_div_x.at[t,K] = c_price / K

        # Store the K values
        k_df.at[t,K] = K

        # Store the dcds values
        dcds.at[t,K] = phi(d_1)

# Used for question 2
bs_call_df.to_csv("./q2_data/bs_call_prices.csv", index = False)
ttm_df.to_csv("./q2_data/ttm_call_prices.csv", index = False)
s_div_x.to_csv("./q2_data/s_div_x.csv", index = False)
c_div_x.to_csv("./q2_data/c_div_x.csv", index = False)
k_df.to_csv("./q2_data/k_df.csv", index = False)
dcds.to_csv("./q2_data/dcds.csv", index = False)
