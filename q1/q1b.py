# Filename: q1b.py
# Date Created: 12-Mar-2019 2:40:46 pm
# Author: zckoh
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


# Create a dataframe to store the output price
bs_call_df = pd.DataFrame().reindex_like(df_calls)
bs_call_df['Date'] = df_calls['Date']

bs_put_df = pd.DataFrame().reindex_like(df_puts)
bs_put_df['Date'] = df_puts['Date']

# Create a dataframe to store the annualised volatility and implied volatility
est_vol_df = pd.DataFrame().reindex_like(df_calls)
est_vol_df['Date'] = df_calls['Date']


# For each column in Calls spreadsheet
for col in df_calls.columns[1:]:
    # Get the strike price from the column header
    K = int(col)
    # Find the TimeSeriesInfo
    total_len, start_idx, end_idx = TimeSeriesInfo(df_calls[col])

    # For each row starting from T/4 + 1
    for t in range(start_idx + int(total_len/4) + 1, end_idx + 1):
        # Compute the annualized volatility of the underlying asset (FTSE100)
        annual_vol = FTSE100['FTSE100'][t-int(total_len/4):t-1].pct_change(1).std() * sqrt(252)

        # Get stock price at time t
        # S = FTSE100['FTSE100'][t]
        S = FTSE100.iloc[t-1,1]

        # Get risk free interest rate at time t
        # r = FTSE100['r'][t]
        r = FTSE100.iloc[t-1,2] / 100

        # Get the ttm value (Asumming last trading day is on the third friday
        # in delivery month - 18th Jan 2019, index=277)
        ttm = (277 - t ) / 365 # expressed in years

        # Find d1 and d2
        d_1 = d1(S,K,r,annual_vol,ttm)
        d_2 = d2(d_1,annual_vol,ttm)

        # Compute the call option price
        c_price = ComputeCallOptionPrice(S,K,d_1,d_2,ttm,r)

        # Compute the put option price
        p_price = ComputePutOptionPrice(c_price,S,K,ttm,r)

        # Store the prices for plotting later
        bs_call_df.at[t,col] = c_price
        bs_put_df.at[t,col] = p_price

        # Store the estimated volatility
        est_vol_df.at[t,col] = annual_vol


# Now compare the prices by plotting both on same figure
compare_plt = pd.concat([df_calls['Date'], df_calls['4000'], bs_call_df['4000']], axis=1)
compare_plt.columns = ['Date','Actual', 'Black-Scholes']

ax = compare_plt.plot(x='Date', y=['Actual','Black-Scholes'], figsize=(5,3), grid=True)
ax.set_ylabel("Call Option Price (£)")
plt.savefig('./plots/4000call.png',dpi=300, bbox_inches='tight', pad_inches=0)

compare_plt = pd.concat([df_calls['Date'], df_calls['6600'], bs_call_df['6600']], axis=1)
compare_plt.columns = ['Date','Actual', 'Black-Scholes']

ax = compare_plt.plot(x='Date', y=['Actual','Black-Scholes'], figsize=(5,3), grid=True)
ax.set_ylabel("Call Option Price (£)")
plt.savefig('./plots/6600call.png',dpi=300, bbox_inches='tight', pad_inches=0)

compare_plt = pd.concat([df_calls['Date'], df_calls['9600'], bs_call_df['9600']], axis=1)
compare_plt.columns = ['Date','Actual', 'Black-Scholes']

ax = compare_plt.plot(x='Date', y=['Actual','Black-Scholes'], figsize=(5,3), grid=True)
ax.set_ylabel("Call Option Price (£)")
plt.savefig('./plots/9600call.png',dpi=300, bbox_inches='tight', pad_inches=0)

# Put Option Pricing
# Now compare the prices by plotting both on same figure
compare_plt = pd.concat([df_puts['Date'], df_puts['4000'], bs_put_df['4000']], axis=1)
compare_plt.columns = ['Date','Actual', 'Black-Scholes']

ax = compare_plt.plot(x='Date', y=['Actual','Black-Scholes'], figsize=(5,3), grid=True)
ax.set_ylabel("Put Option Price (£)")
plt.savefig('./plots/4000put.png',dpi=300, bbox_inches='tight', pad_inches=0)

compare_plt = pd.concat([df_puts['Date'], df_puts['6600'], bs_put_df['6600']], axis=1)
compare_plt.columns = ['Date','Actual', 'Black-Scholes']

ax = compare_plt.plot(x='Date', y=['Actual','Black-Scholes'], figsize=(5,3), grid=True)
ax.set_ylabel("Put Option Price (£)")
plt.savefig('./plots/6600put.png',dpi=300, bbox_inches='tight', pad_inches=0)

compare_plt = pd.concat([df_puts['Date'], df_puts['9600'], bs_put_df['9600']], axis=1)
compare_plt.columns = ['Date','Actual', 'Black-Scholes']

ax = compare_plt.plot(x='Date', y=['Actual','Black-Scholes'], figsize=(5,3), grid=True)
ax.set_ylabel("Put Option Price (£)")
plt.savefig('./plots/9600put.png',dpi=300, bbox_inches='tight', pad_inches=0)

# Save the necessary data for the next question
# Processed actual option prices
df_calls.to_pickle("./outputs/df_calls.pkl")
df_puts.to_pickle("./outputs/df_puts.pkl")
FTSE100.to_pickle("./outputs/FTSE100.pkl")

# Estimated volatility for call
est_vol_df.to_pickle("./outputs/est_vol_df.pkl")

# Computed BS option prices
bs_call_df.to_pickle("./outputs/bs_call_df.pkl")
bs_put_df.to_pickle("./outputs/bs_put_df.pkl")
