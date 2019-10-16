# Filename: q1c.py
# Date Created: 13-Mar-2019 1:12:39 pm
# Description:
import pandas as pd
from math import erf, sqrt, exp, log, pow, pi
import matplotlib.pyplot as plt
import numpy as np
from functions import *
pd.set_option('display.max_columns', 20)

"""
Using risk-free interest rate of 1.2% => 0.012
Obtained from UK 10-Year Bond Yield
https://uk.investing.com/rates-bonds/uk-10-year-bond-yield
"""


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

# Vary the strike prices from 5200 to 10400
# strike_prices = ['5200', '5375', '5600', '5800', '6000', '6200', '6300',
#        '6400', '6500', '6550', '6575', '6600', '6625', '6650', '6675', '6700',
#        '6725', '6750', '6775', '6800', '6825', '6850', '6875', '6900', '6925',
#        '6950', '6975', '7000', '7025', '7050', '7075', '7100', '7125', '7150',
#        '7175', '7200', '7225', '7250', '7275', '7300', '7325', '7350', '7375',
#        '7400', '7425', '7450', '7475', '7500', '7525', '7550', '7575', '7600',
#        '7625', '7650', '7700', '7800', '7900', '8000', '8100', '8200', '8400',
#        '8600', '8800', '9200', '9600', '10400']

strike_prices = ['5200', '5375', '5600', '5800', '6000', '6200', '6300',
       '6400', '6500', '6550', '6575', '6600', '6625', '6650', '6675', '6700',
       '6725', '6750', '6775', '6800', '6825', '6850', '6875', '6900', '6925',
       '6950', '6975', '7000', '7025', '7050', '7075', '7100', '7125', '7150',
       '7175', '7200', '7225', '7250', '7275', '7300', '7325', '7350', '7375',
       '7400', '7425', '8000', '8100', '8200', '8400',
       '8600', '8800', '9200', '9600', '10400']

imp_vol_call_df = pd.DataFrame().reindex_like(df_calls)
imp_vol_call_df['Date'] = df_calls['Date']

imp_vol_put_df = pd.DataFrame().reindex_like(df_puts)
imp_vol_put_df['Date'] = df_puts['Date']

start_time = 162
end_time = 192

imp_vol_put_df['Date'][192]
imp_vol_put_df['Date'][192]

for strike_price in strike_prices:
    K = int(strike_price)

    print("Processing strike price: ", K)
    for t in range(start_time,end_time):
        # Get ttm
        ttm = (277 - t) / 365

        # Get the actual call price and stock price at time t from data.
        C = df_calls[strike_price][t]
        P = df_puts[strike_price][t]

        # Get stock price at time t
        S = FTSE100.iloc[t-1,1]

        # Get risk free interest rate at time t
        r = FTSE100.iloc[t-1,2] / 100

        # Find call implied volatility using Newton's method
        iv_call = imp_vol(Decimal(S), Decimal(K), Decimal(ttm), Decimal(r), Decimal(C),'Call')
        imp_vol_call_df.at[t,strike_price] = iv_call

        # Find put implied volatiliy using Newton's method
        iv_put = imp_vol(Decimal(S), Decimal(K), Decimal(ttm), Decimal(r), Decimal(C),'Put')
        imp_vol_put_df.at[t,strike_price] = iv_put

# Plot 3 different plots of the volatility against time
# For K = 5200
compare_plt = pd.concat([df_calls['Date'], imp_vol_call_df['5200'], est_vol_df['5200']], axis=1)
compare_plt.columns = ['Date','Implied', 'Estimated']
ax = compare_plt.iloc[start_time:end_time].plot(x='Date', marker='o',
                y=['Implied','Estimated'], figsize=(7,3), grid=True)
ax.set_ylabel("Volatility (σ)")
ax.set_xlabel("")
plt.savefig('./plots/5200impest.png',dpi=300, bbox_inches='tight', pad_inches=0)

# For K = 7375
compare_plt = pd.concat([df_calls['Date'], imp_vol_call_df['7375'], est_vol_df['7375']], axis=1)
compare_plt.columns = ['Date','Implied', 'Estimated']
ax = compare_plt.iloc[start_time:end_time].plot(x='Date', marker='o',
                y=['Implied','Estimated'], figsize=(7,3), grid=True)
ax.set_ylabel("Volatility (σ)")
ax.set_xlabel("")
plt.savefig('./plots/7375impest.png',dpi=300, bbox_inches='tight', pad_inches=0)

# For K = 8000
compare_plt = pd.concat([df_calls['Date'], imp_vol_call_df['8000'], est_vol_df['8000']], axis=1)
compare_plt.columns = ['Date','Implied', 'Estimated']
ax = compare_plt.iloc[start_time:end_time].plot(x='Date', marker='o',
                y=['Implied','Estimated'], figsize=(7,3), grid=True)
ax.set_ylabel("Volatility (σ)")
ax.set_xlabel("")
plt.savefig('./plots/8000impest.png',dpi=300, bbox_inches='tight', pad_inches=0)

# Now plot the volatility smile
ax = imp_vol_call_df[strike_prices].mean().plot(marker='o',figsize=(7,3), grid=True)
ax = imp_vol_put_df[strike_prices].mean().plot(marker='o',figsize=(7,3), grid=True)
ax.set_ylabel("Implied Volatility")
ax.set_xlabel("Strike Price")
plt.savefig('./plots/volatilitysmile_putcall.png',dpi=300, bbox_inches='tight', pad_inches=0)

print("lowest IV for call option: ",imp_vol_call_df[strike_prices].mean().idxmin())
print("30 day average of FTSE100:", FTSE100['FTSE100'][start_time:end_time].mean())
