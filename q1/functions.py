# Filename: functions.py
# Date Created: 12-Mar-2019 11:40:09 pm
# Description: Functions used for black-scholes model.
import pandas as pd
from math import erf, sqrt, exp, log, pow
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from decimal import Decimal
import math as m
import numpy as np


def phi(x):
    """
    Cumulative distribution function for the standard normal distribution.
    """
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def TimeSeriesInfo(column):
    """
    Returns total length, when it starts when it ends.
    (total_length, start_idx, end_idx).
    """
    dropped_col = column.dropna()

    start_idx = dropped_col.index.values[0]
    end_idx = dropped_col.index.values[-1]
    total_length = dropped_col.count()

    return total_length, start_idx, end_idx

def d1(S,K,r,vol,ttm):
    """
    Compute d1 for Black-Scholes model.
    """
    d_1 = (log(S/K) + (r+pow(vol,2)/2)*(ttm)) / (vol * sqrt(ttm))
    return d_1

def d2(d_1,vol,ttm):
    """
    Compute d2 for Black-Scholes model.
    """
    d_2 = d_1 - vol * sqrt(ttm)
    return d_2

def ComputeCallOptionPrice(S,K,d_1,d_2,ttm,r):
    """
    Calculate the call price option.
    ttm (time to maturity) expressed in years.
    """
    C = S*phi(d_1) - K * exp(-r * ttm) * phi(d_2)
    return C

def ComputePutOptionPrice(C,S,K,ttm,r):
    """
    Calculate the put price option based on put-call parity.
    """
    P = K * exp(-r * ttm) - S + C
    return P

def call_bsm (S0,K,r,T,Otype,sig):
    d1 = Decimal(m.log(S0/K)) + (r+ (sig*sig)/2)*T/(sig*Decimal(m.sqrt(T)))
    d2 = d1 - sig*Decimal(m.sqrt(T))
    if (Otype == "Call"):
        price = S0*Decimal(ss.norm.cdf(np.float(d1))) \
        - K*Decimal(m.exp(-r*T))*Decimal(ss.norm.cdf(np.float(d2)))
        return (price)
    elif (Otype == "Put"):
        price  = -S0*Decimal(ss.norm.cdf(np.float(-d1)))\
        + K*Decimal(m.exp(-r*T))*Decimal(ss.norm.cdf(np.float(-d2)))
        return (price)

def vega (S0,K,r,T,sig):
    d1 = Decimal(m.log(S0/K))/(sig*Decimal(m.sqrt(T))) + Decimal((r+ (sig*sig)/2)*T/(sig*Decimal(m.sqrt(T))))
    vega = S0*Decimal(ss.norm.pdf(np.float(d1)))*Decimal(m.sqrt(T))
    return(vega)


def imp_vol(S0, K, T, r, market,flag):
    e = 10e-6; x0 = Decimal(1);
    def newtons_method(S0, K, T, r, market,flag,x0, e):
        delta = call_bsm (S0,K,r,T,flag,x0) - market
        while delta > e:
            x0 = Decimal(x0 - (call_bsm (S0,K,r,T,flag,x0) - market)/vega (S0,K,r,T,x0))
            delta = abs(call_bsm (S0,K,r,T,flag,x0) - market)
        return(Decimal(x0))
    sig =  newtons_method(S0, K, T, r, market,flag,x0 , e)
    return(sig)
