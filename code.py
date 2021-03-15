# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:48:40 2021

@author: marco
"""
#Library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA as arima
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import warnings

plt.style.use('dark_background')
#rc('text', usetex=True)
warnings.filterwarnings("ignore")
rc('xtick', labelsize=20) 
rc('ytick', labelsize=20) 
#%%
#Initialising random number generator with my student number
#np.random.seed(17088101)
generator = np.random.default_rng(17088101)
#generator = np.random.default_rng(88888888)
#Generating price series
t = np.arange(0,2001,1)
y = pd.Series(np.zeros(shape=(2001)))
y[0] = 100
y[1] = 100
phi = 0.6
d = 0.025
theta = -0.4
#epsilon = np.random.randn(2001)
epsilon = generator.normal(size=2001)
for i in range(2, len(y)):
    y[i] = phi*(y[i-1]-y[i-2]-d)+epsilon[i]+theta*epsilon[i-1]+d+y[i-1]

#%%
plt.figure(figsize=(9,7))
plt.plot(t,y,linewidth=1)
plt.title('Generated asset price time series',
         fontsize=30)
plt.ylabel(r'$y_t$', fontsize=20)
plt.xlabel('$t$', fontsize=20)
plt.fill_between(t, y.min()-10, y, alpha=0.3)
plt.xlim([t.min(), t.max()])
plt.ylim([y.min()-10, y.max()])
plt.grid(linewidth=0.3)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)

#%%
#Splitting into train and test
train_size = int(0.7*len(y))
test_size = int(0.3*len(y))
train = y[:train_size]
test = y[train_size:]
plt.plot(train, linewidth=1, label='train')
plt.plot(test, linewidth=1, color='orange', label='test')
plt.fill_between(np.arange(len(train)), y.min()-10, train, alpha=0.3)
plt.fill_between(np.arange(len(train), len(y)), y.min()-10, test, alpha=0.3, 
                color='orange')
plt.title('Time series train and test split',
         fontsize=20)
plt.ylabel(r'$y_t$',
          fontsize=16)
plt.xlabel('$t$', 
          fontsize=16)
plt.xlim([t.min(), t.max()])
plt.ylim([y.min()-10, y.max()])
plt.grid(linewidth=0.3)
plt.legend(fontsize=15)

#Reindexing for simplicity 
test = test.reset_index(drop=True)
#%% DEFINING TREND FOLLOWING
def trend_following_strategy_v2(data, time_window_short=3, 
                            time_window_long=6,
                            starting_cash=10000,
                            verbose=False):
    '''
    computes the portfolio value and logarithmic returns of a moving average cross-over strategy
    test = data
    time_window_short = short moving window
    time_window_long = long moving window
    starting_cash = starting capital employed in this self-financing strategy
    '''
    ma_short = data.rolling(window=time_window_short).mean()
    ma_long = data.rolling(window=time_window_long).mean()
    w = np.zeros(np.shape(data))
    C = np.zeros(np.shape(data))

    C[:time_window_long] = starting_cash # setting starting cash for the first x days
        #where x is the number of periods of the longer moving average
    for i in range(time_window_long-1, len(data)):
        if i < len(w)-1:
            #checking condition
            if ma_short[i] == ma_long[i]:
            #print('Do nothing')
            #this means that nothing should happen
                w[i+1] = w[i]
                C[i+1] = C[i]
            if ma_long[i] < ma_short[i]:
                #print('Buy asset')
                #trending upwards
                w[i+1] = C[i]/data[i]  + w[i] #we allocate all cash on the asset
                C[i+1] = 0 
                if verbose:
                    print('Long on asset at {}!'.format(data[i]))
                    print('Long_MA: {}, Short_MA: {}, w: {}, C: {}'.format(
                                        ma_long[i], ma_short[i], w[i+1], C[i+1]))
            if ma_long[i] > ma_short[i]:
                #print('Sell asset')
                C[i+1] = w[i]*data[i] + C[i] #we are allocating 0 cash on the asset
                w[i+1] = 0 
                if verbose:
                    print('Short on asset at {}!'.format(data[i]))
                    print('Long_MA: {}, Short_MA: {}, w: {}, C: {}'.format(
                                        ma_long[i], ma_short[i], w[i+1], C[i+1]))
    tf_strategy = pd.Series([a*b for a,b in zip(w,data)]+ C)   
    log_ret = np.log(tf_strategy) - np.log(tf_strategy.shift(1))
    cumulative_returns = (np.cumprod(1 + log_ret) - 1).iloc[-1]
    return tf_strategy, log_ret, cumulative_returns

def compute_benchmark(data, starting_cash=10000):
    '''
    computes the benchmark returns and portfolio value of a buy and hold strategy of an asset
    data = time series data to buy and hold
    starting_cash = starting capital employed in this self-financing strategy
    '''
    benchmark_time_series = starting_cash*data/data.iloc[0]
    log_ret = np.log(benchmark_time_series) - np.log(benchmark_time_series.shift(1))
    cumulative_returns = (np.cumprod(1 + log_ret) - 1).iloc[-1]
    return benchmark_time_series, log_ret, cumulative_returns

#%% OPTIMISING TREND FOLLOWING
short_time_windows = np.arange(3, 15, 1)
long_time_windows = [2*i for i in short_time_windows]
returns_matrix = np.zeros(shape=(len(short_time_windows), 
                                len(long_time_windows)))
#vertical = short time window
#horizontal = long time window
for i in range(len(short_time_windows)):
    for j in range(len(long_time_windows)):
        if i < j:
            tf_1, returns_trendfollowing, r_trendfollowing = trend_following_strategy_v2(train, 
                                                                    time_window_short=short_time_windows[i], 
                                                                    time_window_long=long_time_windows[j])
            returns_matrix[i, j] = r_trendfollowing
            
np.max(returns_matrix), compute_benchmark(train)[2]
optimal_short_window = short_time_windows[
            int(np.where(returns_matrix==np.max(returns_matrix))[0])
            ]
optimal_long_window = long_time_windows[
            int(np.where(returns_matrix==np.max(returns_matrix))[0])
            ]
print('The optimal window lengths in sample are {} and {}'.format(optimal_short_window, optimal_long_window))

tf_insample, returns_trendfollowing_insample, r_trendfollowing_insample = trend_following_strategy_v2(train, 
                                                            time_window_short=optimal_short_window, 
                                                            time_window_long=optimal_long_window,
                                                            verbose=False)
tf_oosample, returns_trendfollowing_oosample, r_trendfollowing_oosample = trend_following_strategy_v2(test, 
                                                            time_window_short=optimal_short_window, 
                                                            time_window_long=optimal_long_window,
                                                            verbose=False)
benchmark_timeseries_insample, benchmark_returns_insample, r_benchmark_insample = compute_benchmark(train, 
                                                            starting_cash=10000)
benchmark_timeseries_oosample, benchmark_returns_oosample, r_benchmark_oosample = compute_benchmark(test, 
                                                            starting_cash=10000)


rc('xtick', labelsize=18) 
rc('ytick', labelsize=18) 

fig, axs = plt.subplots(1,2, figsize=(14,6))
axs[0].plot(np.arange(len(train)), tf_insample, 
        label='trend following (train)',
        linewidth=1.5)
axs[0].plot(np.arange(len(train)), benchmark_timeseries_insample,
        label='benchmark (train)',
        linewidth=1.5,
        color='orange')
axs[0].set_xlabel('t', fontsize=18)
axs[0].set_ylabel('portfolio value', fontsize=18)
axs[0].legend(fontsize=16)
axs[0].grid(linewidth=0.3)
axs[1].plot(np.arange(len(test)), tf_oosample, 
        label='trend following (test)',
        linewidth=1.5)
axs[1].plot(np.arange(len(test)), benchmark_timeseries_oosample,
        label='benchmark (test)',
        linewidth=1.5,
        color='orange')
axs[1].set_xlabel('t', fontsize=18)
axs[1].set_ylabel('portfolio value', fontsize=18)
axs[1].grid(linewidth=0.3)
axs[1].legend(fontsize=16)
fig.tight_layout()

#%% DEFINING MEAN REVERTING
def mean_reversion_ewma(data, window=10, starting_cash=10000,
                       verbose=False):
    '''
    performs a mean-reverting strategy baed on EWMA
    '''
    ewma = data.ewm(span=window).mean()
    w = np.zeros(np.shape(data))
    C = np.zeros(np.shape(data))
    C[:window] = starting_cash # setting starting cash for the first x days
    #where x is the number of periods of the longer moving average
    for i in range(window-1, len(data)):
        if i < len(w)-1:
            #check if asset is held, if it is, sell it 
            if ewma[i] >= data[i]:
                #print('Buy asset')
                #trending upwards
                w[i+1] = C[i]/data[i]  + w[i] #we allocate all cash on the asset
                C[i+1] = 0 
                if verbose:
                    print('Long on asset at {}!'.format(data[i]))
                    print('ewma: {}, price: {}, w: {}, C: {}'.format(
                                        ewma[i], data[i], w[i+1], C[i+1]))
            else:
                w[i+1] = w[i]
                C[i+1] = C[i]
            if w[i]!=0:
                if ewma[i] < data[i]:
                    C[i+1] = w[i]*data[i] + C[i] #we are allocating 0 cash on the asset
                    w[i+1] = 0 
                    if verbose:
                        print('Short on asset at {}!'.format(data[i]))
                        print('ewma: {}, asset: {}, w: {}, C: {}'.format(
                                            ewma[i], data[i], w[i+1], C[i+1])) 
            
    tf_strategy = pd.Series([a*b for a,b in zip(w,data)]+ C)   
    log_ret = np.log(tf_strategy) - np.log(tf_strategy.shift(1))
    cumulative_returns = np.cumprod(1 + log_ret) - 1
    return tf_strategy, log_ret, cumulative_returns

#%% OPTIMISING MEAN-REVERTING
windows = np.arange(10,20)
opt_returns_array = np.zeros(len(windows))
for i in range(len(windows)):
    mr, returns_mr, r_mr = mean_reversion_ewma(train, window=windows[i])
    opt_returns_array[i] = r_mr.iloc[-1]

np.max(opt_returns_array)
optimal_window = windows[
            int(np.where(np.max(opt_returns_array[opt_returns_array!=0]))[0])
            ]
print(optimal_window)

mr_insample, returns_mr_insample, r_mr_insample = mean_reversion_ewma(train, 
                                                            window=optimal_window,
                                                            verbose=False)
mr_oosample, returns_mr_oosample, r_mr_oosample = mean_reversion_ewma(test, 
                                                            window=optimal_window, 
                                                            verbose=False)
benchmark_timeseries_insample, benchmark_returns_insample, r_benchmark_insample = compute_benchmark(train, 
                                                            starting_cash=10000)
benchmark_timeseries_oosample, benchmark_returns_oosample, r_benchmark_oosample = compute_benchmark(test, 
                                                            starting_cash=10000)
#%%
rc('xtick', labelsize=18) 
rc('ytick', labelsize=18) 

fig, axs = plt.subplots(1,2, figsize=(14,6))
axs[0].plot(np.arange(len(train)), mr_insample, 
        label='mean-reverting (train)',
        linewidth=1.5)
axs[0].plot(np.arange(len(train)), benchmark_timeseries_insample,
        label='benchmark (train)',
        linewidth=1.5,
        color='orange')
axs[0].set_xlabel('t', fontsize=18)
axs[0].set_ylabel('portfolio value', fontsize=18)
axs[0].legend(fontsize=16)
axs[0].grid(linewidth=0.3)
axs[1].plot(np.arange(len(test)), mr_oosample, 
        label='mean-reverting (test)',
        linewidth=1.5)
axs[1].plot(np.arange(len(test)), benchmark_timeseries_oosample,
        label='benchmark (test)',
        linewidth=1.5,
        color='orange')
axs[1].set_xlabel('t', fontsize=18)
axs[1].set_ylabel('portfolio value', fontsize=18)
axs[1].grid(linewidth=0.3)
axs[1].legend(fontsize=16)
fig.tight_layout()
#%% CREATING AUTOREGRESSIVE MODEL
def train_AR(data, time_window=10, starting_cash=10000):
    '''
    trains an autoregressive model 
    window = window of data to train on
    data = initial data input 
    starting_cash = starting capital to invest
    '''
    ar_prediction = np.zeros(np.shape(data))
    data_diff = np.diff(data)
    diff_prediction = np.zeros(np.shape(data_diff))
    for i, x in enumerate(data_diff[:-1], 0):
        diff_prediction[i] = x #first prediction set to initial value
        if i>=time_window: #The amount of observations to train is time window
            X = data_diff[:i] 
            train = X 
            # train autoregression
            #model = AR(train)
            model = AutoReg(train, lags=1)
            model_fit = model.fit()
            #a prediction will be created ONLY at the index of the length of the given trained set
            predictions = model_fit.predict(start=len(train), end=len(train), dynamic=False)
            diff_prediction[i] = predictions[0]
    #Undifferencing the data
    ar_prediction[0] = data[0]
    for i in range(len(ar_prediction)-1):
        ar_prediction[i+1] = data[i] + diff_prediction[i]
    return diff_prediction, ar_prediction

def trade_AR(original_data, starting_cash=10000,
            time_window=10):
    '''
    will trade based on the previously predicted values in a trend-following manner
    this means if the price is higher than the predicted price, the algorithm will buy
    original_data = original timeseries
    time_window = time window to train the AR model with
    starting_cash = initial capital to invest in the self-financing strategy
    '''
    ar_prediction = train_AR(original_data,
                            time_window=time_window)[1]
    w = np.zeros(np.shape(ar_prediction))
    cash = np.zeros(np.shape(ar_prediction))
    cash[0] = starting_cash #initial cash set to this
    for i, x in enumerate(original_data[:-1], 0):
        if ar_prediction[i] == x:
            w[i+1] = w[i]
            cash[i+1] = cash[i]
        if ar_prediction[i] < x: 
            w[i+1] = cash[i]/x  + w[i]
            cash[i+1] = 0
        if ar_prediction[i] > x:
            cash[i+1] = w[i]*x + cash[i]
            w[i+1] = 0
    ar_strategy = [a*b for a,b in zip(w,original_data)]+ cash
    return ar_strategy, ar_prediction

def compute_log_ret(timeseries):
    '''
    function to compute logarithmic returns for each time step
    inputs a time series of portfolio values and returns logarithmic returns
    and cumulative returns over the period
    '''
    series = pd.Series(timeseries)
    log_ret = np.log(series) - np.log(series.shift(1))
    cumulative_returns = (np.cumprod(1 + log_ret) - 1).iloc[-1]
    return log_ret, cumulative_returns

#%% Testing for weak stationarity through ADF test
data = train
data_diff = np.diff(data)
from statsmodels.tsa.stattools import adfuller
result = adfuller(data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

result = adfuller(data_diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

#%%

rc('xtick', labelsize=18) 
rc('ytick', labelsize=18) 

diff_trainset = np.diff(train)
pred_differences_ar = train_AR(train, time_window=20)[0]
plt.plot(diff_trainset, label='actual differences (train)',
          color='orange')
plt.plot(pred_differences_ar, 
         label='predicted differences (train)')
plt.legend(fontsize=14)
plt.grid(linewidth=0.3)
plt.tight_layout()

#%% OPTIMISING AR strategy
windows = [10, 20, 50, 100, 200]
cum_ret_ar = np.zeros(len(windows))
for i in range(len(windows)):
    returns_log, cumulative_log = compute_log_ret(trade_AR(train, 
                                                time_window=windows[i])[0])
    cum_ret_ar[i] = cumulative_log
print(cum_ret_ar)

fig, axs = plt.subplots(1,2, figsize=(12,5))
cash = np.zeros(np.shape(train))
T = len(train)
N = len(train)
t = np.linspace(0, T, N)
cash[0] = 10000
axs[0].plot(t, cash[0]*train/train[0],
            label='benchmark (train)', color='orange',
            linewidth=1.5)
axs[0].plot(t, trade_AR(train, time_window=20)[0],
        label='Autoregressive Model (train)',
        linewidth=1.5)
axs[0].legend(fontsize=12)
axs[0].grid(linewidth=0.3)

T = len(test)
N = len(test)
t = np.linspace(0, T, N)
cash = np.zeros(np.shape(test))
cash[0] = 10000
axs[1].plot(t, cash[0]*test/test[0],
        label='benchmark (test)', color='orange',
        linewidth=1.5)
axs[1].plot(t, trade_AR(test, time_window=20)[0],
        label='Autoregressive Model (test)',
        linewidth=1.5)
axs[1].legend(fontsize=12)
axs[1].grid(linewidth=0.3)

#Defining the 2 timeseries to compute later ratios
ar_strat_insample = trade_AR(train, time_window=20)[0]
T = len(train)
N = len(train)
t_ar_insample = np.linspace(0, T, N)

ar_strat_oosample = trade_AR(test, time_window=20)[0]
T = len(test)
N = len(test)
t_ar_oosample = np.linspace(0, T, N)

ar_returns_insample = compute_log_ret(ar_strat_insample)[1]*100
ar_returns_oosample = compute_log_ret(ar_strat_oosample)[1]*100

#%%
rc('xtick', labelsize=18) 
rc('ytick', labelsize=18) 

fig, axs = plt.subplots(1,2, figsize=(14,6))
axs[0].plot(np.arange(len(train)), ar_strat_insample, 
        label='autoregressive (train)',
        linewidth=1.5)
axs[0].plot(np.arange(len(train)), benchmark_timeseries_insample,
        label='benchmark (train)',
        linewidth=1.5,
        color='orange')
axs[0].set_xlabel('t', fontsize=18)
axs[0].set_ylabel('portfolio value', fontsize=18)
axs[0].legend(fontsize=16)
axs[0].grid(linewidth=0.3)
axs[1].plot(np.arange(len(test)), ar_strat_oosample, 
        label='autoregressive (test)',
        linewidth=1.5)
axs[1].plot(np.arange(len(test)), benchmark_timeseries_oosample,
        label='benchmark (test)',
        linewidth=1.5,
        color='orange')
axs[1].set_xlabel('t', fontsize=18)
axs[1].set_ylabel('portfolio value', fontsize=18)
axs[1].grid(linewidth=0.3)
axs[1].legend(fontsize=16)
fig.tight_layout()

#%% COMPARING 3 TRADING STRATS
fig, axs = plt.subplots(1,2, figsize=(12,5),
                       dpi=100)
axs[0].plot(np.arange(len(train)), tf_insample, 
        label='TF (train)',
        linewidth=1.5, color='white')
axs[0].plot(np.arange(len(train)), mr_insample, 
        label='MR (train)',
        linewidth=1.5)
axs[0].plot(t_ar_insample, ar_strat_insample,
           label='AR (train)',
           linewidth=1.5, color='orange')
axs[0].grid(linewidth=0.3)
axs[0].plot(np.arange(len(train)), benchmark_timeseries_insample,
        label='benchmark (train)',
        linewidth=2.5,
        color='red')
axs[0].legend(fontsize=16)
axs[0].set_xlabel('$t$',
                 fontsize=18)
axs[0].set_ylabel('portfolio value',
                 fontsize=18)
#---------------------------------------------------------------------------
axs[1].plot(np.arange(len(test)), tf_oosample, 
        label='TF (test)',
        linewidth=1.5, color='white')
axs[1].plot(np.arange(len(test)), mr_oosample, 
        label='MR (test)',
        linewidth=1.5)
axs[1].plot(t_ar_oosample, ar_strat_oosample,
           label='AR (test)',
           linewidth=1.5, color='orange')
axs[1].grid(linewidth=0.3)
axs[1].plot(np.arange(len(test)), benchmark_timeseries_oosample,
        label='benchmark (test)',
        linewidth=2.5,
        color='red')
axs[1].legend(fontsize=16,
              loc='lower left')
axs[1].set_xlabel('$t$',
                 fontsize=18)
axs[1].set_ylabel('portfolio value',
                 fontsize=18)
fig.tight_layout()

#%% PERFORMANCE MEASUREMENT
#Plotting the returns that the strategy gets over the cumulative time period
from scipy.stats import norm

ar_timeseries_insample = pd.Series(ar_strat_insample)
ar_timeseries_oosample = pd.Series(ar_strat_oosample)
returns_timeseries_ar = compute_log_ret(ar_strat_oosample)[0]
returns_timeseries_ar_insample = compute_log_ret(ar_strat_insample)[0]
average_return_daily = returns_timeseries_ar.mean()
plt.title('Returns on autoregressive strategy',
         fontsize=18)
plt.plot(returns_timeseries_ar, label='average = {}'.format(round(average_return_daily, 6)),
        linewidth=1.5)
plt.legend(fontsize=14, loc='lower left')
plt.grid(linewidth=0.3)

def compute_sharpe_annualised(returns_timeseries):
    mu_timeseries = returns_timeseries.mean()
    sigma_timeseries = returns_timeseries.std()
    sharpe_timeseries = mu_timeseries/sigma_timeseries
    return sharpe_timeseries * np.sqrt(252)

def compute_var(logret, confint=0.95):
    var = norm.ppf(1-confint, logret.mean(), logret.std())
    return var*100

def compute_var_historical(logret, confint=0.95):
    logret.sort_values(inplace=True, ascending=True)
    var = logret.quantile(1-confint)
    return var*100

def compute_ES(logret, confint=0.05):
    mu = logret.mean()*np.sqrt(252)
    std = logret.std()*np.sqrt(252)
    var = compute_var_historical(logret, confint=confint)
    ES = confint**-1 * norm.pdf(norm.ppf(confint))*std - mu
    return ES*100

def compute_strategy_ratios(insample_timeseries, oosample_timeseries,
                            confint=0.95):
    logret_insample = compute_log_ret(insample_timeseries)[0]
    logret_oosample = compute_log_ret(oosample_timeseries)[0]
    sharpe_insample = compute_sharpe_annualised(logret_insample)
    sharpe_oosample = compute_sharpe_annualised(logret_oosample)
    var_insample_historical = compute_var_historical(logret_insample,
                                                     confint=confint)
    var_oosample_historical = compute_var_historical(logret_oosample,
                                                     confint=confint)
    ES_insample = compute_ES(logret_insample, confint=1-confint)
    ES_oosample = compute_ES(logret_oosample, confint=1-confint)
    print('-------------------------------------')
    print('Annualised Sharpe ratios')
    print('In-sample: {}, Out-of-sample: {}'.format(round(sharpe_insample, 3),
                                                    round(sharpe_oosample, 3)))
    print('-------------------------------------')
    print('Value at risk at {}% confidence'.format(confint*100))
    print('In-sample: {}%, Out-of-sample: {}%'.format(round(var_insample_historical, 3),
                                                    round(var_oosample_historical, 3)))
    print('-------------------------------------')
    print('Expected Shortfall at {}% confidence'.format(confint*100))
    print('In-sample: {}%, Out-of-sample: {}%'.format(round(ES_insample, 3),
                                                    round(ES_oosample, 3)))

annual_sharpe_tf_insample = compute_sharpe_annualised(returns_trendfollowing_insample)
annual_sharpe_mr_insample = compute_sharpe_annualised(returns_mr_insample)
annual_sharpe_ar_insample = compute_sharpe_annualised(returns_timeseries_ar_insample)
print('SR tf: {}, SR mr: {}, SR ar: {}'.format(annual_sharpe_tf_insample,
                                              annual_sharpe_mr_insample,
                                              annual_sharpe_ar_insample))
print('------------------------------------------------------------------------------')
annual_sharpe_tf = compute_sharpe_annualised(returns_trendfollowing_oosample)
annual_sharpe_mr = compute_sharpe_annualised(returns_mr_oosample)
annual_sharpe_ar = compute_sharpe_annualised(returns_timeseries_ar)
print('SR tf: {}, SR mr: {}, SR ar: {}'.format(annual_sharpe_tf,
                                              annual_sharpe_mr,
                                              annual_sharpe_ar))
#%%
#ALL ratios computed
compute_strategy_ratios(benchmark_timeseries_insample, 
                        benchmark_timeseries_oosample)
compute_strategy_ratios(tf_insample, tf_oosample)
compute_strategy_ratios(mr_insample, mr_oosample)
compute_strategy_ratios(ar_timeseries_insample, ar_timeseries_oosample)
#%% ADJUSTING SHARPE RATIO
T = len(test)
df = T-1

#Original out-of-sample p-values
t_stat_tf = annual_sharpe_tf*np.sqrt(T/252)
p_tf_oosample = stats.t.sf(t_stat_tf, df)

t_stat_mr = annual_sharpe_mr*np.sqrt(T/252)
p_mr_oosample = stats.t.sf(t_stat_mr, df)

t_stat_ar = annual_sharpe_ar*np.sqrt(T/252)
p_ar_oosample = stats.t.sf(t_stat_ar, df)

p_tf_oosample
p_mr_oosample
p_ar_oosample

print(round(p_tf_oosample, 4), round(p_mr_oosample, 4), 
      round(p_ar_oosample, 4))
#%% BONFERRONI ADJUSTED SHARPE OUT-OF SAMPLE
#N_tf = len(returns_matrix[returns_matrix!=0.0])
N_tf = 3
p_oosample_adjusted_tf = p_tf_oosample*N_tf
#print(tf_insample_adjusted)
t_tf = stats.t.isf(p_oosample_adjusted_tf, df)
adjusted_tf_sharpe = t_tf/np.sqrt(T/252)

N_ar = 3
p_oosample_adjusted_ar = p_ar_oosample*N_ar
#print(tf_insample_adjusted)
t_ar = stats.t.isf(p_oosample_adjusted_ar, df)
adjusted_ar_sharpe = t_ar/np.sqrt(T/252)

print(round(adjusted_tf_sharpe, 4), 
      round(adjusted_ar_sharpe, 5))











