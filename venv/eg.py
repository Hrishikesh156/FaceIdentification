import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib
inline

XGB_df = pd.read_csv("train.csv")
XGB_df.head()

XGB_df = XGB_df[(XGB_df.store == 1) & (XGB_df.item == 1)]

XGB_df.shape

from datetime import datetime

XGB_df['date'] = pd.to_datetime(XGB_df['date'])

# Feature Engineering
XGB_df.head()

Split_df1 = "2016-12-31"
Split_df2 = "2017-01-01"

XGB_df['Day'] = XGB_df['date'].dt.dayofyear
XGB_df['Weekday'] = XGB_df['date'].dt.weekday
XGB_df['Month'] = XGB_df['date'].dt.month
XGB_df['Dayofmonth'] = XGB_df['date'].dt.day
XGB_df['Quarter'] = XGB_df['date'].dt.quarter
XGB_df['Year'] = XGB_df['date'].dt.year
XGB_df['WeekOfYear'] = XGB_df['date'].dt.weekofyear
XGB_df['Item_Store_Combined'] = XGB_df['item'].map(str) + '-' + XGB_df['store'].map(str)

jour = XGB_df['date'].dt.day_name()
XGB_df['jour'] = XGB_df['date'].dt.day_name()

XGB_df = XGB_df.set_index('date')
XGB_df.head()

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


def error_calculation(y_true, y_pred):
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'Mean Absolute Percentage Error': MAPE, 'Root Mean Square Error': RMSE,
            'Mean Squared Error': MSE, 'Mean Absolute Error': MAE}


# Splitting and make Test Train and Main Tracking Option
XGB_df_train = XGB_df.loc[:Split_df1]
XGB_df_train.head()

XGB_df_test = XGB_df.loc[Split_df2:]
XGB_df_test.head()

XGB_df_test_final = XGB_df_test.copy()

XGB_df_test_final = XGB_df_test_final.drop(['Day', 'Weekday', 'Month', 'Quarter', 'Year', 'WeekOfYear'], axis=1)

X_train_XGB = XGB_df_train.drop(['sales'], axis=1)
y_train_XGB = XGB_df_train.loc[:, 'sales']
X_test_XGB = XGB_df_test.drop(['sales'], axis=1)
y_test_XGB = XGB_df_test.loc[:, 'sales']

# XG Boost
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from xgboost import XGBClassifier

XGB_model = xgb.XGBRegressor(n_estimators=1000)

X_test_XGB = X_test_XGB.drop(['Item_Store_Combined', 'jour'], axis=1)
X_train_XGB = X_train_XGB.drop(['Item_Store_Combined', 'jour'], axis=1)  # ,'jour'

X_test_XGB.head()

XGB_model.fit(X_train_XGB, y_train_XGB, eval_set=[(X_test_XGB, y_test_XGB)], early_stopping_rounds=50, verbose=False)

plot_importance(XGB_model, height=0.8)

XGB_test_prediction = XGB_model.predict(X_test_XGB).round()

XGB_test_all = X_test_XGB.copy()
XGB_train_all = X_train_XGB.copy()
XGB_test_all['XGB predicted sales'] = XGB_model.predict(X_test_XGB).round()
XGB_train_all['XGB predicted sales'] = XGB_model.predict(X_train_XGB).round()
XGB_test_all['sales'] = y_test_XGB
XGB_train_all['sales'] = y_train_XGB
df_XGB_all = pd.concat([XGB_train_all, XGB_test_all], sort=False)

# Results
XGB_result = error_calculation(XGB_test_all['sales'], XGB_test_all['XGB predicted sales'])
XGB_result = pd.DataFrame(XGB_result, index=[0])
XGB_result

df_XGB_all['error_forecast'] = df_XGB_all['XGB predicted sales'] - df_XGB_all['sales']

df_XGB_all['jour'] = XGB_df['jour']


def replne(days):
    if days == 'Monday' or days == 'Wednesday':
        return 'True'
    elif days == 'Friday' or days == 'Sunday':
        return 'True'
    elif days == 'Tuesday' or days == 'Thursday':
        return 'False'
    elif days == 'Saturday':
        return 'False'


# create the column of the class
df_XGB_all['repln'] = df_XGB_all['jour'].apply(replne)

# Demand Planning: XGBoost vs. Rolling Mean

# Demand Planning using Rolling Mean

# Capacity per item
capacity_n = 10  # Capacity = n x Sales Max
rolling_ndays = 8
n_days = rolling_ndays
n_day = rolling_ndays
XGB_df = XGB_df.reset_index()
list_dates = XGB_df['date'].unique()

df_sales = df_XGB_all['sales']
df_fcst = df_XGB_all['XGB predicted sales']

# Sales rolling mean
df_roll = df_sales.rolling(n_days).mean().round(0)
df_roll.dropna()

# XGBoost vs. Rolling Mean
# Sum of the forecast sales of the next n days
df_ft3n = df_fcst.rolling(window=n_window).sum().shift(-2)
df_ft3n = df_ft3n.iloc[rolling_ndays - 1:]


# Sum of the forecast sales of the next n days
def forecastxgb_sum_n(df_fcst, n_window, rolling_ndays):
    return df_ft3n


# p * day n sales applied for the sales of the next n days
def forecastrm_sum_n(df_roll, n_window):
    return df_roll * n_window


# Calculate Sum of Actual sales: Day n, Day n+1, Day n+2, Day n + p
df_act_p = df_sales.rolling(window=days_p).sum().shift(-(days_p - 1))

# Error Forecast Calculation

# Sum of the forecast sales of the next n days
df_ft_n = forecastxgb_sum_n(df_fcst, frcst_n_days, rolling_ndays)

# Sum of the rolling mean sales of the next n days
df_rm_n = forecastrm_sum_n(df_roll, frcst_n_days)

# XGBoost vs. Rolling Mean: p = 8 days

# 1- Initiate Inventory records

# Create first column with max capacity
df_inv = pd.DataFrame((df_XGB_all.groupby(['store', 'item'])['sales'].max() // 50 + capacity_n - 1) * 50)
df_inv.columns = ['capacity']
# df_inv = df_inv.reset_index()

# Inventory Day 1
df_inv[list_dates[n_day_start - 1]] = df_inv['capacity']
# df_inv = df_inv.set_index(['store', 'item']).T
df_inv.head()

# 2- Create DataFrame of storage capacity per item

# Capacity DataFrame = n times maximum sales
df_cap = pd.DataFrame(df_XGB_all.groupby(['item'])['sales'].max())
df_cap.columns = ['sales_max']
df_cap['capacity'] = (df_cap['sales_max'] // 50 + capacity_n - 1) * 50

# Dictionnary
dict_cap = dict(zip(df_cap.index.values, df_cap.capacity.values))
df_cap = df_cap.reset_index().set_index(['item', 'sales_max'])

df_XGB_all.reset_index()

# Replenishment days
days_reappro = ['Monday', 'Wednesday', 'Friday', 'Sunday']
df_XGB_all['replen'] = df_XGB_all.jour.map(lambda t: t in days_reappro)

# Dictionnary with date:replenishment_bool
# x=XGB_df['date'].dt.date.values
# y=df_XGB_all.replen.values
# dict_rep = dict(zip(x, y))

dict_rep = dict(zip(XGB_df['date'].dt.date.values, df_XGB_all.replen.values))

# Create DataFrames to track replenishment
# Inventory Records per day at each step: Sales Reduction, Replenishmen
df_invday = df_inv.copy()
# Inventory is Positive: After Sales Reduction >= 0 ?
df_invpos = df_inv.copy() * 0
# Replenishment record per day: Replenishment_Qty = Capacity – Inventory Level
df_repm1 = df_inv.copy()


# Method 1
# Replenishment = Max Capacity – Inventory after Sales Reduction
def rep_m1(df_repm1, n_day, df_invday, dict_rep, list_dates):
    # Calculate Replenishment Qty
    if dict_rep[list_dates[n_day - 1]] == True:
        df_repm1.loc[list_dates[n_day - 1]] = (df_invday.loc['capacity']) - df_invday.loc[list_dates[n_day - 1]]
    else:
        df_repm1.loc[list_dates[n_day - 1]] = 0

    # Apply Replenishment Qty to Inventory
    df_invday.loc[list_dates[n_day - 1]] = df_invday.loc[list_dates[n_day - 1]] + df_repm1.loc[list_dates[n_day - 1]]
    return df_repm1, df_invday


def sales_reduction(df_invday, df_sales, n_day, list_dates):
    return df_invday


# Simulation Replenishment Method 1
def simul_rem1(rolling_ndays, days, list_dates, dict_rep,
               df_invday, df_sales, df_repm1, df_invpos):
    # First for n_day = 1
    n_day = rolling_ndays
    # Sales Substraction: Inventory = Inventory – Sales
    df_invday.loc[list_dates[n_day - 1]] = df_invday.loc[list_dates[n_day - 1]] - df_sales.loc[list_dates[n_day - 1]]
    # Replenishment Method 1: Replenish to Capacity
    df_repm1.loc[list_dates[n_day - 1]] = df_invday.loc['capacity'] - df_invday.loc[list_dates[n_day - 1]]
    df_invday.loc[list_dates[n_day - 1]] = df_invday.loc[list_dates[n_day - 1]] + df_repm1.loc[list_dates[n_day - 1]]
    # Is Inventory Positive after Sales Reduction ? [1: True, 0: False]
    df_invpos.loc[list_dates[n_day - 1]] = (df_invday.loc[list_dates[n_day - 1]] < 0) * 1

    for n_day in range(rolling_ndays + 1, days + rolling_ndays):  # len(list_dates) +1

        # Sales Reduction: Inventory = Inventory – Sales
        df_invday = sales_reduction(df_invday, df_sales, n_day, list_dates)

        # Replenishment Method 1: Replenish to Capacity
        df_repm1, df_invday = rep_m1(df_repm1, n_day, df_invday, dict_rep, list_dates)

        # Is Inventory Positive after Sales Reduction ? [1: True, 0: False]
        df_invpos = inv_pos(df_invpos, df_invday, n_day, list_dates)

    # Analysis
    # Caculate Inventory level = % Total Capacity
    df_invday['total_inv'] = df_invday.sum(axis=1)
    df_invday['%total_inv'] = (100 * df_invday.iloc[1:, -1].div(df_invday.iloc[0, -1])).round(2)

    return df_invday, df_repm1


simul_rem1(rolling_ndays, days, list_dates, dict_rep,
           df_invday, df_sales, df_repm1, df_invpos)

