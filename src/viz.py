# Text wrap for nicer graph output
import textwrap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                                    break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


def energyDelivered_choice(df):
    """
    Visualize the relations between delivered energy (column name: cumEnergy_Wh) and several other key metrics
    df: Pandas Dataframe
    """
    f_enerDlv_choice, ax = plt.subplots(figsize=(10, 5))
    f_enerDlv_choice = sns.histplot(df, x=df['cumEnergy_Wh'],
                                    hue=df['choice'],
                                    stat='probability',  # The y-axis is "Count" when stat is set as Default
                                    multiple='dodge',
                                    kde=True,
                                    bins=20)
    ax.set_title('Energy Delivered -- Choice', fontsize=18)
    ax.set_xlabel('Energy Delivered (Wh)', fontsize=15)
    ax.set_ylabel('Possibility', fontsize=15)
    plt.show()
    # return f_enerDlv_choice


def energyDelivered_DurationHrs(df):
    f_enerDlv_dur, ax = plt.subplots(figsize=(12, 5))
    f_enerDlv_dur = sns.histplot(df, x=df['DurationHrs'],
                                 y=df['cumEnergy_Wh'],
                                 hue=df['choice'],
                                 stat='probability',  # The y-axis is "Count" when stat is set as Default
                                 multiple='layer',
                                 kde=True,
                                 cbar=True,
                                 bins=100)
    ax.set_title('Energy Delivered -- Duration Hours', fontsize=18)
    ax.set_xlabel('Duration Hours (h)', fontsize=15)
    ax.set_ylabel('Energy Delivered (Wh)', fontsize=15)
    plt.show()
    # return f_enerDlv_dur


def energyDelivered_DurationHrs_joint(df):
    f_enerDlv_durMaxChar1 = sns.jointplot(data=df,
                                          x="DurationHrs",
                                          y="cumEnergy_Wh",
                                          kind="hex",
                                          color="#4CB391")
    plt.show()
    # return f_enerDlv_durMaxChar1


def Time_processing(df):
    """
    Process arrival and departure time data
    df: DataFrame
    """
    df['connectTime'] = pd.to_datetime(df['connectTime'], errors='coerce')
    # Change the minimum unit of resampling to change the accuracy
    # Resampling unit can also be set as "5Min""30Min""1H" etc.
    df1 = df.resample('10Min', on='connectTime').sum()
    df1['cumEnergy_Wh'] = df1.groupby(['connectTime'])['cumEnergy_Wh'].sum()
    df1 = df1.drop(df1[(df1['cumEnergy_Wh'] == 0)].index).reset_index(level=['connectTime'])

    # Arrival Time Distribution
    df1_arr_time = df1['connectTime'].dt.hour \
                   + df1['connectTime'].dt.minute / 60

    # Departure Time Distribution
    df1_dept_time = df1['connectTime'].dt.hour \
                    + df1['connectTime'].dt.minute / 60 \
                    + df1['DurationHrs']
    # If a user stays overnight, subtract 24/48 hours from the departure time
    df1_dept_time[(df1_dept_time >= 48)] = [df1_dept_time[(df1_dept_time >= 48)] - 48]
    df1_dept_time[(df1_dept_time >= 24)] = [df1_dept_time[(df1_dept_time >= 24)] - 24]

    return df1_arr_time, df1_dept_time


def energyDelivered_Arrvltime_joint(df, arrival_time):
    f_enerDlv_arrvl1 = sns.jointplot(data=df,
                                     x=arrival_time,
                                     y="cumEnergy_Wh",
                                     kind="hex",
                                     color="#4CB391")
    plt.show()
    # return f_enerDlv_arrvl1


def energyDelivered_Arrvltime(df, arrival_time):
    f_enerDlv_arrvl, ax = plt.subplots(figsize=(8, 5))
    f_enerDlv_arrvl = sns.histplot(df,
                                   x=arrival_time,
                                   y='cumEnergy_Wh',
                                   bins=70,
                                   cbar=True)
    ax.set_title('Energy Delivered -- Arrival Time (per 10 mins)', fontsize=18)
    ax.set_xlabel('Time in day', fontsize=14)
    ax.set_ylabel('Energy Delivered in total (Wh)', fontsize=14)
    plt.show()
    # return f_enerDlv_arrvl


def energyDelivered_Depttime_joint(df, departure_time):
    """
    This function is to draw a
    df: Pandas DataFrame
    departure_time: Pandas Series.
    """
    f_enerDlv_dept1 = sns.jointplot(data=df,
                                    x=departure_time,
                                    y='cumEnergy_Wh',
                                    kind="hex",
                                    color="#4CB391")
    plt.show()
    # return f_enerDlv_dept1


def energyDelivered_Depttime(df, departure_time):
    """
    This function is to draw a
    df: Pandas DataFrame
    departure_time:
    """
    f_enerDlv_dept, ax = plt.subplots(figsize=(8, 5))
    f_enerDlv_dept = sns.histplot(df,
                                  x=departure_time,
                                  y='cumEnergy_Wh',
                                  palette="ch:s=-.2,r=.6",
                                  bins=70,
                                  cbar=True)
    ax.set_title('Energy Delivered -- Departure Time (per 10 mins)', fontsize=18)
    ax.set_xlabel('Time in day', fontsize=14)
    ax.set_ylabel('Energy Delivered in total (Wh)', fontsize=14)
    plt.show()
    # return f_enerDlv_dept


def energyDelivered_DurMaxChar(df):
    """
    This function is to draw a
    df: DataFrame
    """
    # Delete one user's car with too large charging rate of 7,680 kW
    # The charging power of Tesla Model Y according Tesla database:
    # Charge Power = 11 kW AC    |    Fastcharge Power (max) = 250 kW DC
    df = df.drop(df[(df['vehicle_maxChgRate_W'] == 7680000)].index)
    f_enerDlv_durMaxChar, ax = plt.subplots(figsize=(10, 5))
    sns.despine(f_enerDlv_durMaxChar, left=True, bottom=True)
    f_enerDlv_durMaxChar = sns.scatterplot(data=df,
                                           x="DurationHrs",
                                           y="cumEnergy_Wh",
                                           hue="vehicle_maxChgRate_W",
                                           size="choice",
                                           palette='crest',
                                           sizes=(20, 6),
                                           linewidth=0,
                                           ax=ax)
    ax.set_title('Energy Delivered -- Duration & Max Charging Rate', fontsize=18)
    ax.set_xlabel('Duration Hours (h)', fontsize=14)
    ax.set_ylabel('Energy Delivered (Wh)', fontsize=14)
    plt.show()
    # return f_enerDlv_durMaxChar


# def energyDelivered_DurMaxChar(df):
#     """
#     This function is to draw a
#     df: Pandas DataFrame
#     """
#     f_stationId_Dem, ax = plt.subplots(figsize=(10, 5))
#     f_stationId_Dem = sns.violinplot(data=df,
#                                      x='stationId',
#                                      y='cumEnergy_Wh',
#                                      hue='choice',
#                                      split=True,
#                                      cut=0,
#                                      inner='quart',  # Set inner to 'stick' to see trend details
#                                      linewidth=1)
#     sns.despine(left=True)
#     ax.set_title('Station ID -- Power Demand', fontsize=18)
#     ax.set_xlabel('Station ID (UCSD: 3-10, UCB: 11-18)', fontsize=14)
#     ax.set_ylabel('Power Demand (Wh)', fontsize=14)
#     plt.show()
#     # return f_stationId_Dem


def stationID_PwrDem(df):
    f_stationId_Dem, ax = plt.subplots(figsize=(10, 5))
    f_stationId_Dem = sns.violinplot(data=df,
                                     x='stationId',
                                     y='cumEnergy_Wh',
                                     hue='regular',
                                     split=True,
                                     cut=0,
                                     inner='quart',  # Set inner to 'stick' to see trend details
                                     linewidth=1)
    sns.despine(left=True)
    ax.set_title('Station ID -- Power Demand', fontsize=18)
    ax.set_xlabel('Station ID (UCSD: 3-10, UCB: 11-18)', fontsize=14)
    ax.set_ylabel('Power Demand (Wh)', fontsize=14)
    plt.show()
    # return f_stationId_Dem


def stationID_PwrDem_Time(df):
    # df = pd.read_csv("./data/Sessions2_20221020.csv")
    # # Arrival Time Distribution
    # df['connectTime'] = pd.to_datetime(df['connectTime'],
    #                                     errors='coerce')

    prepare_df = locals()
    for i in range(3, 19):
        df_ = 'df_' + str(i)
        prepare_df['df_' + str(i)] = df[df['stationId'].isin([i])]
        prepare_df['df_' + str(i)] = prepare_df['df_' + str(i)].resample('10Min', on='connectTime').sum()
        # Change the minimum unit of resampling to change the accuracy
        prepare_df['df_' + str(i)]['cumEnergy_Wh'] = prepare_df['df_' + str(i)].groupby(['connectTime'])[
            'cumEnergy_Wh'].sum()
        prepare_df['df_' + str(i)] = prepare_df['df_' + str(i)].drop(
            prepare_df['df_' + str(i)][(prepare_df['df_' + str(i)]['cumEnergy_Wh'] == 0)].index).reset_index(
            level=['connectTime'])
        prepare_df['df_' + str(i)]['connectTime'] = prepare_df['df_' + str(i)]['connectTime'].dt.hour \
                                                    + prepare_df['df_' + str(i)]['connectTime'].dt.minute / 60

    y_max = prepare_df[df_]['cumEnergy_Wh'].max() + 2000

    f_stationId_Dem_Time, ax = plt.subplots(8, 2, figsize=(12, 18))
    for j in range(3, 11):
        ax[j - 3][0].scatter(data=prepare_df['df_' + str(j)], x='connectTime', y='cumEnergy_Wh',
                             label='Station ' + str(j))
        ax[j - 3][0].set_xlabel('Time (h)', fontsize=10)
        ax[j - 3][0].set_ylabel('Power Demand (Wh)', fontsize=10)
        ax[j - 3][0].set_xlim([0, 24])
        ax[j - 3][0].set_ylim([0, y_max])

    for j in range(11, 19):
        ax[j - 11][1].scatter(data=prepare_df['df_' + str(j)], x='connectTime', y='cumEnergy_Wh',
                              label='Station ' + str(j))
        ax[j - 11][1].set_xlabel('Time (h)', fontsize=10)
        ax[j - 11][1].set_ylabel('Power Demand (Wh)', fontsize=10)
        ax[j - 11][1].set_xlim([0, 24])
        ax[j - 11][1].set_ylim([0, y_max])

    plt.suptitle('Station-level Power Demand over Time', fontsize=15)
    plt.tight_layout()
    plt.show()
    # return f_stationId_Dem_Time


# df1 = pd.read_csv("../tests/data/Sessions2_20221020.csv")
# energyDelivered_choice(df1)
# energyDelivered_DurationHrs(df1)
# energyDelivered_DurationHrs_joint(df1)
# arr_Time, dept_Time = Time_processing(df1)
# energyDelivered_Arrvltime_joint(df1, arr_Time)
