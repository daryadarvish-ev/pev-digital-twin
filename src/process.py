from datetime import datetime

import pandas as pd


class InvervalError(Exception):
    """Please specify interval as 'day', 'week', 'month', or 'year'"""
    pass


def compute_slack(input_df):
    """Computes Slack_Hr from the input dataframe

    Slack_Hr = (DurationHrs) - (EnergyConsumed_kWh / max_AC_power)
    Slack = 0hrs means there is no room to move charging demand
    As Slack increases, it means they have more “flexibility” in terms of time-shifting their power demand.

    Parameters
    ----------
    input_df : pandas dataframe
        input dataframe containing historical charging data

    Returns
    -------
    output_df : pandas dataframe
        input dataframe with additional column added for Slack_Hr
    """

    input_df['Slack_Hr'] = input_df['DurationHrs'] - (input_df['cumEnergy_Wh'] / input_df['vehicle_maxChgRate_W'])
    return input_df


def compute_revenue(input_df, interval='week'):
    """Computes revenue from the input dataframe

    Revenue can be computed per day, week, month, or year as specified, default is weekly.
    Uses connectTime as aggregation column.

    Parameters
    ----------
    input_df : pandas dataframe
        input dataframe containing historical charging data
    interval : str
        'day', 'week', 'month', 'year'
    
    Returns
    -------
    output_df : pandas dataframe
        pandas dataframe containing revenue per interval
    """
    input_df['connectTime'] = pd.to_datetime(input_df['connectTime'])
    if interval == 'day':
        revenue_df = input_df.groupby(pd.Grouper(key='connectTime', freq='D'))['estCost'].sum()
    elif interval == 'week':
        revenue_df = input_df.groupby(pd.Grouper(key='connectTime', freq='W'))['estCost'].sum()
    elif interval == 'month':
        revenue_df = input_df.groupby(pd.Grouper(key='connectTime', freq='M'))['estCost'].sum()
    elif interval == 'year':
        revenue_df = input_df.groupby(pd.Grouper(key='connectTime', freq='Y'))['estCost'].sum()
    else:
        raise InvervalError
    return revenue_df


def compute_utility_cost(input_df):
    """Compute utility cost and add a column to the dataframe for the utility cost per session

    Utility cost calculation: cumEnergy_Wh/avg(Time-of-use)

    Parameters
    ----------
    input_df : pandas dataframe
        input dataframe containing historical charging data

    Returns
    -------
    output_df : pandas dataframe
        pandas dataframe containing input data with column added for utility cost per session

    """

    # off-peak 0.175  cents / kwh
    TOU_OFF_PEAK = 17.5
    ## 4 pm - 9 pm peak 0.367 cents / kwh
    TOU_4PM_9PM = 36.7
    ## 9 am - 2 pm super off-peak 0.49 $ / kWh  to cents / kwh
    TOU_9AM_2PM = 14.9

    def get_utility_cost(row):
        unpacked_power_list = unpack_power(row['power'])

        total_utility_cost = 0

        for power_kw, timestamp in unpacked_power_list:
            if 16 < timestamp.hour < 21:
                total_utility_cost += TOU_OFF_PEAK * power_kw * 5 / 60  # multiply by # hours
            elif 9 < timestamp.hour < 14:
                total_utility_cost += TOU_4PM_9PM * power_kw * 5 / 60
            else:
                total_utility_cost += TOU_9AM_2PM * power_kw * 5 / 60
        return total_utility_cost

    input_df['utility_cost'] = input_df.apply(lambda row: get_utility_cost(row), axis=1)
    return input_df


def unpack_power(power_row):
    """Unpacks power column into list containing tuple of (kW, timestamp) for a charging session

    Parameters
    ----------
    power_row :
        row containing power data in string format

    Returns
    -------
    unpacked_power_list : list
        list of power in kW and timestamp

    """

    power_list = power_row.strip('][').split(', ')
    unpacked_power_list = []

    for i in range(0, len(power_list) // 2):
        try:
            power_kw = int(power_list[2 * i][21:25]) / 1000
        except:
            continue
        timestamp = datetime.fromtimestamp(int(power_list[2 * i + 1][22:32]))
        unpacked_power_list.append((power_kw, timestamp))
    return unpacked_power_list
