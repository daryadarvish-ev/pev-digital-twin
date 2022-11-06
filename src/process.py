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


def compute_utility_cost(input_df, interval='week'):
    """Compute historical utility cost
    """
