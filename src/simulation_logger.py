import math
import datetime

import pandas as pd
from carbon_emissions import *


class SimulationLogger:
    def __init__(self):
        self.user_data = pd.DataFrame(columns=['e_need', 'ASAP_Power', 'FLEX_power', 'flex_tarrif', 'asap_tarrif',
                                               'choice', 'station_power_profile'])
        self.station_power = pd.DataFrame(columns=['station_power'])
        self.aggregate_metrics = pd.DataFrame(columns=['isotime', 'utility cost','carbon_emissions', 'revenue'])

    def add_data(self, time, res):
        station_power_profile = res["sch_agg"] if res['choice'] == 'Scheduled' else res['reg_agg']
        self.user_data.loc[time] = [res['e_need'], res['asap_powers'], res['flex_powers'], res['tariff_flex'], res['tariff_asap'], res['choice'], station_power_profile]
        return self.user_data

    def get_eneed(self, time):
        e_need = self.user_data.loc[time, 'e_need']
        return e_need

    def get_asap_power(self, time):
        asap_power = self.user_data.loc[time, 'ASAP_Power']
        return asap_power

    def get_flex_power(self, time):
        flex_power = self.user_data.loc[time, 'FLEX_power']
        return flex_power

    def get_tariff_flex(self, time):
        tariff_flex = self.user_data.loc[time, 'flex_tarrif']
        return tariff_flex

    def get_tariff_asap(self, time):
        tariff_asap = self.user_data.loc[time, 'asap_tarrif']
        return tariff_asap

    def process_station_power(self):
        for index, row in self.user_data.iterrows():
            if row['choice'] != 'Leave':
                for timestep in range(len(row['station_power_profile'])):
                    self.station_power.loc[int(index) + 15*timestep] = row['station_power_profile'][timestep]

    def compute_aggregate_metrics(self):
        """Computes station carbon emissions, utility cost, and revenue.
        Make sure to call get_station_power_before this"""
        # off-peak 0.175  cents / kwh
        TOU_OFF_PEAK = 17.5
        # 4 pm - 9 pm peak 0.367 cents / kwh
        TOU_4PM_9PM = 36.7
        # 9 am - 2 pm super off-peak 0.49 $ / kWh  to cents / kwh
        TOU_9AM_2PM = 14.9

        # get utility cost and carbon emissions based on station power
        token = login()

        # latitude and longitude for UC Berkeley
        latitude = '37.87'
        longitude = '-122.25'
        determine_grid_region(token, latitude, longitude)

        # Assume simulation starts simulation length days ago

        today = datetime.datetime.now()
        start_of_today = datetime.datetime(today.year, today.month, today.day)
        sim_length_days = math.ceil(self.station_power.index.max()/1440)
        starttime_datetime = start_of_today - datetime.timedelta(days=sim_length_days)
        starttime = starttime_datetime.isoformat()
        endtime = start_of_today.isoformat()
        emissions_df = get_emissions(token, starttime, endtime)

        for index, row in self.station_power.iterrows():
            hour_of_day = (int(index)%1440)/60
            index_datetime = starttime_datetime + datetime.timedelta(minutes=int(index))
            endtime_datetime = index_datetime + datetime.timedelta(minutes=15)

            # C02 Emissions (lbs) = power_kw * hr * 0.001mw/1kw *lbs/mwh
            carbon_emissions = emissions_df[(emissions_df['point_time'] >= index_datetime.isoformat()) &
                                            (emissions_df['point_time'] <= endtime_datetime.isoformat())]['value'].sum() * row['station_power'] * (15/60) * 0.001

            if 16 < hour_of_day < 21:
                self.aggregate_metrics.loc[int(index)] = [index_datetime, TOU_OFF_PEAK * row['station_power'] * 15 / (60*100), carbon_emissions, 0]
            elif 9 < hour_of_day < 14:
                self.aggregate_metrics.loc[int(index)] = [index_datetime, TOU_4PM_9PM * row['station_power'] * 15 / (60*100), carbon_emissions, 0]
            else:
                self.aggregate_metrics.loc[int(index)] = [index_datetime, TOU_9AM_2PM * row['station_power'] * 15 / (60*100), carbon_emissions, 0]

        # get revenue per user
        for index, row in self.user_data.iterrows():
            choice = row['choice']
            if choice == 'Scheduled':
                self.aggregate_metrics.at[int(index), 'revenue'] = row['flex_tarrif']
            elif choice == 'Regular':
                self.aggregate_metrics.at[int(index), 'revenue'] = row['asap_tarrif']
            else:
                self.aggregate_metrics.at[int(index), 'revenue'] = 0
