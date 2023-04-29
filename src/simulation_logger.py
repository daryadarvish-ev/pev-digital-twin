import math
import datetime

from carbon_emissions import *


class SimulationLogger:
    def __init__(self):
        self.user_data = pd.DataFrame(columns=['arrival_time', 'e_need', 'ASAP_Power', 'FLEX_power', 'flex_tarrif', 'asap_tarrif',
                                               'choice', 'station_power_profile'])
        self.station_power = pd.DataFrame(columns=['time', 'station_power'])
        self.aggregate_metrics = pd.DataFrame(columns=['isotime', 'utility cost', 'carbon_emissions', 'revenue'])
        self.user_count = 0

    def add_data(self, time, res):
        self.user_count += 1
        station_power_profile = res["sch_agg"] if res['choice'] == 'Scheduled' else res['reg_agg']
        self.user_data.loc[self.user_count] = [time, res['e_need'], res['asap_powers'], res['flex_powers'], res['tariff_flex'], res['tariff_asap'], res['choice'], station_power_profile]

    def process_station_power(self):
        for index, row in self.user_data.iterrows():
            if row['choice'] != 'Leave':
                for timestep in range(len(row['station_power_profile'])):
                    self.station_power.loc[index] = [row['arrival_time'] + 15*timestep, row['station_power_profile'][timestep]]

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
            hour_of_day = (row['time']%1440)/60
            index_datetime = starttime_datetime + datetime.timedelta(minutes=row['time'])
            endtime_datetime = index_datetime + datetime.timedelta(minutes=15)

            # C02 Emissions (lbs) = power_kw * hr * 0.001mw/1kw *lbs/mwh
            carbon_emissions = emissions_df[(emissions_df['point_time'] >= index_datetime.isoformat()) &
                                            (emissions_df['point_time'] <= endtime_datetime.isoformat())]['value'].sum() * row['station_power'] * (15/60) * 0.001

            if 16 < hour_of_day < 21:
                self.aggregate_metrics.loc[index] = [index_datetime, TOU_OFF_PEAK * row['station_power'] * 15 / (60*100), carbon_emissions, 0]
            elif 9 < hour_of_day < 14:
                self.aggregate_metrics.loc[index] = [index_datetime, TOU_4PM_9PM * row['station_power'] * 15 / (60*100), carbon_emissions, 0]
            else:
                self.aggregate_metrics.loc[index] = [index_datetime, TOU_9AM_2PM * row['station_power'] * 15 / (60*100), carbon_emissions, 0]

        # get revenue per user
        for index, row in self.user_data.iterrows():
            choice = row['choice']
            if choice == 'Scheduled':
                self.aggregate_metrics.at[index, 'revenue'] = row['flex_tarrif']
            elif choice == 'Regular':
                self.aggregate_metrics.at[index, 'revenue'] = row['asap_tarrif']
