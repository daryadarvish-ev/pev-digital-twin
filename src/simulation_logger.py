import pandas as pd


class SimulationLogger:
    def __init__(self):
        self.user_data = pd.DataFrame(columns=['e_need', 'ASAP_Power', 'FLEX_power', 'flex_tarrif', 'asap_tarrif', 'choice'])
        self.station_data = pd.DataFrame(columns=['station_power'])
        self.aggregate_metrics = pd.DataFrame(columns=['revenue'])

    def add_data(self, time, res):
        self.user_data.loc[time] = [res['e_need'], res['asap_powers'], res['flex_powers'], res['tariff_flex'], res['tariff_asap'], res['choice']]
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