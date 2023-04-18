import pandas as pd
import numpy as np

class user_info:

    def __init__(self, input_df, time ,user, delta_t):
        self.input_df = input_df
        self.time = time
        self.user = user
        self.delta_t = delta_t

    def retrieve_user_information(self):

        arrival_hour = self.input_df['arrivalHour']
        duration_hour = self.input_df['durationHour']
        e_need = self.input_df['cumEnergy_kWh']
        event = {
            'arrival_hour' : arrival_hour[self.user-1],
            "time": int(arrival_hour[self.user - 1] / 0.25),  # INTERVAL
            "e_need": e_need[self.user - 1],  # FLOAT
            "duration": int(duration_hour[self.user - 1] / 0.25),  # the slider allows 15 min increments INT
            "station_pow_max": 6.6,
            "user_power_rate": 6.6,
            "arrivalMinGlobal": self.input_df['arrivalMinGlobal'][self.user - 1],
            "departureMinGlobal": int(self.input_df['arrivalMinGlobal'][self.user - 1] + self.input_df['durationHour'][self.user - 1] * 60)
        }
        return event




