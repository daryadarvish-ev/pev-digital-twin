import random
import simpy
import numpy as np
import pandas as pd
from state_info import Station

import warnings

warnings.filterwarnings("ignore")


class Driver:
    def __init__(self, data_path, daily_sessions, total_day, delta_t, random_seeds=(150, 250, 350)):
        self.models = ['Prius Prime', 'Model 3', 'Bolt EV', 'Kona', 'Model X 100 Dual', 'Clarity',
                       'Volt (second generation)', 'B-Class Electric Drive', 'Model S 100 Dual',
                       'Mustang Mach-E', 'Model S 90 Dual', 'Bolt EUV 2LT']
        self.data_path = data_path
        self.session = daily_sessions
        self.total_day = total_day
        self.ses = [daily_sessions * 3] * total_day
        self.delta_t = delta_t
        self.car_type = []
        self.energyreq = []
        self.events_q = []
        # self.env = simpy.Environment()
        self.rnd_seeds = random_seeds

    class SessionGen:

        def read_origin_data(self):
            return pd.read_csv(self.data_path, parse_dates=['connectTime', 'startChargeTime', 'Deadline', 'lastUpdate'])

        def create_empty_df(self):
            return pd.DataFrame(columns=['arrivalDay', 'arrivalHour', 'arrivalMin', 'arrivalMinGlobal'])

        def arrival_gen(self, data):
            """
            Generate arrival time for each session.
            :param data: Dataframe from read_origin_data().
            """
            data['arrivalMin'] = data['connectTime'].apply(lambda x: x.hour * 60 + x.minute)
            data['arrivalHour'] = data['connectTime'].apply(lambda x: x.hour)
            df = self.create_empty_df()

            for i in range(len(self.ses)):
                np.random.seed(self.rnd_seeds[0] + i)
                quantiles = sorted(np.random.rand(self.ses[i]))
                aux_df = pd.DataFrame()
                aux_df['arrivalDay'] = [i] * self.ses[i]
                aux_df['arrivalHour'] = (np.quantile(data['arrivalMin'], quantiles)) / 60  # added arrival hours
                aux_df['arrivalMin'] = np.quantile(data['arrivalMin'], quantiles)
                aux_df['arrivalMinGlobal'] = aux_df['arrivalDay'] * 24 * 60 + aux_df['arrivalMin']
                df = pd.concat([df, aux_df])
            df.reset_index(inplace=True, drop=True)
            df['arrivalMin'] = df['arrivalMin'].apply(lambda x: int(x))
            df['arrivalHour'] = df['arrivalHour'].apply(lambda x: (x))
            df['arrivalMinGlobal'] = df['arrivalMinGlobal'].apply(lambda x: int(x))

        def generate_random_user(self, user, env):
            car = random.choices(self.models, weights=(151, 110, 86, 51, 42, 42, 28, 24, 20, 19, 15, 14))
            self.car_type.append(car)
            self.session.append(user)
            desired_energy = self.input_df['cumEnergy_kWh'][user - 1]
            self.energyreq.append(desired_energy)
            self.events_q.append((int(env.now), "%s : Car %s arrived" % (str(env.now), user)))
            inst_dept = int(env.now) + 1

            arrival_hour = self.input_df['arrivalHour']
            duration_hour = self.input_df['durationHour']
            e_need = self.input_df['cumEnergy_kWh']
            event = {
                "time": int(arrival_hour[user - 1] / self.delta_t),
                "e_need": e_need[user - 1],
                "duration": int(duration_hour[user - 1] / self.delta_t),
                "station_pow_max": 6.6,
                "user_power_rate": 6.6,
                "arrivalMinGlobal": self.input_df['arrivalMinGlobal'][user - 1],
                "departureMinGlobal": int(
                    self.input_df['arrivalMinGlobal'][user - 1] + self.input_df['durationHour'][user - 1] * 60)
            }
            return self.events_q

    # User information
    class user_info:

        def __init__(self, input_df, time, user, delta_t):
            self.input_df = input_df
            self.time = time
            self.user = user
            self.delta_t = delta_t

        def retrieve_user_information(self):
            arrival_hour = self.input_df['arrivalHour']
            duration_hour = self.input_df['durationHour']
            e_need = self.input_df['cumEnergy_kWh']
            event = {
                'arrival_hour': arrival_hour[self.user - 1],
                "time": int(arrival_hour[self.user - 1] / 0.25),  # INTERVAL
                "e_need": e_need[self.user - 1],  # FLOAT
                "duration": int(duration_hour[self.user - 1] / 0.25),  # the slider allows 15 min increments INT
                "station_pow_max": 6.6,
                "user_power_rate": 6.6,
                "arrivalMinGlobal": self.input_df['arrivalMinGlobal'][self.user - 1],
                "departureMinGlobal": int(
                    self.input_df['arrivalMinGlobal'][self.user - 1] + self.input_df['durationHour'][
                        self.user - 1] * 60)
            }
            return event

        def charging_evaluation(self, arrival_hour, duration_hour, desired_energy):
            """
            Evaluate the charging process for each session.
            :param arrival_hour: Arrival hour of the session.
            :param duration_hour: Duration of the session.
            :param actual_energy_delivered: Actual energy delivered to the session.
            :param desired_energy: Desired energy of the session.
            :return: Charging evaluation of the session.
            """
            # Calculate the actual energy delivered
            actual_energy_delivered = 0
            for i in range(int(duration_hour * 4)):
                actual_energy_delivered += self.delta_t * self.user_power_rate
            # Calculate the charging evaluation


    # Behavior model
    class UserBehavior:
        def __init__(self, model, asap_quantiles, flex_quantiles):
            self.model = model
            self.asap_quantiles = asap_quantiles
            self.flex_quantiles = flex_quantiles

        def random_leave(self, asap_price, flex_price):
            """Create a function to randomly assign the user to take a new option 'Leave'"""
            # if min(asap_price, flex_price) < min(self.asap_quantiles[0.25], self.flex_quantiles[0.25]):
            if min(asap_price, flex_price) < 21:
                leave_probability = 0.05
            # elif min(asap_price, flex_price) < min(self.asap_quantiles[0.50], self.flex_quantiles[0.50]):
            elif min(asap_price, flex_price) < 22:
                leave_probability = 0.075
            # elif min(asap_price, flex_price) < min(self.asap_quantiles[0.75], self.flex_quantiles[0.75]):
            elif min(asap_price, flex_price) < 24:
                leave_probability = 0.1
            else:
                leave_probability = 0.125

            leave = np.random.poisson(leave_probability)
            return 'Leave' if leave > 0 else None

        def basic_choice_function(self, asap_price, flex_price):
            """Basic choice function which chooses the lowest price"""

            if self.random_leave(asap_price, flex_price) == 'Leave':
                return "Leave", 9999
            # if random.uniform(0, 1) > 0.9:
            #     return "Leave", 9999
            elif asap_price > flex_price:
                return "Scheduled", flex_price
            else:
                return "Regular", asap_price

        def basic_choice_function_ml(self, asap_price, flex_price, model):
            """Basic choice function which chooses the lowest price with Machine Learning models"""

            if self.random_leave(asap_price, flex_price) == 'Leave':
                return "Leave", 9999
            # elif Station.data_analyze.predict_user_choice(model, asap_price, flex_price) == 'Leave':
            elif asap_price > flex_price:
                return "Scheduled", flex_price
            else:
                return "Regular", asap_price

    def append_choice(self, choice, new_user, current_user, user_choice, price, station, opt):
        FLEX_user = list()  # reset the user lists
        ASAP_user = list()
        LEAVE_user = list()

        print(new_user)
        print(current_user)

        for user in current_user:
            if (user_choice['EV' + str(user)] == "Regular"):
                ASAP_user.append('EV' + str(user))
            elif (user_choice['EV' + str(user)] == 'Scheduled'):
                FLEX_user.append('EV' + str(user))
            elif (user_choice['EV' + str(user)] == 'Leave'):
                LEAVE_user.append('EV' + str(user))
            else:
                None

        if (choice == "Regular"):
            ASAP_user.append('EV' + str(new_user))
            # station['EV' + str(new_user)].price = price
        elif (choice == 'Scheduled'):
            FLEX_user.append('EV' + str(new_user))
            # station['EV' + str(new_user)].price = price
        elif (choice == 'Leave'):
            LEAVE_user.append('EV' + str(new_user))
            # station['EV' + str(new_user)].price = price
        else:
            None

        station['FLEX_list'] = FLEX_user
        station['ASAP_list'] = ASAP_user
        station['LEAVE_list'] = LEAVE_user
        station['EV' + str(new_user)] = opt
        station['EV' + str(new_user)].price = price

        for ev in station['ASAP_list'].copy():
            if ev not in station:
                station['ASAP_list'].remove(ev)

        for ev in station['FLEX_list'].copy():
            if ev not in station:
                station['FLEX_list'].remove(ev)

        # Get a list of all EV numbers in the station dictionary
        station_ev_numbers = [int(key[2:]) for key in station.keys() if key.startswith("EV")]

        print(station_ev_numbers)
