import random
import simpy

class driver:
    def __init__(self, input_df, delta_t):
        self.models = ['Prius Prime','Model 3','Bolt EV', 'Kona','Model X 100 Dual','Clarity','Volt (second generation)','B-Class Electric Drive','Model S 100 Dual','Mustang Mach-E','Model S 90 Dual','Bolt EUV 2LT']
        self.input_df = input_df
        self.delta_t = delta_t
        self.car_type = []
        self.session = []
        self.energyreq = []
        self.events_q = []
        # self.env = simpy.Environment()

    def generate_random_user(self, user):
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
            "arrivalMinGlobal": self.input_df['arrivalMinGlobal'][user-1],
            "departureMinGlobal": int(self.input_df['arrivalMinGlobal'][user-1] + self.input_df['durationHour'][user-1] * 60)
        }
        return self.events_q