# Class of site state information

import random
import simpy
import pandas as pd
import numpy as np

class site_state:
    """
    Stores charging station information, including pricing menu, onsite ev status, current TOU, etc.


    Parameters:
        environment      : 'object'. A 'simpy.core.Environment' object.
        station_name     : 'str'. Charging site name. (For now we only use "UC Berkeley RSF Lot".
                                                       Can make extensions in the future.)
        station_location : 'str'. Station location. (Set default as "RSF Parking Garage, Berkeley, CA 94704".
                                                     Should include street address in future versions.)
        capacity         : 'int'. Station capacity. Total number of chargers .
        onsite_ev_number : 'int'. Number of EVs currently onsite.
        pricing_menu     : 'dict'. Menu of current price.
        current_TOU      : 'float'. Current time of use.


    """

    # CODE NOTES #######################

    def __init__(self, environment, station_name, station_location, capacity, onsite_ev_number, pricing_menu, current_TOU):
        self.environment = environment
        self.station_name = station_name
        self.station_location = station_location
        self.capacity = capacity
        self.pricing_menu = pricing_menu
        self.onsite_ev_number = onsite_ev_number
        self.current_TOU = current_TOU

    def curr_price_menu(self):
        """
        This function is
        """
        return self.pricing_menu

    def occupancy_rate_calculate(self):
        return 100*(self.onsite_ev_number/self.capacity)


    def display_general_info(self):
        print(f"Charging Station: {self.name}")
        print(f"Location: {self.location}")
        print(f"Capacity: {self.capacity}")

    def display_realtime_info(self):
        print(f"Occupancy rate: {self.occupancy_rate_calculate()} %")
        print("Pricing Menu:")
        for item, price in self.pricing_menu.items():
            print(f"{item}: {price}")
        print(f"Onsite EV Numbers: {self.onsite_ev_status}")
        print(f"Onsite EV Status: ")
        print(f"Current TOU: {self.current_tou}")


# Example
env = simpy.Environment()
name = "UC Berkeley RSF Lot"
location = "RSF Parking Garage, Berkeley, CA 94704"

reg_price = 1.5
sch_price = 2.0
pricing = {"REG": f"$ {reg_price} /hour",
           "SCH": f"$ {sch_price} /hour"}
onsite_ev = 5
TOU_tariff = np.ones((96,)) * 17.5
TOU_tariff[36:56] = 14.9
TOU_tariff[64:84] = 36.7
TOU = TOU_tariff


cs = site_state(environment=env,
                station_name=name,
                station_location=location,
                capacity=8,
                pricing_menu=pricing,
                onsite_ev_number=onsite_ev,
                current_TOU=TOU)

# import simpy
# import random
#
#
# class EV:
#     def __init__(self, env, inputgen, site_state):
#         self.env = env
#         self.inputgen = inputgen
#         self.site_state = site_state
#         self.arrival_time = env.now
#         self.departure_time = None
#         self.charging_choice = inputgen.choose_charging()
#         self.energy_requested = inputgen.request_energy()
#
#     def charge(self):
#         charging_time = self.energy_requested / self.site_state.get_charging_rate(self.charging_choice)
#         yield self.env.timeout(charging_time)
#
#     def run(self):
#         with self.site_state.charge_queue.request() as req:
#             yield req
#             self.site_state.onsite_evs += 1
#             yield self.env.process(self.charge())
#             self.site_state.onsite_evs -= 1
#             self.departure_time = self.env.now
#
#
# class InputGen:
#     def __init__(self, env, max_time, regular_prob, energy_req_mean, energy_req_std):
#         self.env = env
#         self.max_time = max_time
#         self.regular_prob = regular_prob
#         self.energy_req_mean = energy_req_mean
#         self.energy_req_std = energy_req_std
#
#     def choose_charging(self):
#         if random.random() < self.regular_prob:
#             return 'regular'
#         else:
#             return 'scheduled'
#
#     def request_energy(self):
#         return max(0, random.normalvariate(self.energy_req_mean, self.energy_req_std))
#
#     def run(self):
#         while self.env.now < self.max_time:
#             yield self.env.timeout(random.expovariate(1.0))
#             ev = EV(self.env, self, self.site_state)
#             self.env.process(ev.run())
#
#
# class site_state:
#     def __init__(self, env, charging_rate_regular, charging_rate_scheduled, max_onsite_evs):
#         self.env = env
#         self.charging_rate_regular = charging_rate_regular
#         self.charging_rate_scheduled = charging_rate_scheduled
#         self.max_onsite_evs = max_onsite_evs
#         self.onsite_evs = 0
#         self.charge_queue = simpy.Resource(env, capacity=max_onsite_evs)
#
#     def get_charging_rate(self, charging_choice):
#         if charging_choice == 'regular':
#             return self.charging_rate_regular
#         else:
#             return self.charging_rate_scheduled
#
#
# def run_simulation(max_time, regular_prob, energy_req_mean, energy_req_std, charging_rate_regular,
#                    charging_rate_scheduled, max_onsite_evs):
#     env = simpy.Environment()
#     site = site_state(env, charging_rate_regular, charging_rate_scheduled, max_onsite_evs)
#     inputgen = InputGen(env, max_time, regular_prob, energy_req_mean, energy_req_std)
#     env.process(inputgen.run())
#     env.run(until=max_time)
#
#     return site
#
#
# # Example usage:
# site = run_simulation(max_time=100, regular_prob=0.8, energy_req_mean=10, energy_req_std=3, charging_rate_regular=1,
#                       charging_rate_scheduled=2, max_onsite_evs=5)
#
# print(f"Number of onsite EVs: {site.onsite_evs}")
# print(f"Number of EVs in queue: {len(site.charge_queue.queue)}")
