
# Class of Users(EV Drivers)

import random
import numpy as np
import pandas as pd
import site_state

Choice = []

class EV:
    """
    Store individual EV driver information.

    Parameters:
        environment     : 'object'. A 'simpy.core.Environment' object.
        ev_id           : 'int'.
        pricing_menu    :
        arrival_time    :
        departure_time  :
        energy_demand   :

    """

    # CODE NOTES #######################

    # basic_session_info: t_arrival, t_depart, P_max, B_max, e_init, e_req
    # session_info_after_decision: stated_td, stated_e_req, actual_td

    # Maybe need a new parameter to indicate whether this EV is entering or is leaving.

    def __init__(self, environment, ev_id, pricing_menu, arrival_time, departure_time, energy_demand, charging_rate):
        self.environment = environment
        self.ev_id = ev_id
        self.pricing_menu = pricing_menu
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.energy_demand = energy_demand
        self.charging_rate = charging_rate
        print(f"EV {ev_id} arrives at {self.arrival_time}. Need {self.energy_demand} kWh")

    def basic_session_info(self):
        """
        Store EV user information in a dictionary.
        ####################################TO BE FINISHED####################################################
        """
        dic1 = {"ed_id": self.ev_id,
                "Arrival Time": self.arrival_time}
        pass

    def state_of_charge(self):
        """
        State of charge (SOC). Measurement of the charging process of a user at a specific point
        in time expressed as a PERCENTAGE.
        """
        # charged time = current time from self.environment - self.arrival time
        # charging_rate = 3.3 kW, or 6.6 kW
        # Formula:
        # [(charged time * charging_rate) / (energy_demand)] * 100%
        charged_time = self.environment.now() - self.arrival_time
        percentage = (charged_time * self.charging_rate / self.energy_demand) * 100
        return percentage

    def calculate_probability(self):
        """
        This function is to calculate the probability of users choice.
        """
        reg = self.pricing_menu.get("REG")
        sch = self.pricing_menu.get("SCH")

        u_reg = 0.3411 - 0.0184 * (sch - reg) * .5
        u_sch = 0.0184 * (sch - reg) * .5
        u_leave = -1. + 0.005 * (np.mean([sch, reg]))

        #     print(u_leave, u_sch, u_reg)
        denom = np.exp(u_reg) + np.exp(u_sch) + np.exp(u_leave)

        p_reg = np.exp(u_reg) / denom
        p_sch = np.exp(u_sch) / denom
        p_leave = np.exp(u_leave) / denom

        #     print(p_reg, p_sch, p_leave, np.sum([p_reg, p_sch, p_leave]))
        return p_reg, p_sch, p_leave

    def choice_function_rdn(self, ratio):
        """
        This function is the choice function with a random
        """

        # Users intend to choose lower price,
        # but their choices will be disturbed by random items,
        # and they may not always choose a lower price under "perfect rationality".
        reg = self.pricing_menu.get("REG")
        sch = self.pricing_menu.get("SCH")

        if random.uniform(0, 1) > 0.9 or site_state.occupancy_rate:
            # choice = 3
            Choice.append("Leave")
        if abs(reg - sch) <= ratio * reg:
            # choice = 1
            Choice.append(random.choice(["Scheduled", "Regular"]))
        else:
            # choice = 2
            Choice.append("Regular")
        return Choice

    def choice_function_dtm(self):
        """
        Users strictly select charging mode with lower price.
        """
        reg = self.pricing_menu.get("REG")
        sch = self.pricing_menu.get("SCH")

        if random.uniform(0, 1) > 0.9:
            choice = 3
            Choice.append("Leave")
        if reg > sch:
            choice = 1
            Choice.append("Scheduled")
        else:
            choice = 2
            Choice.append("Regular")
        return Choice

    def evaluate_session(self):
        """

        """

        pass

    def return_info(self):
        """
        Return individual EV user information.
        """

        pass


