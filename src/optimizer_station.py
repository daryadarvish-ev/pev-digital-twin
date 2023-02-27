import math
import timeit
import warnings
import cvxpy as cp
import numpy as np
import copy

warnings.filterwarnings("ignore")

class Parameters:
    """
    Class to hold all parameters for all simulations which will be the same for each problem.
    """
    def __init__(self, 
                z0 = np.array([1,1,1,1]).reshape(4,1),
                v0 = np.array([0.3333, 0.3333, 0.3333]).reshape(3,1) ,  
                Ts = 0.25, # 1 control horizon interval = 0.25 global hour
                base_tarriff_overstay = 1.0, 
                eff = 1,  # Charging efficient, assumed to be 100%
                soft_v_eta = 1e-4, # For convex relaxation in constraints.
                opt_eps = 0.0001, 
                TOU = np.ones((96,))): # Time-of-user charging tariff: 96 intervals, 1 day.

        # TOU Tariff in cents / kwh
        # TOU * power rate = cents  / hour

        self.v0 = v0
        self.z0 = z0
        self.Ts = Ts
        self.base_tariff_overstay = base_tarriff_overstay
        self.TOU = TOU
        self.eff = eff # power efficiency
        self.dcm_choices = ['charging with flexibility', 'charging asap', 'leaving without charging']
        self.soft_v_eta = soft_v_eta #softening equality constraint for v; to avoid numerical error
        self.opt_eps = opt_eps
        self.cost_dc = 0  # Cost for demand charge. This value is arbitrary now. A larger value means the charging profile will go average.
        # 18.8 --> 300 We can change this value to show the effect of station-level impact.

        assert len(self.TOU) == int(24 / self.Ts), "Mismatch between TOU cost array size and discretization steps"

class Problem:
    """
    This class encompasses the current user information which will change for every user optimization.

    time, int, user interval
    duration, int, number of charging intervals
    """
    def __init__(self, par, **kwargs):
        self.par = par
        event = kwargs["event"]
        self.event = event
        
        self.user_time = event["time"]
        self.e_need = event["e_need"]

        self.user_duration = event["duration"]
        # self.user_overstay_duration = round(event["overstay_duration"] / par.Ts) * par.Ts
        # Power cap for the station charger
        self.station_pow_max = event["station_pow_max"]
        self.user_power_rate = event['user_power_rate']

        self.power_rate = min(self.user_power_rate, self.station_pow_max) # The actual power cap for user

        self.dcm_charging_sch_params = np.array([[ - self.power_rate * 0.0184 / 2], [self.power_rate * 0.0184 / 2], [0], [0]])
        #% DCM parameters for choice 1 -- charging with flexibility
        self.dcm_charging_reg_params = np.array([[self.power_rate * 0.0184 / 2], [- self.power_rate * 0.0184 / 2], [0], [0.341]])
        #% DCM parameters for choice 2 -- charging as soon as possible
        self.dcm_leaving_params = np.array([[self.power_rate * 0.005 / 2], [self.power_rate * 0.005 / 2], [0], [-1]])
        
        #% DCM parameters for choice 3 -- leaving without charging
        self.THETA = np.vstack((self.dcm_charging_sch_params.T, self.dcm_charging_reg_params.T, self.dcm_leaving_params.T))

        # problem specifications
        self.N_sch = self.user_duration
        self.N_reg = math.ceil((self.e_need / self.power_rate / par.eff * int(1 / par.Ts)))
        self.N_reg_remainder = (self.e_need / self.power_rate / par.eff * int(1 / par.Ts)) % 1

        self.assertion_flag = 0
        ## Option 1: Update the e_need to avoid opt failure
        if self.N_sch < self.N_reg:
            # self.e_need = self.N_sch * self.power_rate * par.eff * par.Ts
            # self.assertion_flag = 1
        ## Option 2: Update the N_sch to avoid opt failure
            self.N_sch = self.N_reg
            self.user_duration = self.N_sch
            self.assertion_flag = 1

        if len(par.TOU) < self.user_time + self.user_duration: # if there is overnight chaarging
            par.TOU = np.concatenate([par.TOU,par.TOU])
        self.TOU = par.TOU[self.user_time:(self.user_time + self.user_duration)]

        self.limit_reg_with_sch = event["limit_reg_with_sch"]
        self.limit_sch_with_constant = event["limit_sch_with_constant"]
        self.sch_limit = event["sch_limit"] if self.limit_sch_with_constant else None

class Optimization_station:
    """
    This class encompasses the main optimizer at the station level.
    """
    def __init__(self, par, prb, station_info, k):
        self.Parameters = par
        self.Problem = prb
        self.opt_z = None
        self.opt_tariff_asap = None
        self.opt_tariff_flex = None
        self.opt_tariff_overstay = None
        self.station_info = copy.deepcopy(station_info) if station_info else None # "station" ndarray.
        self.k = k # Current global time indices(hour unit, for example 1.0, 1.25, 1.5, 1.75, 2.0...)
    def argmin_v(self, u, z):
        """
        Parameters 
        Decision Variables: 
        v: softmax probability[ sm(theta_sch, z), sm(theta_reg, z), sm(theta_leave, z) ], shape: (3,1)
        """
        ### Read parameters
        THETA = self.Problem.THETA 
        soft_v_eta = self.Parameters.soft_v_eta

        ### Decision Variables
        v = cp.Variable(shape = (3), pos = True)

        ### Objective Function
        J, J_array = self.constr_J(u, z, v)
        ### Log sum function conjugate: negative entropy 
        # lse_conj = - cp.sum(cp.entr(v))
        # func = v.T @ (THETA @ z)
        # # J_4 = mu * (lse_conj - func) 
        # constraints += [ v <= np.array((1,1,1))] # What is this? 
        # constraints += [ cp.sum(v) >= 1 - soft_v_eta ]

        ### Constraints
        constraints = [v >= 0]
        constraints += [cp.sum(v) == 1]
        constraints += [cp.log_sum_exp(THETA @ z) - cp.sum(cp.entr(v)) - v.T @ (THETA @ z) <= soft_v_eta]
        
        ### Solve
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        return np.round(v.value, 4)
    def argmin_z(self, u, v):
        """
        Function to determine prices 

        Decision Variables: 
        z: price [tariff_flex, tariff_asap, tariff_overstay, leave = 1 ]

        Parameters: 
        u, array, power for flex charging 
        v, array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
        lam_x, regularization parameter for sum squares of the power var (u)
        lam_z_c, regularization parameter for sum squares of the price flex (u)
        lam_z_uc, regularization parameter for sum squares of the price asap (u)
        lam_h_c, regularization parameter for g_flex
        lam_h_uc, regularization parameter for g_asap
        N_sch: timesteps arrival to departure 
        N_reg: timesteps required when charging at full capacity

        """
        # if sum(v) < 0 | (np.sum(v) < 1 - self.Parameters.soft_v_eta) | (np.sum(v) > 1 + self.Parameters.soft_v_eta):
        #     raise ValueError('[ ERROR] invalid $v$')
        
        ### Read parameters
        soft_v_eta = self.Parameters.soft_v_eta
        THETA = self.Problem.THETA

        ### Decision Variables
        z = cp.Variable(shape = (4), pos = True)

        ### Objective Function
        J, _ = self.constr_J(u, z, v)

        ### Constraints
        constraints = [z[3] == 1]
        constraints += [cp.log_sum_exp(THETA @ z) - cp.sum(cp.entr(v)) - v.T @ (THETA @ z) <= soft_v_eta]
        constraints += [z <= 40] # For better convergence guarantee.
        if self.Problem.limit_reg_with_sch:
            constraints += [z[1] <= z[0]]
        if self.Problem.limit_sch_with_constant:
            constraints += [z[0] == self.Problem.sch_limit]
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        return z.value
    def argmin_u(self, z, v):
        """
        Function to minimize charging cost. Flexible charging with variable power schedule
        Inputs: 

        Parameters: 
        z, array where [tariff_flex, tariff_asap, tariff_overstay, leave = 1 ]
        v, array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
        lam_x, regularization parameter for sum squares of the power var (u)
        lam_h_c, regularization parameter for g_flex
        lam_h_uc, regularization parameter for g_asap
        N_sch: timesteps arrival to departure 
        N_reg: timesteps required when charging at full capacity

        Parameters: 
        Decision Variables:
        SOC: state of charge (%)
        u: power (kW)

        Objective Function:
        Note: delta_k is not in the objective function!! 
        Check if it could make difference since power is kW cost is per kWh 

        Outputs
        u: power 
        SOC: SOC level 
        """

        ### Read parameters
        eff = 1
        delta_t = self.Parameters.Ts

        ### Decision Variables
        num_sch_user = len(self.station["SCH_list"]) + 1 # num of all SCH users
        e_delivered = cp.Variable(shape = ((self.var_dim_constant + 1) * num_sch_user, 1))
        u = cp.Variable(shape = (self.var_dim_constant * num_sch_user, 1))

        ### Objective Function
        J, _ = self.constr_J(u, z, v)

        ### Constraints (should incorporate all SCH users)
        constraints = [u >= 0]
        # The following constraints iterates through all existing flex users
        for i in range(num_sch_user):  # For all possible SCH users
            if i == 0:  # For the new user
                N_remain = self.Problem.N_sch
                e_need = self.Problem.e_need
                power_rate = self.Problem.power_rate
            else: # For existing users
                user_key = self.station["SCH_list"][i - 1]
                user = [d for d in self.station_info if d["dcosId"] == user_key][0]
                N_remain = int(user["N_remain"])
                e_need = float(user["energyNeeded"])
                power_rate = float(user["power_rate"])
            # Shape of e_delivered: (num_sch * (self.var_dim_constant + 1), 1)
            # Shape of u: (num_sch * self.var_dim_constant, 1)

            e_start = int(i * (self.var_dim_constant + 1))
            e_end = int(i * (self.var_dim_constant + 1) + N_remain)
            e_max = int(i * (self.var_dim_constant + 1) + self.var_dim_constant)
            u_start = int(i * self.var_dim_constant)
            u_end = int(i * self.var_dim_constant + N_remain)

            constraints += [u[u_start: u_end] <= power_rate]
            constraints += [e_delivered[e_start] == 0]
            constraints += [e_delivered[e_end] >= e_need]
            constraints += [e_delivered[e_start: e_max+1] <= e_need]
            # Implication: e_end = e_need.

            # Charging dynamics within each user
            for j in range(self.var_dim_constant):
                constraints += [e_delivered[j + e_start + 1] == e_delivered[j + e_start] + (float(eff) * delta_t * u[u_start + j])]

        ## Solve 
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        return u.value, e_delivered.value
    def constr_J(self, u, z, v):

        ### Read parameters for the new session
        N_reg = self.Problem.N_reg
        N_sch = self.Problem.N_sch
        TOU = self.Problem.TOU
        power_rate = self.Problem.power_rate
        N_reg_remainder = self.Problem.N_reg_remainder
        delta_t = self.Parameters.Ts

        ### Retrieve parameters for existing users
        existing_sch_obj = 0
        user_keys = self.station['SCH_list']
        num_sch_user = len(user_keys) + 1
        if user_keys:
            users = [d for d in self.station_info if d["dcosId"] in user_keys]
            for i in range(1, num_sch_user):  # EVs other than the new user
                # Here we need "i - 1", since the first row of existing_user_info is a new user
                user = users[i - 1]
                adj_constant = int(i * self.var_dim_constant)
                N_remain = user["N_remain"]
                TOU_idx = user["TOU_idx"]
                existing_sch_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user["TOU"][TOU_idx:] - user["price"]).reshape(-1, 1)

        existing_reg_obj = 0
        user_keys = self.station['REG_list']
        if user_keys:
            users = [d for d in self.station_info if d["dcosId"] in user_keys]
            TOU_idx = np.int_(self.k / delta_t - np.array([user["start_time"] for user in users]))
            existing_reg_obj = np.sum([user["optPower"][TOU_idx[i]:].T @ (
                    user["TOU"][TOU_idx[i]: user["N_reg"]] - user["price"]).reshape(-1, 1) for i, user
                                       in enumerate(users)])

        ## Existing user charging profile summation
        # REG
        reg_power_sum_profile = np.zeros(self.var_dim_constant)
        if self.station['REG_list']:
            users = [d for d in self.station_info if d["dcosId"] in self.station['REG_list']]
            for i in range(len(self.station['REG_list'])): # for all ASAP users
                user = users[i]
                TOU_idx = int(self.k / delta_t - user["start_time"])
                reg_power_sum_profile[: user["N_reg"] - TOU_idx] += user["optPower"][TOU_idx:].reshape(-1)
                # N_remain = N_reg - TOU_idx

        # SCH
        num_sch = len(self.station["SCH_list"]) + 1
        # Row: # of user, Col: Charging Profile
        sch_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_sch)).T
        sch_power_sum_profile = cp.sum(sch_power_sum_profile[1:, :], axis=0) # Shape: (self.var_dim_constant,)
        # The shape of sch_power_sum_profile is (self.var_dim_constant, 1)

        ## New user charging profile(ASAP)
        reg_new_user_profile = np.zeros(self.var_dim_constant)
        reg_new_user_profile[: N_reg - 1] = power_rate
        reg_new_user_profile[N_reg - 1] = (power_rate * N_reg_remainder) if N_reg_remainder > 0 else power_rate

        c_co = cp.reshape((TOU[:N_sch] - z[0]), (N_sch, 1))

        # Use cvxpy so all variables are cvxpy variables
        new_sch_obj = (u[: N_sch].T @ c_co) * delta_t
        new_reg_obj = (cp.sum(power_rate * (TOU[:N_reg - 1] - z[1])) + (power_rate * N_reg_remainder) * (
                    TOU[N_reg - 1] - z[1])) * delta_t if N_reg_remainder > 0 else cp.sum(
            power_rate * (TOU[:N_reg] - z[1])) * delta_t
        new_leave_obj = 0

        new_sch_obj = new_sch_obj.flatten()[0]
        new_reg_obj = new_reg_obj.flatten()[0]

        J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(
            reg_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_sch)).T, axis=0))) * v[0]
        J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(
            reg_power_sum_profile + sch_power_sum_profile + reg_new_user_profile)) * v[1]
        J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(
            reg_power_sum_profile + sch_power_sum_profile)) * v[2]

        J = J0 + J1 + J2
        J_array = np.array([J0.value, J1.value, J2.value])

        return J, J_array
    def run_opt(self):
        ### This is the main optimization function (multi-convex principal problem)
        ### station_info: a copy of the state dictionary. The re-optimized profiles will be passed in this dictionary.

        start = timeit.timeit()

        ### Existing FLEX Users
        station = dict()
        self.station = station
        station["REG_list"] = list()
        station["SCH_list"] = list()
        if self.station_info:
            station_info = self.station_info # External input: all information about the states
            for user in station_info:
                if user["choice"] == "REG":
                    station["REG_list"].append(user["dcosId"])
                elif user["choice"] == "SCH":
                    station["SCH_list"].append(user["dcosId"])

        # Remove finished users
        # We first check if the users in "SCH_list" are still in the station.(N_remain > 0?)

        # user_keys = station['SCH_list'].copy()
        # if user_keys:
        #     for user_key in user_keys:
        #         user = [d for d in station_info if d["dcosId"] == user_key][0]
        #         start_time = user["start_time"] / self.Parameters.Ts
        #         end_time = user["end_time"] / self.Parameters.Ts
        #         TOU_idx = int(self.k / self.Parameters.Ts - start_time)
        #         N_remain = int(end_time - self.k / self.Parameters.Ts) # Number of intervals left for the existing users
        #         e_needed_now = user["energyNeeded"] - np.sum(user["optPower"][: TOU_idx] * self.Parameters.eff * self.Parameters.Ts)
        #         if N_remain <= 1e-5 or e_needed_now <= 1e-5: # If all flex intervals expire or the energy demand is met, remove the user
        #             station["SCH_list"].remove(user_key)

        # If they are still in the station, update the remaining energy needed for their following intervals(N_remain).

        # Update existing user info(e_needed), transform the input dict to internal nparray
        user_keys = station['SCH_list']
        num_sch_user = len(station['SCH_list'])
        if user_keys:
            users = [d for d in station_info if d["dcosId"] in user_keys]
            for i in range(len(user_keys)):
                user = users[i]
                user["start_time"] = int(user["start_time"] / self.Parameters.Ts) # The time user arrives
                user["end_time"] = int(user["end_time"] / self.Parameters.Ts) # The time user leaves
                # Number of intervals left for the existing users
                user["N_remain"] = int(user["end_time"] - self.k / self.Parameters.Ts)
                # Current local time indices for User i
                user["TOU_idx"] = int(self.k / self.Parameters.Ts - user["start_time"])
                user["TOU"] = self.Parameters.TOU[user["start_time"]: user["end_time"]].reshape(-1, 1)
                # How much power we already charged?
                user["energyNeeded"] = user["energyNeeded"] - np.sum(user["optPower"][: user["TOU_idx"]] * self.Parameters.eff * self.Parameters.Ts)

        # ### Existing REG Users(check if they are still there)
        # user_keys = station["REG_list"].copy()
        # if user_keys:
        #     for user_key in user_keys:
        #         user = [d for d in station_info if d["dcosId"] == user_key][0]
        #         end_time = user["end_time"] / self.Parameters.Ts
        #         N_remain = end_time - self.k / self.Parameters.Ts
        #         if N_remain <= 0:
        #             station["REG_list"].remove(user_key)

        # For the REG users, integrate their information in reg_user_info list.
        user_keys = station["REG_list"]
        if user_keys:
            users = [d for d in station_info if d["dcosId"] in user_keys]
            for i in range(len(user_keys)):
                user = users[i]
                user["start_time"] = int(user["start_time"] / self.Parameters.Ts)
                user["end_time"] = int(user["end_time"] / self.Parameters.Ts)
                user["N_remain"] = int(user["end_time"] - self.k / self.Parameters.Ts) # Number of intervals left
                user["TOU_idx"] = int(self.k / self.Parameters.Ts - user["start_time"])
                user["TOU"] = self.Parameters.TOU[user["start_time"]: user["end_time"]].reshape(-1, 1)
                user["N_reg"] = math.ceil((user["energyNeeded"] / user["power_rate"] / self.Parameters.eff * int(1 / self.Parameters.Ts)))

        ### New User information & All SCH users info (existing_user_info)
        if self.station_info:
            N_remain_all = [user["N_remain"] for user in station_info if user["choice"] == "SCH"]
        else:
            N_remain_all = []
        N_remain_all.append(self.Problem.N_sch)

        var_dim_constant = int(max(N_remain_all)) # maximum remaining duration of all users
        self.var_dim_constant = var_dim_constant

        # Initial values for decision variable U: Shape: (all possible SCH users * dimension_constant, 1)
        uk_flex = self.Problem.power_rate * np.zeros([var_dim_constant * (num_sch_user + 1), 1])

        def charging_revenue(z, u):

            # I did not modify or use this function in station-level opt. If you want to leverage it, please check the formulations.
            N_reg = self.Problem.N_reg
            TOU = self.Problem.TOU
            station_pow_max = self.Problem.power_rate

            delta_t = self.Parameters.Ts 

            f_flex = u.T @ (z[0]- TOU) * delta_t
            ## u : kW , z: cents / kWh, TOU : cents / kWh , delta_t : 1 \ h
            # f_asap = np.sum(station_pow_max * (z[1] - TOU[:N_reg])) * delta_t 

            N_reg_remainder  = self.Problem.N_reg_remainder 
            
            if N_reg_remainder > 0:
                f_asap = (np.sum(station_pow_max * (TOU[:N_reg - 1] - z[1])) + (station_pow_max * N_reg_remainder) * (TOU[N_reg - 1] - z[1]) )* delta_t 
                
            else: 
                f_asap = np.sum(station_pow_max * (TOU[:N_reg ] - z[1])) * delta_t

            return f_flex, f_asap

        # Iteration information

        itermax = 1000
        count = 0
        improve = np.inf

        zk = self.Parameters.z0
        vk = self.Parameters.v0

        ### THIS VALUES ARE STORED FOR DEBUGGING
        Jk = np.zeros((itermax))
        rev_flex = np.zeros((itermax))
        rev_asap = np.zeros((itermax))
        z_iter = np.zeros((4,itermax))
        v_iter = np.zeros((3,itermax))
        J_sub = np.zeros((3,itermax))

        while (count < itermax) & (improve >= 0) & (abs(improve) >= 0.00001):

            _, J_array = self.constr_J(uk_flex, zk, vk)

            Jk[count] = J_array.sum()
            J_sub[:, count] = J_array.reshape(3,)
            # rev_flex[count], rev_asap[count] = charging_revenue(zk, uk_flex)
            z_iter[:, count] = zk.reshape((4,))
            v_iter[:, count] = vk.reshape((3,))

            try:
                uk_flex, e_deliveredk_flex = self.argmin_u(zk, vk)
            except:
                print('uk is not updated')
                pass

            try:
                vk = self.argmin_v(uk_flex, zk)
            except:
                print('vk is not updated')
                pass

            try:
                zk = self.argmin_z(uk_flex, vk)
            except:
                print("zk is not updated")
                pass

            _, J_array = self.constr_J(uk_flex, zk, vk)
            improve = Jk[count] - J_array.sum()
            # print(J_func(zk, uk_flex, vk))
            count += 1

            if count >= 50:
                print("Too much time for iteration(iteration times exceed 50)")
                break

        print("After %d iterations," % count, "we got %f " % improve, "improvement, and claim convergence.")
        print("The prices(SCH & REG) are %f" %zk[0], "%f" %zk[1])

        # Update the existing users in the station (SCH): their charging profile after re-optimization.
        # We directly operate on the station_info.
        user_keys = station["SCH_list"]
        if user_keys:
            users = [d for d in station_info if d["dcosId"] in user_keys]
            for i in range(len(user_keys)):
                user = users[i] # SCH User i
                N_remain = user["N_remain"]
                TOU_idx = user["TOU_idx"]
                # Update the power profile, shape: (-1, 1)
                user["optPower"][TOU_idx:] = uk_flex[int((i + 1) * var_dim_constant): int((i + 1) * var_dim_constant + N_remain)].reshape(-1)

        ### Output the results
        opt = dict()
        opt['e_need'] = self.Problem.e_need

        # Part 1: Prices
        opt["z"] = zk
        opt["z_hr"] = zk * self.Problem.power_rate
        # cents / kwh and cents / hour
        opt["tariff_sch"] = zk[0]
        opt["tariff_reg"] = zk[1]
        opt["sch_centsPerHr"] = opt["z_hr"][0]
        opt["reg_centsPerHr"] = opt["z_hr"][1]

        # Part 2: Power Profiles
        opt["power_rate"] = self.Problem.power_rate
        opt["peak_pow"] = max(uk_flex)
        opt["sch_e_delivered"] = e_deliveredk_flex
        N_remain = int(self.Problem.user_duration)
        opt["sch_powers"] = uk_flex[: N_remain]

        # For a possible "NEW" "ASAP" user, we assume that it's at the maximum for all ASAP intervals
        reg_powers = np.ones((self.Problem.N_reg, 1)) * self.Problem.power_rate
        if self.Problem.N_reg_remainder != 0: # For the last time slot, ASAP may not occupy the whole slot.
            reg_powers[self.Problem.N_reg - 1] = self.Problem.power_rate * self.Problem.N_reg_remainder
        opt["reg_powers"] = reg_powers

        # If the REG and SCH are actually identical, we pick REG as final value.

        if self.Problem.assertion_flag == 1:
            opt["sch_powers"] = reg_powers

        # Part 3: Probability & Iteration Parameters Output

        opt["v"] = vk
        opt["prob_flex"] = vk[0]
        opt["prob_asap"] = vk[1]
        opt["prob_leave"] = vk[2]

        opt["N_sch"] = self.Problem.N_sch
        opt["N_reg"] = self.Problem.N_reg

        opt["J"] = Jk[:count]
        opt["J_sub"] = J_sub[:, :count]
        opt["z_iter"] = z_iter[:, :count]
        opt["v_iter"] = v_iter[:, :count]

        opt["num_iter"] = count
        opt["time_start"] = self.Problem.user_time
        opt["time_end_SCH"] = self.Problem.user_time + self.Problem.user_duration
        opt["time_end_REG"] = self.Problem.user_time + self.Problem.N_reg

        # Part 4: General Problem Space
        opt["prb"] = self.Problem
        opt["par"] = self.Parameters

        end = timeit.timeit()
        station_info = self.station_info

        return station_info, opt