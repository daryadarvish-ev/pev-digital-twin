import math
import os
import timeit
import warnings

import cvxpy as cp
import numpy as np
from numpy import ndarray

warnings.filterwarnings("ignore")

class Parameters:
    """
    Class to hold all parameters for all simulations which will be the same for each problem.
    """
    def __init__(self, 
                z0 = np.array([1,1,1,1]).reshape(4,1),
                v0 = np.array([0.333, 0.333, 0.333]).reshape(3,1) ,
                Ts = 0.25, # 1 control horizon interval = 0.25 global hour
                base_tarriff_overstay = 1.0, 
                eff = 1,  # Charging efficient, assumed to be 100%
                soft_v_eta = 1e-4, # For convex relaxation in constraints.
                opt_eps = 0.0001,
                TOU = np.ones((96,))): # Time-of-user charging tariff: 96 intervals, 1 day.

        # TOU Tariff in cents / kwh
        # TOU * power rate = cents  / hour

        # off-peak 0.194  cents / kwh
        TOU_tariff = np.ones((96,)) * 19.4
        ## 4 pm - 9 pm peak 0.385 cents / kwh
        TOU_tariff[64:84] = 38.5
        ## 9 am - 2 pm super off-peak 0.16.6 $ / kWh  to cents / kwh
        TOU_tariff[36:56] = 16.6

        self.v0 = v0
        self.z0 = z0
        self.Ts = Ts
        self.base_tariff_overstay = base_tarriff_overstay
        self.TOU = TOU_tariff
        self.eff = eff # power efficiency
        self.dcm_choices = ['charging with flexibility', 'charging asap', 'leaving without charging']
        self.soft_v_eta = soft_v_eta #softening equality constraint for v; to avoid numerical error
        self.opt_eps = opt_eps

        self.cost_dc = 100  # Cost for demand charge. This value is arbitrary now. A larger value means the charging profile will go average.
        self.cost_dc = 100  # Cost for demand charge. This value is arbitrary now. A larger value means the charging profile will go average.
        self.cost_dc = 0  # Cost for demand charge. This value is arbitrary now. A larger value means the charging profile will go average.
        # 18.8 --> 300 We can change this value to show the effect of station-level impact.

        assert len(self.TOU) == int(24 / self.Ts), "Mismatch between TOU cost array size and discretization steps"

class Problem:
    """
    This class encompasses the current user information which will change for every user optimization.

    time, int, user interval
    duration, int, number of charging intervals
    """
    def __init__(self, par ,**kwargs):
        self.par = par
        event = kwargs["event"]
        self.event = event
        
        self.user_time = event["time"]
        self.e_need = event["e_need"]

        self.user_duration = event["duration"]
        # self.user_overstay_duration = round(event["overstay_duration"] / par.Ts) * par.Ts
        self.station_pow_max = event["station_pow_max"]
        # self.station_pow_min = event["pow_min"]
        self.user_power_rate = event['user_power_rate']

        self.power_rate = min(self.user_power_rate,self.station_pow_max)

        self.dcm_charging_flex_params = np.array([[ - self.power_rate * 0.0184 / 2], [ self.power_rate * 0.0184 / 2], [0], [0]])
        #% DCM parameters for choice 1 -- charging with flexibility
        self.dcm_charging_asap_params = np.array([[self.power_rate * 0.0184 / 2], [- self.power_rate * 0.0184 / 2], [0],[0.341 ]])
        #% DCM parameters for choice 2 -- charging as soon as possible
        self.dcm_leaving_params = np.array([[self.power_rate * 0.005 / 2], [self.power_rate * 0.005 / 2], [0], [-1 ]])
        
        #% DCM parameters for choice 3 -- leaving without charging
        self.THETA = np.vstack((self.dcm_charging_flex_params.T, self.dcm_charging_asap_params.T,
                     self.dcm_leaving_params.T))
        # problem specifications
        self.N_flex = self.user_duration # charging duration that is not charged, hour
        
        ### IS THIS CORRECT? WHATS SOC NEED REPRESENTS? 
        # self.N_asap = math.floor((self.user_SOC_need - self.user_SOC_init) *
        #                          self.user_batt_cap / self.station_pow_max / par.eff / par.Ts)

        ## HERE 12 IS SELF CODED 
        self.N_asap = math.ceil((self.e_need / self.power_rate / par.eff * int(1 / par.Ts)))
        self.N_asap_remainder = (self.e_need / self.power_rate / par.eff * int(1 / par.Ts)) % 1

#         print(par.TOU) 
        if len(par.TOU) < self.user_time + self.user_duration: # if there is overnight chaarging 
            par.TOU = np.concatenate([par.TOU,par.TOU]) 

        self.TOU = par.TOU[self.user_time:(self.user_time + self.user_duration)]
#         print(self.TOU) 
        # self.TOU = interpolate.interp1d(np.arange(0, 24 - 0.25 + 0.1, 0.25), par.TOU, kind = 'nearest')(np.arange(self.user_time, 0.1 + self.user_time + self.user_duration - par.Ts, par.Ts)).T
        
        assert self.N_asap <= self.N_flex, print("Not enought time (n_asap,n_flex)",self.N_asap,self.N_flex)

class Optimization_station:
    """
    This class encompasses the main optimizer at the station level.
    """
    def __init__(self, par, prb, station, k):
        self.Parameters = par
        self.Problem = prb
        self.opt_z = None
        self.opt_tariff_asap = None
        self.opt_tariff_flex = None
        self.opt_tariff_overstay = None
        self.station = station # "station" dict: maintain two lists: "FLEX_user", "ASAP_user" and the corresponding user's "Problem"
        self.k = k # Current global time indices(hour unit, for example 1.0, 1.25, 1.5, 1.75, 2.0...)

    def argmin_v(self, u, z):

        """
        Parameters 
        Decision Variables: 
        v: price [ sm(theta_flex, z), sm(theta_asap, z), sm(theta_leave, z) ], (3,1)
        """
        ### Read parameters 

        THETA = self.Problem.THETA 
        soft_v_eta = self.Parameters.soft_v_eta

        ### Decision Variables
        v = cp.Variable(shape = (3), pos = True)

        def constr_J(v):
            N_asap = self.Problem.N_asap
            TOU = self.Problem.TOU
            station_pow_max = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            N_flex = self.Problem.N_flex

            # Existing FLEX user term. Equation (24) - (27) in Teng's Paper.
            user_keys = self.station['FLEX_list']
            existing_flex_obj = 0
            if user_keys:
                # For every adj_constant, it includes the charging profile of one user for N_max interval
                # (max remaining intervals) for all existing and the new user. The length of adj_constant is (N_max).
                for i in range(1, self.existing_user_info.shape[0]): # EVs other than the new user
                    adj_constant = int(i * self.var_dim_constant)
                    # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                    N_remain = int(self.existing_user_info[i, 2]) # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                    TOU_idx = int(self.existing_user_info[i, 3]) # TOU_index
                    user = self.station[user_keys[i - 1]] # Here we need "i - 1", since the first row of existing_user_info is a new user
                    existing_flex_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1) # No problem with indices

            # Exising ASAP user cost term. Equ (29) - (32)
            user_keys = self.station['ASAP_list']
            existing_asap_obj = 0
            if user_keys:
                for i in range(len(user_keys)):
                    user = self.station[user_keys[i]]
                    TOU_idx = int(self.k / delta_t - user.Problem.user_time)  # The local indice for the duration(len(TOU) is duration)
                    # existing_asap_obj += cp.sum(cp.multiply(user.asap_powers * (user.Problem.TOU[TOU_idx:user.Problem.N_asap] - user.price)))
                    existing_asap_obj += user.asap_powers[TOU_idx:].T @ (user.Problem.TOU[TOU_idx: user.Problem.N_asap] - user.price).reshape(-1, 1)

            # Existing user charging profile summation
            asap_power_sum_profile = np.zeros(self.var_dim_constant)
            for i in range(len(self.station['ASAP_list'])): # for all ASAP users
                user = self.station[user_keys[i]]
                TOU_idx = int(self.k / delta_t - user.Problem.user_time)
                asap_power_sum_profile[: user.Problem.N_asap - TOU_idx] += user.asap_powers[TOU_idx:].squeeze()

            # flex_power_sum_profile = np.zeros(self.var_dim_constant)
            # for i in range(len(self.station['FLEX_list'])): # for all ASAP users
            #     user = self.station[user_keys[i]]
            #     TOU_idx = int(self.k / delta_t - user.Problem.user_time)
            #     flex_power_sum_profile[N_max + 1: N_max + 1 + user.Problem.N_flex - TOU_idx] += user.flex_powers[TOU_idx:]

            num_flex = self.existing_user_info.shape[0]

            flex_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_flex)).T # Row: # of user, Col: Charging Profile
            flex_power_sum_profile = cp.sum(flex_power_sum_profile[1:, :], axis = 0)

            # New user charging profile(ASAP)

            asap_new_user_profile = np.zeros(self.var_dim_constant)
            asap_new_user_profile[: N_asap] = self.Problem.station_pow_max

            # Use cvxpy so all variables are cvxpy variables
            # Teng's Paper's Eq(17) - (20)
            new_flex_obj = u[: N_flex].T @ (TOU[:N_flex] - z[0]).reshape(-1, 1)
            new_asap_obj = cp.sum(station_pow_max * (TOU[:N_asap] - z[1]))
            new_leave_obj = 0

            J0 = (new_flex_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
            J1 = (new_asap_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + flex_power_sum_profile + asap_new_user_profile)) * v[1]
            J2 = (new_leave_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + flex_power_sum_profile)) * v[2]

            J = J0 + J1 + J2

            return J

        J = constr_J(v)

        ### Log sum function conjugate: negative entropy 
        # lse_conj = - cp.sum(cp.entr(v))
        # func = v.T @ (THETA @ z)
        # # J_4 = mu * (lse_conj - func) 
        # constraints += [ v <= np.array((1,1,1))] # What is this? 
        # constraints += [ cp.sum(v) >= 1 - soft_v_eta ]

        constraints = [v >= 0]
        constraints += [cp.sum(v) == 1]

        constraints += [cp.log_sum_exp(THETA @ z) - cp.sum(cp.entr(v)) - v.T @ (THETA @ z) <= soft_v_eta]
        
        ## Solve 
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        # try:
        #     # print(  "v",v.value)
        #     # print(  "status",prob.status)
        #     temp = v.value
        # except:
        #     print(  "status",prob.status)
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
        N_flex: timesteps arrival to departure 
        N_asap: timesteps required when charging at full capacity

        """
        # if sum(v) < 0 | (np.sum(v) < 1 - self.Parameters.soft_v_eta) | (np.sum(v) > 1 + self.Parameters.soft_v_eta):
        #     raise ValueError('[ ERROR] invalid $v$')
        
        ### Read parameters
#         vehicle_power_rate = self.Problem.power_rate
        soft_v_eta = self.Parameters.soft_v_eta
        THETA = self.Problem.THETA

        ### Decision Variables
        
        z = cp.Variable(shape = (4), pos = True)
        def constr_J(z):
            N_asap = self.Problem.N_asap
            TOU = self.Problem.TOU
            station_pow_max = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            N_flex = self.Problem.N_flex

            user_keys = self.station['FLEX_list']
            existing_flex_obj = 0
            if user_keys:
                for i in range(1, self.existing_user_info.shape[0]): # EVs other than the new user
                    adj_constant = int(i * self.var_dim_constant)
                    # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                    N_remain = int(self.existing_user_info[i, 2]) # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                    TOU_idx = int(self.existing_user_info[i, 3]) # TOU_index
                    user = self.station[user_keys[i - 1]] # Here we need "i - 1", since the first row of existing_user_info is a new user
                    existing_flex_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1) # No problem with indices

            user_keys = self.station['ASAP_list']
            existing_asap_obj = 0
            if user_keys:
                for i in range(len(user_keys)):
                    user = self.station[user_keys[i]]
                    TOU_idx = int(self.k / delta_t - user.Problem.user_time)  # The local indice for the duration(len(TOU) is duration)
                    # existing_asap_obj += cp.sum(cp.multiply(user.asap_powers * (user.Problem.TOU[TOU_idx:user.Problem.N_asap] - user.price)))
                    existing_asap_obj += user.asap_powers[TOU_idx:].T @ (user.Problem.TOU[TOU_idx: user.Problem.N_asap] - user.price).reshape(-1, 1)

            # Existing user charging profile summation
            asap_power_sum_profile = np.zeros(self.var_dim_constant)
            for i in range(len(self.station['ASAP_list'])): # for all ASAP users
                user = self.station[user_keys[i]]
                TOU_idx = int(self.k / delta_t - user.Problem.user_time)
                asap_power_sum_profile[: user.Problem.N_asap - TOU_idx] += user.asap_powers[TOU_idx:].squeeze()

            # flex_power_sum_profile = np.zeros(self.var_dim_constant)
            # for i in range(len(self.station['FLEX_list'])): # for all ASAP users
            #     user = self.station[user_keys[i]]
            #     TOU_idx = int(self.k / delta_t - user.Problem.user_time)
            #     flex_power_sum_profile[N_max + 1: N_max + 1 + user.Problem.N_flex - TOU_idx] += user.flex_powers[TOU_idx:]

            num_flex = self.existing_user_info.shape[0]

            flex_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_flex)).T # Row: # of user, Col: Charging Profile
            flex_power_sum_profile = cp.sum(flex_power_sum_profile[1:, :], axis = 0)

            # New user charging profile(ASAP)

            asap_new_user_profile = np.zeros(self.var_dim_constant)
            asap_new_user_profile[: N_asap] = self.Problem.station_pow_max

            c_co = cp.reshape((TOU[:N_flex] - z[0]), (N_flex, 1))

            # Use cvxpy so all variables are cvxpy variables
            new_flex_obj = u[: N_flex].T @ c_co
            new_asap_obj = cp.sum(station_pow_max * (TOU[:N_asap] - z[1]))
            new_leave_obj = 0

            J0 = (new_flex_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
            J1 = (new_asap_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + flex_power_sum_profile + asap_new_user_profile)) * v[1]
            J2 = (new_leave_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + flex_power_sum_profile)) * v[2]

            J = J0 + J1 + J2

            return J

        J = constr_J(z)

        constraints = [z[3] == 1]
        constraints += [cp.log_sum_exp(THETA @ z) - cp.sum(cp.entr(v)) - v.T @ (THETA @ z) <= soft_v_eta]
        constraints += [z <= 40] # For better convergence guarantee.
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        # try:
        #     # print("z",np.round(z.value,5))
        #     temp = np.round(z.value,5)
        # except:
        #     print(  "z status",prob.status)
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
        N_flex: timesteps arrival to departure 
        N_asap: timesteps required when charging at full capacity

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

        station_pow_max = self.Problem.power_rate

        N_max = self.var_dim_constant
        eff = 1
        delta_t = self.Parameters.Ts

        # if sum(v) < 0 | (np.sum(v) < 1 - self.Parameters.soft_v_eta) | (np.sum(v) > 1 + self.Parameters.soft_v_eta):
        #     raise ValueError('[ ERROR] invalid $v$')

        ### Decision Variables
        num_user = self.existing_user_info.shape[0] # No. all flex users
        e_delivered = cp.Variable(shape = ((self.var_dim_constant + 1) * num_user, 1))
        u = cp.Variable(shape = (self.var_dim_constant * num_user, 1))

        def constr_J(u):
            N_asap = self.Problem.N_asap
            TOU = self.Problem.TOU
            station_pow_max = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            N_flex = self.Problem.N_flex

            user_keys = self.station['FLEX_list']
            existing_flex_obj = 0
            if user_keys:
                for i in range(1, self.existing_user_info.shape[0]): # EVs other than the new user
                    adj_constant = int(i * self.var_dim_constant)
                    # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                    N_remain = int(self.existing_user_info[i, 2]) # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                    TOU_idx = int(self.existing_user_info[i, 3]) # TOU_index
                    user = self.station[user_keys[i - 1]] # Here we need "i - 1", since the first row of existing_user_info is a new user
                    existing_flex_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1) # No problem with indices

            user_keys = self.station['ASAP_list']
            existing_asap_obj = 0
            if user_keys:
                for i in range(len(user_keys)):
                    user = self.station[user_keys[i]]
                    TOU_idx = int(self.k / delta_t - user.Problem.user_time)  # The local indice for the duration(len(TOU) is duration)
                    # existing_asap_obj += cp.sum(cp.multiply(user.asap_powers * (user.Problem.TOU[TOU_idx:user.Problem.N_asap] - user.price)))
                    existing_asap_obj += user.asap_powers[TOU_idx:].T @ (user.Problem.TOU[TOU_idx: user.Problem.N_asap] - user.price).reshape(-1, 1)

            # Existing user charging profile summation
            asap_power_sum_profile = np.zeros(self.var_dim_constant)
            for i in range(len(self.station['ASAP_list'])): # for all ASAP users
                user = self.station[user_keys[i]]
                TOU_idx = int(self.k / delta_t - user.Problem.user_time)
                asap_power_sum_profile[: user.Problem.N_asap - TOU_idx] += user.asap_powers[TOU_idx:].squeeze()

            # flex_power_sum_profile = np.zeros(self.var_dim_constant)
            # for i in range(len(self.station['FLEX_list'])): # for all ASAP users
            #     user = self.station[user_keys[i]]
            #     TOU_idx = int(self.k / delta_t - user.Problem.user_time)
            #     flex_power_sum_profile[N_max + 1: N_max + 1 + user.Problem.N_flex - TOU_idx] += user.flex_powers[TOU_idx:]

            num_flex = self.existing_user_info.shape[0]

            flex_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_flex)).T # Row: # of user, Col: Charging Profile
            flex_power_sum_profile = cp.sum(flex_power_sum_profile[1:, :], axis = 0)

            # New user charging profile(ASAP)

            asap_new_user_profile = np.zeros(self.var_dim_constant)
            asap_new_user_profile[: N_asap] = self.Problem.station_pow_max

            # Use cvxpy so all variables are cvxpy variables
            new_flex_obj = u[: N_flex].T @ (TOU[:N_flex] - z[0]).reshape(-1, 1)
            new_asap_obj = cp.sum(station_pow_max * (TOU[:N_asap] - z[1]))
            new_leave_obj = 0

            J0 = (new_flex_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
            J1 = (new_asap_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + flex_power_sum_profile + asap_new_user_profile)) * v[1]
            J2 = (new_leave_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + flex_power_sum_profile)) * v[2]

            J = J0 + J1 + J2

            return J

        J = constr_J(u)

        ## Constraints should incorporate all existing users
        constraints = [u >= 0]
        constraints += [u <= station_pow_max]

        ## Influence of existing user.
        ## THe following constraints means that: for each flex user, we shall meet the total energy demand when the charging
        # is over.

        user_keys = self.station['FLEX_list']
        for i in range(self.existing_user_info.shape[0]):  # For all possible flex users
            
            N_remain = int(self.existing_user_info[i, 2])
            e_need = self.existing_user_info[i, 4]

            e_start = int(i * (self.var_dim_constant + 1))
            e_end = int(i * (self.var_dim_constant + 1) + N_remain)
            e_max = int(i * (self.var_dim_constant + 1) + self.var_dim_constant)
            u_start = int(i * self.var_dim_constant)

            constraints += [e_delivered[e_start] == 0]
            constraints += [e_delivered[e_end] >= e_need]
            constraints += [e_delivered[e_start: e_max+1] <= e_need]
            for j in range(self.var_dim_constant):
                constraints += [e_delivered[j + e_start + 1] == e_delivered[j + e_start] + (float(eff) * delta_t * u[u_start + j])]

        ## Solve 
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        # try:
        #     print("u:",np.round(u.value,2 ))
        # except:
        #     print("status",prob.status)
        return u.value, e_delivered.value

    def run_opt(self):
        # This is the main optimization function(multi-convex principal problem)

        start = timeit.timeit()

        ### Existing FLEX Users

        # Remove finished users
        # We first check if the users in "FLEX_list" are still in the station.(N_remain > 0?)

        user_keys = self.station['FLEX_list'].copy()
        if user_keys:
            for user_key in user_keys:
                user = self.station[user_key].Problem
                end_time = user.user_time + user.user_duration
                TOU_idx = int(self.k / self.Parameters.Ts - user.user_time)
                N_remain = int(end_time - self.k / self.Parameters.Ts) # Number of intervals left for the existing users
                e_needed_now = user.e_need - np.sum(user.powers[: TOU_idx] * self.Parameters.eff * self.Parameters.Ts)
                if N_remain <= 1e-5 or e_needed_now <= 1e-5: # If all flex intervals expire or the energy demand is met, remove the user
                    del(self.station[user_key])
                    self.station["FLEX_list"].remove(user_key)

        # If they are still in the station, update the remaining energy needed for their following intervals(N_remain).
        # Update existing user info(e_needed)
        user_keys = self.station['FLEX_list']
        num_flex_user = len(self.station['FLEX_list'])
        flex_user_info = np.zeros([num_flex_user, 5]) # [user_key, user_time, N_remain, user_duration, e_need]
        if user_keys:
            for i in range(num_flex_user):
                user = self.station[user_keys[i]].Problem
                start_time = user.user_time
                end_time = user.user_time + user.user_duration
                N_remain = int(end_time - self.k / self.Parameters.Ts) # Number of intervals left for the existing users
                TOU_idx = int(self.k / self.Parameters.Ts - start_time) # Current local time indices for User i
                # How much power we already charged?
                e_needed_now = user.e_need - np.sum(user.powers[: TOU_idx] * self.Parameters.eff * self.Parameters.Ts)
                flex_user_info[i, :] = [start_time, end_time, N_remain, TOU_idx, e_needed_now]

        ### Existing ASAP Users(check if they are still there)
        user_keys = self.station["ASAP_list"].copy()

        if user_keys:
            for user_key in user_keys:
                user = self.station[user_key].Problem
                end_time = user.user_time + user.N_asap
                N_remain = end_time - self.k / self.Parameters.Ts
                if N_remain <= 0:
                    del(self.station[user_key])
                    self.station["ASAP_list"].remove(user_key)

        # For the ASAP users, integrate their information in asap_user_info list.
        user_keys = self.station["ASAP_list"]
        num_asap_user = len(self.station['ASAP_list'])
        asap_user_info = np.zeros([num_asap_user, 5])
        if user_keys:
            for i in range(len(user_keys)):
                user = self.station[user_keys[i]].Problem
                start_time = user.user_time
                end_time = user.user_time + user.N_asap
                N_remain = int(end_time - self.k / self.Parameters.Ts) # Number of intervals left for the existing users
                TOU_idx = int(self.k / self.Parameters.Ts - start_time) # Current local time indices for User i
                asap_user_info[i, :] = [start_time, end_time, N_remain, TOU_idx, 0]


        ### New User information, incorporate it in self.existing_user_info
        new_user = self.Problem # The struct for the incoming user
        start_time = new_user.user_time
        existing_user_info = np.array([[start_time, -1, new_user.N_flex, 0, self.Problem.e_need]]) # Actually, all existing flex user info.
        existing_user_info = np.concatenate((existing_user_info, flex_user_info), axis = 0)
        self.existing_user_info = existing_user_info
        # Concatenate, and pick the largest interval as the Power Profile length.

        var_dim_constant = int(max(np.concatenate((existing_user_info, asap_user_info), axis = 0)[:, 2])) # chosen from maximum remaining duration for all EVs
        self.var_dim_constant = var_dim_constant

        # Initial values for uk
        uk_flex = self.Problem.station_pow_max * np.ones([var_dim_constant * (num_flex_user + 1), 1]) # We are optimizing the FLEX profile, so the dimension is all possible flex users * dimension_con
        
        def J_func(z, u, v):
            # See the detailed comments of all J_func funcitons in self.argmin_u() (All of them are nearly identical)
            N_asap = self.Problem.N_asap
            TOU = self.Problem.TOU
            station_pow_max = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            N_flex = self.Problem.N_flex


            user_keys = self.station['FLEX_list']
            existing_flex_obj = 0
            if user_keys:
                for i in range(1, self.existing_user_info.shape[0]): # EVs other than the new user
                    adj_constant = int(i * self.var_dim_constant)
                    # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                    N_remain = int(self.existing_user_info[i, 2]) # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                    TOU_idx = int(self.existing_user_info[i, 3]) # TOU_index
                    user = self.station[user_keys[i - 1]] # Here we need "i - 1", since the first row of existing_user_info is a new user
                    existing_flex_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1) # No problem with indices

            user_keys = self.station['ASAP_list']
            existing_asap_obj = 0
            if user_keys:
                for i in range(len(user_keys)):
                    user = self.station[user_keys[i]]
                    TOU_idx = int(self.k / delta_t - user.Problem.user_time)  # The local indice for the duration(len(TOU) is duration)
                    # existing_asap_obj += cp.sum(cp.multiply(user.asap_powers * (user.Problem.TOU[TOU_idx:user.Problem.N_asap] - user.price)))
                    existing_asap_obj += user.asap_powers[TOU_idx:].T @ (user.Problem.TOU[TOU_idx: user.Problem.N_asap] - user.price).reshape(-1, 1)

            # Existing user charging profile summation
            asap_power_sum_profile = np.zeros(self.var_dim_constant)
            for i in range(len(self.station['ASAP_list'])): # for all ASAP users
                user = self.station[user_keys[i]]
                TOU_idx = int(self.k / delta_t - user.Problem.user_time)


                try:
                    asap_power_sum_profile[: user.Problem.N_asap - TOU_idx] += user.asap_powers[TOU_idx:user.Problem.N_asap].squeeze()
                except:
                    print(' user_asap_power', user.asap_powers)
                    print('sum_profile', asap_power_sum_profile)

            num_flex = self.existing_user_info.shape[0]

            flex_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_flex)).T # Row: # of user, Col: Charging Profile
            flex_power_sum_profile = cp.sum(flex_power_sum_profile[1:, :], axis = 0)

            # New user charging profile(ASAP)

            asap_new_user_profile = np.zeros(self.var_dim_constant)
            asap_new_user_profile[: N_asap] = self.Problem.station_pow_max

            # Use cvxpy so all variables are cvxpy variables
            new_flex_obj = u[: N_flex].T @ (TOU[:N_flex] - z[0]).reshape(-1, 1)

            try :
                new_asap_obj = cp.sum(station_pow_max * (TOU[:N_asap] - z[1]))
            except:
                print('what is this', new_asap_obj)
            new_leave_obj = 0

            J0 = (new_flex_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
            J1 = (new_asap_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + flex_power_sum_profile + asap_new_user_profile)) * v[1]
            J2 = (new_leave_obj + existing_flex_obj + existing_asap_obj + self.Parameters.cost_dc * cp.max(asap_power_sum_profile + flex_power_sum_profile)) * v[2]

            return np.array([J0.value, J1.value, J2.value])
        def charging_revenue(z, u):

            # I did not modify or use this function in station-level opt. If you want to leverage it, please check the formulations.
            N_asap = self.Problem.N_asap
            TOU = self.Problem.TOU
            station_pow_max = self.Problem.power_rate

            delta_t = self.Parameters.Ts 

            f_flex = u.T @ (z[0]- TOU) * delta_t
            ## u : kW , z: cents / kWh, TOU : cents / kWh , delta_t : 1 \ h
            # f_asap = np.sum(station_pow_max * (z[1] - TOU[:N_asap])) * delta_t 

            N_asap_remainder  = self.Problem.N_asap_remainder 
            
            if N_asap_remainder > 0:
                f_asap = (np.sum(station_pow_max * (TOU[:N_asap - 1] - z[1])) + (station_pow_max * N_asap_remainder) * (TOU[N_asap - 1] - z[1]) )* delta_t 
                
            else: 
                f_asap = np.sum(station_pow_max * (TOU[:N_asap ] - z[1])) * delta_t

            return f_flex, f_asap

        # Iteration information

        itermax = 1000
        count = 0
        improve = np.inf

        zk = self.Parameters.z0
        vk = self.Parameters.v0

        ###     THIS VALUES ARE STORED FOR DEBUGGING     ##
        Jk = np.zeros((itermax))
        rev_flex = np.zeros((itermax))
        rev_asap = np.zeros((itermax))
        z_iter = np.zeros((4,itermax))
        v_iter = np.zeros((3,itermax))
        J_sub = np.zeros((3,itermax))

        while (count < itermax) & (improve >= 0) & (abs(improve) >= 0.00001):

            Jk[count] = J_func(zk, uk_flex, vk).sum()
            J_sub[:, count] = J_func(zk, uk_flex, vk).reshape(3,)
            # rev_flex[count], rev_asap[count] = charging_revenue(zk, uk_flex)
            z_iter[:, count] = zk.reshape((4,))
            v_iter[:, count] = vk.reshape((3,))
            try:
                uk_flex, e_deliveredk_flex = self.argmin_u(zk, vk)
            except:
                print('uk is not updated')
                pass

            uk = uk_flex.reshape(-1, self.var_dim_constant)
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

            improve = Jk[count] - J_func(zk, uk_flex, vk).sum()
            # print(J_func(zk, uk_flex, vk))
            count += 1

        # Iteration finished.
        if zk[0] >= 30:
            zk = z_iter[:, 1]
            vk = v_iter[:, 1]

        if zk[0] >= 30:
            zk = z_iter[:, 1]
            vk = v_iter[:, 1]

        print("After %d iterations," % count, "we got %f " % improve, "improvements, and claim convergence.")
        print("The prices for flex : %f" %zk[0], ",  asap : %f" %zk[1])

        # Iteration Ends. Now we need to: 1. update flex user profile 2. Output [station, res]
        N_max = self.var_dim_constant # The maximum possible intervals

        # Update the existing users in the station (FLEX): their charging profile after optimization.
        user_keys = self.station["FLEX_list"]
        if user_keys:
            for i in range(len(user_keys)):
                user = self.station[user_keys[i]].Problem
                end_time = user.user_time + user.user_duration
                N_remain = int(end_time - self.k / self.Parameters.Ts)
                TOU_idx = int(self.k / self.Parameters.Ts - user.user_time)
                # Update the power profile
                user.powers = user.powers.reshape(-1, 1)
                user.powers[TOU_idx:] = uk_flex[int((i + 1) * var_dim_constant): int((i + 1) * var_dim_constant + N_remain)]
                # Update the dict.
                self.station[user_keys[i]].Problem = user

        opt = {}
        opt['e_need'] = self.Problem.e_need
        opt["z"] = zk
        opt["z_hr"] = zk * self.Problem.power_rate
        # Add a self.price = chosen mode price in outer loop by calling all z values in optimizer
        opt["tariff_flex"] = zk[0]
        opt["tariff_asap"] = zk[1]
        # opt["x"] = xk
        # update demand charge
        opt["peak_pow"] = max(uk_flex)
        opt["flex_e_delivered"] = e_deliveredk_flex
        N_remain = int(self.Problem.user_duration)
        opt["flex_powers"] = uk_flex[: N_remain]
        self.Problem.powers = uk_flex[: N_remain]

        # For a possible "NEW" "ASAP" user, we assume that it's at the maximum for all ASAP intervals
        asap_powers = np.ones((self.Problem.N_asap, 1)) * self.Problem.station_pow_max
        if self.Problem.N_asap_remainder != 0: # For the last time slot, ASAP may not occupy the whole slot.
            asap_powers[self.Problem.N_asap - 1] = self.Problem.station_pow_max * self.Problem.N_asap_remainder
        opt["asap_powers"] = asap_powers
        self.asap_powers = asap_powers

        self.flex_poewrs = opt["flex_powers"]

        opt["v"] = vk
        opt["prob_flex"] = vk[0]
        opt["prob_asap"] = vk[1]
        opt["prob_leave"] = vk[2]
        opt["N_flex"] = self.Problem.N_flex
        opt["N_asap"] = self.Problem.N_asap

        opt["N_flex"] = self.Problem.N_flex
        opt["N_asap"] = self.Problem.N_asap

        # flex_power_sum_profile = uk_flex.reshape(-1, self.var_dim_constant)  # Row: # of user, Col: Charging Profile
        # flex_power_sum_profile = np.sum(flex_power_sum_profile, axis=0)
        #
        # asap_power_sum_profile = np.zeros(self.var_dim_constant)
        # user_keys = self.station["ASAP_list"]
        # for i in range(len(self.station['ASAP_list'])):  # for all ASAP users
        #     try:
        #         user = self.station[user_keys[i]]
        #     except:
        #         print("ASAP user not found")
        #     TOU_idx = int(self.k / self.Parameters.Ts - user.Problem.user_time)
        #     asap_power_sum_profile[: user.Problem.N_asap - TOU_idx] += user.asap_powers[TOU_idx:].squeeze()
        # power_sum = flex_power_sum_profile + asap_power_sum_profile

        # opt["power_sum"] = power_sum.reshape(-1, 1)
        # opt["power_sum_N"] = N_max

        opt["J"] = Jk[:count]
        opt["J_sub"] = J_sub[:, :count]
        opt["z_iter"] = z_iter[:, :count]
        opt["v_iter"] = v_iter[:, :count]
        opt["rev_flex"] = rev_flex[:count]
        opt["rev_asap"] = rev_asap[:count]

        opt["num_iter"] = count
        opt["prb"] = self.Problem
        opt["par"] = self.Parameters
        opt["time_start"] = self.Problem.user_time
        opt["time_end_flex"] = self.Problem.user_time + self.Problem.user_duration
        opt["time_end_asap"] = self.Problem.user_time + self.Problem.N_asap 
        end = timeit.timeit()

        station = self.station # We update the station struct every round.

        return station, opt

class Optimization_charger:

    def __init__(self, par, prb):
        self.Parameters = par
        self.Problem = prb
        self.opt_z = None
        self.opt_tariff_asap = None
        self.opt_tariff_flex = None
        self.opt_tariff_overstay = None

    def argmin_v(self, u, z):

        """

        Parameters
        Decision Variables:
        v: price [ sm(theta_flex, z), sm(theta_asap, z), sm(theta_leave, z) ], (3,1)

        """

        ### Read parameters

        N_flex = self.Problem.N_flex
        N_asap = self.Problem.N_asap
        N_asap_remainder = self.Problem.N_asap_remainder
        TOU = self.Problem.TOU
        station_pow_max = self.Problem.power_rate

        # mu = self.Parameters.mu
        THETA = self.Problem.THETA
        soft_v_eta = self.Parameters.soft_v_eta
        delta_t = self.Parameters.Ts

        ### Decision Variables
        v = cp.Variable(shape=(3), pos=True)

        ### Define objective function
        # Flex Charging
        # reg_flex =  cp.norm(u,2) * lam_x + cp.norm(z[0],2) * lam_z_c

        f_flex = u.T @ (TOU - z[0]) * delta_t

        J_1 = v[0] * (f_flex)

        # ASAP Charging
        # reg_asap =  cp.norm(z[1],2) * lam_z_uc

        # We
        if N_asap_remainder > 0:
            if N_asap <= 1:
                f_asap = ((station_pow_max * N_asap_remainder) * (TOU[N_asap - 1] - z[1])) * delta_t
                # print(self.Problem.N_asap)
            # print(N_asap_remainder, N_asap_remainder.shape)
            # print(TOU[:N_asap - 1], TOU[:N_asap - 1].shape)
            else:
                f_asap = (cp.sum(station_pow_max * (TOU[:N_asap - 1] - z[1])) + (station_pow_max * N_asap_remainder) * (
                            TOU[N_asap - 1] - z[1])) * delta_t

        else:
            f_asap = cp.sum(station_pow_max * (TOU[:N_asap] - z[1])) * delta_t

        J_2 = v[1] * (f_asap)

        # Leave
        # J_3 = v[2] * cp.sum(TOU[:N_asap] * station_pow_max) * delta_t
        J_3 = 0
        J = J_1 + J_2 + J_3

        ### Log sum function conjugate: negative entropy
        # lse_conj = - cp.sum(cp.entr(v))
        # func = v.T @ (THETA @ z)
        # # J_4 = mu * (lse_conj - func)
        # constraints += [ v <= np.array((1,1,1))] # What is this?
        # constraints += [ cp.sum(v) >= 1 - soft_v_eta ]

        constraints = [v >= 0]
        constraints += [cp.sum(v) == 1]
        # constraints += [v[2] <= 0.50 ]

        constraints += [cp.log_sum_exp(THETA @ z) - cp.sum(cp.entr(v)) - v.T @ (THETA @ z) <= soft_v_eta]

        ## Solve
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        # try:
        #     # print(  "v",v.value)
        #     # print(  "status",prob.status)
        #     temp = v.value
        # except:
        #     print(  "status",prob.status)
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
        N_flex: timesteps arrival to departure
        N_asap: timesteps required when charging at full capacity

        """
        #############################################  Erase this part ##############################################
        # print("what is v", v)
        # if sum(v) < 0 | (np.sum(v) < 1 - self.Parameters.soft_v_eta) | (np.sum(v) > 1 + self.Parameters.soft_v_eta):
        #     # print("what is v", v)
        #     # print("NP sum V", np.sum(v), "\n self.parameters.sfotveta", self.Parameters.soft_v_eta)
        #     raise ValueError('[ ERROR] invalid $v$')

        ### Read parameters
        N_flex = self.Problem.N_flex
        N_asap = self.Problem.N_asap
        TOU = self.Problem.TOU
        station_pow_max = self.Problem.power_rate
        #         vehicle_power_rate = self.Problem.power_rate
        delta_t = self.Parameters.Ts
        soft_v_eta = self.Parameters.soft_v_eta
        THETA = self.Problem.THETA

        ### Decision Variables

        z = cp.Variable(shape=(4), pos=True)

        f_flex = u.T @ (TOU - z[0]) * delta_t
        # g_flex = lam_h_c * cp.inv_pos(z[2])

        J_1 = v[0] * (f_flex)

        # ASAP Charging
        N_asap_remainder = self.Problem.N_asap_remainder
        # print(N_asap_remainder)

        if N_asap_remainder > 0:
            if N_asap <= 1:
                f_asap = ((station_pow_max * N_asap_remainder) * (TOU[N_asap - 1] - z[1])) * delta_t
                # print(self.Problem.N_asap)
            # print(N_asap_remainder, N_asap_remainder.shape)
            # print(TOU[:N_asap - 1], TOU[:N_asap - 1].shape)
            else:
                f_asap = (cp.sum(station_pow_max * (TOU[:N_asap - 1] - z[1])) + (station_pow_max * N_asap_remainder) * (
                            TOU[N_asap - 1] - z[1])) * delta_t

        else:
            f_asap = cp.sum(station_pow_max * (TOU[:N_asap] - z[1])) * delta_t

            # g_asap =  lam_h_c* cp.inv_pos(z[2])

        J_2 = v[1] * (f_asap)
        # Leave
        J_3 = 0

        J = J_1 + J_2 + J_3

        ### Log sum function
        # lse = cp.log_sum_exp(THETA @ z)
        # func = z.T @ (THETA.T @ v)
        # J_4 = mu * (lse - func)
        constraints = [z[3] == 1]
        constraints += [cp.log_sum_exp(THETA @ z) - cp.sum(cp.entr(v)) - v.T @ (THETA @ z) <= soft_v_eta]

        ## Solve
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)

        prob.solve()

        # try:
        #     # print("z",np.round(z.value,5))
        #     temp = np.round(z.value,5)
        # except:
        #     print(  "z status",prob.status)

        return z.value

    def argmin_x(self, z, v):
        """
        Function to minimize charging cost. Flexible charging with variable power schedule
        Inputs:

        Parameters:
        z, array where [tariff_flex, tariff_asap, tariff_overstay, leave = 1 ]
        v, array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
        lam_x, regularization parameter for sum squares of the power var (u)
        lam_h_c, regularization parameter for g_flex
        lam_h_uc, regularization parameter for g_asap
        N_flex: timesteps arrival to departure
        N_asap: timesteps required when charging at full capacity

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
        N_flex = self.Problem.N_flex
        N_asap = self.Problem.N_asap
        TOU = self.Problem.TOU
        station_pow_max = self.Problem.power_rate
        vehicle_power_rate = self.Problem.power_rate
        e_need = self.Problem.e_need
        eff = 1
        # user_bat_cap = self.Problem.user_batt_cap
        delta_t = self.Parameters.Ts
        soft_v_eta = self.Parameters.soft_v_eta

        if sum(v) < 0 | (np.sum(v) < 1 - self.Parameters.soft_v_eta) | (np.sum(v) > 1 + self.Parameters.soft_v_eta):
            raise ValueError('[ ERROR] invalid $v$')

        ### Decision Variables
        e_delivered = cp.Variable(shape=(N_flex + 1))
        u = cp.Variable(shape=(N_flex))
        f_flex = u.T @ (TOU - z[0]) * delta_t
        # g_flex = lam_h_c * cp.inv_pos(z[2])

        J_1 = v[0] * (f_flex)

        # ASAP Charging
        # reg_asap =  cp.norm(z[1],2) * lam_z_uc
        # f_asap = cp.sum(station_pow_max * (TOU[:N_asap] - z[1])) * delta_t
        N_asap_remainder = self.Problem.N_asap_remainder

        if N_asap_remainder > 0:
            if N_asap <= 1:
                f_asap = ((station_pow_max * N_asap_remainder) * (TOU[N_asap - 1] - z[1])) * delta_t
                # print(self.Problem.N_asap)
            # print(N_asap_remainder, N_asap_remainder.shape)
            # print(TOU[:N_asap - 1], TOU[:N_asap - 1].shape)
            else:
                f_asap = (cp.sum(station_pow_max * (TOU[:N_asap - 1] - z[1])) + (station_pow_max * N_asap_remainder) * (
                            TOU[N_asap - 1] - z[1])) * delta_t

        else:
            f_asap = cp.sum(station_pow_max * (TOU[:N_asap] - z[1])) * delta_t

            # g_asap = lam_h_uc * cp.inv_pos(z[2])
        J_2 = v[1] * (f_asap)
        # Leave
        # J_3 = v[2] * cp.sum(TOU[:N_asap] * station_pow_max * delta_t)
        J_3 = 0

        J = J_1 + J_2 + J_3

        ## Constraints

        constraints = [e_delivered[0] == 0]
        constraints += [e_delivered[N_flex] >= e_need]
        constraints += [e_delivered <= e_need]
        constraints += [u >= 0]
        constraints += [u <= station_pow_max]

        # System dynamics
        for i in range(0, N_flex):
            constraints += [e_delivered[i + 1] == e_delivered[i] + (eff * delta_t * u[i])]

        ## Solve
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        # try:
        #     print("u:",np.round(u.value,2 ))
        # except:
        #     print(  "status",prob.status)
        return u.value, e_delivered.value

    def run_opt(self):
        start = timeit.timeit()

        def J_func(z, u, v):
            ### Read parameters
            N_asap = self.Problem.N_asap
            TOU = self.Problem.TOU
            station_pow_max = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            soft_v_eta = self.Parameters.soft_v_eta

            # reg_flex =  np.linalg.norm(u,2) * lam_x + z[0]**2 * lam_z_c

            f_flex = u.T @ (TOU - z[0]) * delta_t
            # g_flex = lam_h_c * 1 / z[2]

            J_1 = v[0] * (f_flex)

            # ASAP Charging
            # reg_asap =  z[1]**2 * lam_z_uc
            # f_asap = np.sum(station_pow_max * (TOU[:N_asap] - z[1])) * delta_t
            N_asap_remainder = self.Problem.N_asap_remainder

            if N_asap_remainder > 0:
                f_asap = (np.sum(station_pow_max * (TOU[:N_asap - 1] - z[1])) + (station_pow_max * N_asap_remainder) * (
                            TOU[N_asap - 1] - z[1])) * delta_t
            else:
                f_asap = np.sum(station_pow_max * (TOU[:N_asap] - z[1])) * delta_t

                # g_asap = lam_h_uc * 1 / z[2]
            J_2 = v[1] * (f_asap)

            # Leave
            # Include the p_max
            # J_3 = v[2] * np.sum(TOU[:N_asap]) * station_pow_max * delta_t
            J_3 = 0

            return np.array([J_1, J_2, J_3])

        def charging_revenue(z, u):

            N_asap = self.Problem.N_asap
            TOU = self.Problem.TOU
            station_pow_max = self.Problem.power_rate

            delta_t = self.Parameters.Ts

            f_flex = u.T @ (z[0] - TOU) * delta_t
            ## u : kW , z: cents / kWh, TOU : cents / kWh , delta_t : 1 \ h
            # f_asap = np.sum(station_pow_max * (z[1] - TOU[:N_asap])) * delta_t

            N_asap_remainder = self.Problem.N_asap_remainder

            if N_asap_remainder > 0:
                f_asap = (np.sum(station_pow_max * (TOU[:N_asap - 1] - z[1])) + (station_pow_max * N_asap_remainder) * (
                            TOU[N_asap - 1] - z[1])) * delta_t

            else:
                f_asap = np.sum(station_pow_max * (TOU[:N_asap] - z[1])) * delta_t

            return f_flex, f_asap

        itermax = 1000
        count = 0
        improve = np.inf

        # [z_c, z_uc, y, 1];
        # xk = np.ones((2 * self.Problem.N_flex + 1, 1)) # [soc0, ..., socN, u0, ..., uNm1]; - multiple dimensions 1 +  # of FLEX

        zk = self.Parameters.z0
        # uk_flex = self.Problem.station_pow_max * np.ones((self.Problem.N_flex))
        uk_flex = np.zeros((self.Problem.N_flex))
        vk = self.Parameters.v0  # [sm_c, sm_uc, sm_y]

        ###     THIS VALUES ARE STORED FOR DEBUGGING     ##

        Jk = np.zeros((itermax))
        rev_flex = np.zeros((itermax))
        rev_asap = np.zeros((itermax))
        z_iter = np.zeros((4, itermax))
        v_iter = np.zeros((3, itermax))
        J_sub = np.zeros((3, itermax))

        # print(J_func(zk, uk_flex, vk))

        while (count < itermax) & (improve >= 0) & (abs(improve) >= 0.00001):
            Jk[count] = J_func(zk, uk_flex, vk).sum()
            J_sub[:, count] = J_func(zk, uk_flex, vk).reshape(3, )
            rev_flex[count], rev_asap[count] = charging_revenue(zk, uk_flex)
            z_iter[:, count] = zk.reshape((4,))
            v_iter[:, count] = vk.reshape((3,))

            uk_flex, e_deliveredk_flex = self.argmin_x(zk, vk)

            vk = self.argmin_v(uk_flex, zk)
            zk = self.argmin_z(uk_flex, vk)

            # compute residual
            # print(Jk[count])

            improve = Jk[count] - J_func(zk, uk_flex, vk).sum()
            # print(J_func(zk, uk_flex, vk))
            count += 1

        print("After %d iterations," % count, "we got %f " % improve, "improvements, and claim convergence.")
        # print("The prices are %f" %zk[0], "%f" %zk[1])
        print("The prices for flex : %f" % zk[0], ",  asap : %f" % zk[1])
        opt = {}
        opt['e_need'] = self.Problem.e_need
        opt["z"] = zk
        opt["z_hr"] = zk * self.Problem.power_rate
        opt["tariff_flex"] = zk[0]
        opt["tariff_asap"] = zk[1]
        opt["tariff_overstay"] = zk[2]
        # opt["x"] = xk
        # update demand charge
        opt["peak_pow"] = max(uk_flex)
        opt["flex_e_delivered"] = e_deliveredk_flex

        # In outer simulator, it chooses the "flex" or "asap" as powers.
        opt["flex_powers"] = uk_flex
        self.flex_powers = uk_flex

        asap_powers = np.ones((self.Problem.N_asap, 1)) * self.Problem.station_pow_max
        if self.Problem.N_asap_remainder != 0:
            asap_powers[self.Problem.N_asap - 1] = self.Problem.station_pow_max * self.Problem.N_asap_remainder
        opt["asap_powers"] = asap_powers
        self.asap_powers = asap_powers
        self.Problem.powers = uk_flex
        opt["v"] = vk
        opt["prob_flex"] = vk[0]
        opt["prob_asap"] = vk[1]
        opt["prob_leave"] = vk[2]

        opt["J"] = Jk[:count]
        opt["J_sub"] = J_sub[:, :count]
        opt["z_iter"] = z_iter[:, :count]
        opt["v_iter"] = v_iter[:, :count]
        opt["rev_flex"] = rev_flex[:count]
        opt["rev_asap"] = rev_asap[:count]
        opt["power_sum"] = uk_flex.reshape(-1, 1)
        opt["N_flex"] = self.Problem.N_flex
        opt["N_asap"] = self.Problem.N_asap
        opt["N_flex"] = self.Problem.N_flex
        opt["N_asap"] = self.Problem.N_asap


        opt["num_iter"] = count
        opt["prb"] = self.Problem
        opt["par"] = self.Parameters
        opt["power_sum"] = uk_flex.reshape(-1, 1)
        opt["time_start"] = self.Problem.user_time
        opt["time_end_flex"] = self.Problem.user_time + self.Problem.user_duration
        opt["time_end_asap"] = self.Problem.user_time + self.Problem.N_asap
        end = timeit.timeit()
        return opt
def save_look_up_tables(output_folder_name, reg_price, sch_price, sch_power):
    """ Outputs 6 files """
    
    # check if outputs folder exists 
    MYDIR = ("Outputs")
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    
    MYDIR = ("Outputs/Recent")
    CHECK_FOLDER = os.path.isdir(MYDIR)

    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)

        reg_price.to_excel( os.path.join("Outputs/Recent","Regular_Prices.xlsx") )
        sch_price.to_excel( os.path.join("Outputs/Recent","Scheduled_Prices.xlsx") )

    # check if outputs folder exists 
    MYDIR = (os.path.join("Outputs",output_folder_name))
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
        
        reg_price.to_excel( os.path.join(MYDIR,"Regular_Prices.xlsx") )
        sch_price.to_excel( os.path.join(MYDIR,"Scheduled_Prices.xlsx") )
        sch_power.to_excel( os.path.join(MYDIR,"Scheduled_Powers.xlsx") )

    elif CHECK_FOLDER:
        print("already exists: ", MYDIR)
        reg_price.to_excel( os.path.join(MYDIR,"Regular_Prices_1.xlsx") )
        sch_price.to_excel( os.path.join(MYDIR,"Scheduled_Prices_1.xlsx") )
        sch_power.to_excel( os.path.join(MYDIR,"Scheduled_Powers_1.xlsx") )


