import math
import timeit
import warnings

import cvxpy as cp
import numpy as np

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
        self.station_pow_max = event["station_pow_max"] # Power cap for the station charger
        self.user_power_rate = event['user_power_rate']

        self.power_rate = min(self.user_power_rate, self.station_pow_max) # The actual power cap for user

        self.dcm_charging_sch_params = np.array([[ - self.power_rate * 0.0184 / 2], [ self.power_rate * 0.0184 / 2], [0], [0]])
        #% DCM parameters for choice 1 -- charging with flexibility
        self.dcm_charging_reg_params = np.array([[self.power_rate * 0.0184 / 2], [- self.power_rate * 0.0184 / 2], [0],[0.341]])
        #% DCM parameters for choice 2 -- charging as soon as possible
        self.dcm_leaving_params = np.array([[self.power_rate * 0.005 / 2], [self.power_rate * 0.005 / 2], [0], [-1 ]])
        
        #% DCM parameters for choice 3 -- leaving without charging
        self.THETA = np.vstack((self.dcm_charging_sch_params.T, self.dcm_charging_reg_params.T,
                     self.dcm_leaving_params.T))
        # problem specifications
        self.N_sch = self.user_duration # charging duration that is not charged, hour
        
        ### IS THIS CORRECT? WHATS SOC NEED REPRESENTS? 
        # self.N_reg = math.floor((self.user_SOC_need - self.userx_SOC_init) *
        #                          self.user_batt_cap / self.station_pow_max / par.eff / par.Ts)

        ## HERE 12 IS SELF CODED 
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
            N_reg = self.Problem.N_reg
            TOU = self.Problem.TOU
            power_rate = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            N_sch = self.Problem.N_sch

            ### Calculate the existing users' objective function values(ASAP/FLEX)

            # Existing FLEX user term. Equation (24) - (27) in Teng's Paper.
            user_keys = self.station['FLEX_list']
            existing_sch_obj = 0
            if user_keys:
                # For every adj_constant, it includes the charging profile of one user for N_max interval
                # (max remaining intervals) for all existing and the new user. The length of adj_constant is (N_max).
                for i in range(1, self.existing_user_info.shape[0]): # EVs other than the new user
                    adj_constant = int(i * self.var_dim_constant)
                    # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                    N_remain = int(self.existing_user_info[i, 2]) # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                    TOU_idx = int(self.existing_user_info[i, 3]) # TOU_index
                    user = self.station[user_keys[i - 1]] # Here we need "i - 1", since the first row of existing_user_info is a new user
                    existing_sch_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1) # No problem with indices

            # Exising ASAP user cost term. Equ (29) - (32).
            # The goal is to calculate the remaining energy demand of the existing ASAP users.

            existing_reg_obj = 0
            user_keys = self.station['ASAP_list']
            if user_keys:
                users = [self.station[key] for key in user_keys]
                TOU_idx = np.int_(self.k / delta_t - np.array([user.Problem.user_time for user in users]))
                existing_reg_obj = np.sum([user.reg_powers[TOU_idx[i]:].T @ (
                            user.Problem.TOU[TOU_idx[i]: user.Problem.N_reg] - user.price).reshape(-1, 1) for i, user
                                            in enumerate(users)])

            # Calculate the existing users' load summation.(ASAP/FLEX)
            reg_power_sum_profile = np.zeros(self.var_dim_constant)
            for i in range(len(self.station['ASAP_list'])):
                user = self.station[user_keys[i]]
                TOU_idx = int(self.k / delta_t - user.Problem.user_time)
                reg_power_sum_profile[: user.Problem.N_reg - TOU_idx] += user.reg_powers[TOU_idx:].squeeze()

            num_flex = self.existing_user_info.shape[0]
            sch_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_flex)).T # Row: # of user, Col: Charging Profile
            sch_power_sum_profile = cp.sum(sch_power_sum_profile[1:, :], axis = 0)

            # Calculate the charging profile of the new user (FLEX: derive from 'u', ASAP: directly calculate.

            N_reg_remainder = self.Problem.N_reg_remainder

            reg_new_user_profile = np.zeros(self.var_dim_constant)
            reg_new_user_profile[: N_reg - 1] = power_rate
            reg_new_user_profile[N_reg - 1] = (power_rate * N_reg_remainder) if N_reg_remainder > 0 else power_rate

            # We use cvxpy here, so all variables are cvxpy variables.
            # Teng's Paper's Eq(17) - (20)
            new_sch_obj = u[: N_sch].T @ (TOU[:N_sch] - z[0]).reshape(-1, 1) * delta_t
            new_reg_obj = (cp.sum(power_rate * (TOU[:N_reg - 1] - z[1])) + (power_rate * N_reg_remainder) * (TOU[N_reg - 1] - z[1])) * delta_t if N_reg_remainder > 0 else cp.sum(power_rate * (TOU[:N_reg] - z[1])) * delta_t
            new_leave_obj = 0

            J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
            J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + sch_power_sum_profile + reg_new_user_profile)) * v[1]
            J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + sch_power_sum_profile)) * v[2]

            J = J0 + J1 + J2

            return J

        J = self.constr_J(u, z, v)

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
        N_sch: timesteps arrival to departure 
        N_reg: timesteps required when charging at full capacity

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
            N_reg = self.Problem.N_reg
            TOU = self.Problem.TOU
            power_rate = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            N_sch = self.Problem.N_sch

            user_keys = self.station['FLEX_list']
            existing_sch_obj = 0
            if user_keys:
                for i in range(1, self.existing_user_info.shape[0]): # EVs other than the new user
                    adj_constant = int(i * self.var_dim_constant)
                    # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                    N_remain = int(self.existing_user_info[i, 2]) # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                    TOU_idx = int(self.existing_user_info[i, 3]) # TOU_index
                    user = self.station[user_keys[i - 1]] # Here we need "i - 1", since the first row of existing_user_info is a new user
                    existing_sch_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1) # No problem with indices

            existing_reg_obj = 0
            user_keys = self.station['ASAP_list']
            if user_keys:
                users = [self.station[key] for key in user_keys]
                TOU_idx = np.int_(self.k / delta_t - np.array([user.Problem.user_time for user in users]))
                existing_reg_obj = np.sum([user.reg_powers[TOU_idx[i]:].T @ (
                            user.Problem.TOU[TOU_idx[i]: user.Problem.N_reg] - user.price).reshape(-1, 1) for i, user
                                            in enumerate(users)])

            # Existing user charging profile summation
            reg_power_sum_profile = np.zeros(self.var_dim_constant)
            for i in range(len(self.station['ASAP_list'])): # for all ASAP users
                user = self.station[user_keys[i]]
                TOU_idx = int(self.k / delta_t - user.Problem.user_time)
                reg_power_sum_profile[: user.Problem.N_reg - TOU_idx] += user.reg_powers[TOU_idx:].squeeze()

            num_flex = self.existing_user_info.shape[0]

            sch_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_flex)).T # Row: # of user, Col: Charging Profile
            sch_power_sum_profile = cp.sum(sch_power_sum_profile[1:, :], axis = 0)

            # New user charging profile(ASAP)

            N_reg_remainder = self.Problem.N_reg_remainder

            reg_new_user_profile = np.zeros(self.var_dim_constant)
            reg_new_user_profile[: N_reg - 1] = power_rate
            reg_new_user_profile[N_reg - 1] = (power_rate * N_reg_remainder) if N_reg_remainder > 0 else power_rate

            c_co = cp.reshape((TOU[:N_sch] - z[0]), (N_sch, 1))

            # Use cvxpy so all variables are cvxpy variables
            new_sch_obj = (u[: N_sch].T @ c_co) * delta_t
            new_reg_obj = (cp.sum(power_rate * (TOU[:N_reg - 1] - z[1])) + (power_rate * N_reg_remainder) * (TOU[N_reg - 1] - z[1])) * delta_t if N_reg_remainder > 0 else cp.sum(power_rate * (TOU[:N_reg] - z[1])) * delta_t
            new_leave_obj = 0

            J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
            J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + sch_power_sum_profile + reg_new_user_profile)) * v[1]
            J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + sch_power_sum_profile)) * v[2]

            J = J0 + J1 + J2

            return J

        J = self.constr_J(u, z, v)

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
            N_reg = self.Problem.N_reg
            TOU = self.Problem.TOU
            power_rate = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            N_sch = self.Problem.N_sch

            user_keys = self.station['FLEX_list']
            existing_sch_obj = 0
            if user_keys:
                for i in range(1, self.existing_user_info.shape[0]): # EVs other than the new user
                    adj_constant = int(i * self.var_dim_constant)
                    # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                    N_remain = int(self.existing_user_info[i, 2]) # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                    TOU_idx = int(self.existing_user_info[i, 3]) # TOU_index
                    user = self.station[user_keys[i - 1]] # Here we need "i - 1", since the first row of existing_user_info is a new user
                    existing_sch_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1) # No problem with indices

            existing_reg_obj = 0
            user_keys = self.station['ASAP_list']
            if user_keys:
                users = [self.station[key] for key in user_keys]
                TOU_idx = np.int_(self.k / delta_t - np.array([user.Problem.user_time for user in users]))
                existing_reg_obj = np.sum([user.reg_powers[TOU_idx[i]:].T @ (
                            user.Problem.TOU[TOU_idx[i]: user.Problem.N_reg] - user.price).reshape(-1, 1) for i, user
                                            in enumerate(users)])

            # Existing user charging profile summation
            reg_power_sum_profile = np.zeros(self.var_dim_constant)
            for i in range(len(self.station['ASAP_list'])): # for all ASAP users
                user = self.station[user_keys[i]]
                TOU_idx = int(self.k / delta_t - user.Problem.user_time)
                reg_power_sum_profile[: user.Problem.N_reg - TOU_idx] += user.reg_powers[TOU_idx:].squeeze()

            num_flex = self.existing_user_info.shape[0]

            sch_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_flex)).T # Row: # of user, Col: Charging Profile
            sch_power_sum_profile = cp.sum(sch_power_sum_profile[1:, :], axis = 0)

            # New user charging profile(ASAP)

            N_reg_remainder = self.Problem.N_reg_remainder

            reg_new_user_profile = np.zeros(self.var_dim_constant)
            reg_new_user_profile[: N_reg - 1] = power_rate
            reg_new_user_profile[N_reg - 1] = (power_rate * N_reg_remainder) if N_reg_remainder > 0 else power_rate

            # Use cvxpy so all variables are cvxpy variables
            new_sch_obj = u[: N_sch].T @ (TOU[:N_sch] - z[0]).reshape(-1, 1) * delta_t
            new_reg_obj = (cp.sum(power_rate * (TOU[:N_reg - 1] - z[1])) + (power_rate * N_reg_remainder) * (TOU[N_reg - 1] - z[1])) * delta_t if N_reg_remainder > 0 else cp.sum(power_rate * (TOU[:N_reg] - z[1])) * delta_t
            new_leave_obj = 0

            J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
            J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + sch_power_sum_profile + reg_new_user_profile)) * v[1]
            J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + sch_power_sum_profile)) * v[2]

            J = J0 + J1 + J2

            return J

        J = self.constr_J(u, z, v)

        ## Constraints should incorporate all existing users
        constraints = [u >= 0]
        constraints += [u <= station_pow_max]  ## *****for existing vehicles: need to modify!

        ## Influence of existing user.
        ## THe following constraints means that: for each flex user, we shall meet the total energy demand when the charging
        # is over.

        user_keys = self.station['FLEX_list']
        for i in range(self.existing_user_info.shape[0]):  # For all possible flex users
            
            N_remain = int(self.existing_user_info[i, 2])
            e_need = self.existing_user_info[i, 4]

            # Shape of e_delivered: (num * (self.var_dim_constant + 1), 1)
            # Shape of u: (num * self.var_dim_constant, 1)
            # "num" is the number of FLEX(new + existing) users

            e_start = int(i * (self.var_dim_constant + 1))
            e_end = int(i * (self.var_dim_constant + 1) + N_remain)
            e_max = int(i * (self.var_dim_constant + 1) + self.var_dim_constant)
            u_start = int(i * self.var_dim_constant)

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
        
        # try:
        #     print("u:",np.round(u.value,2 ))
        # except:
        #     print("status",prob.status)
        return u.value, e_delivered.value

    def constr_J(self, u, z, v):
        N_reg = self.Problem.N_reg
        TOU = self.Problem.TOU
        power_rate = self.Problem.power_rate
        delta_t = self.Parameters.Ts
        N_sch = self.Problem.N_sch

        user_keys = self.station['FLEX_list']
        existing_sch_obj = 0
        if user_keys:
            for i in range(1, self.existing_user_info.shape[0]):  # EVs other than the new user
                adj_constant = int(i * self.var_dim_constant)
                # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                N_remain = int(self.existing_user_info[
                                   i, 2])  # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                TOU_idx = int(self.existing_user_info[i, 3])  # TOU_index
                user = self.station[
                    user_keys[i - 1]]  # Here we need "i - 1", since the first row of existing_user_info is a new user
                existing_sch_obj += u[adj_constant: (adj_constant + N_remain)].T @ (
                            user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1)  # No problem with indices

        existing_reg_obj = 0
        user_keys = self.station['ASAP_list']
        if user_keys:
            users = [self.station[key] for key in user_keys]
            TOU_idx = np.int_(self.k / delta_t - np.array([user.Problem.user_time for user in users]))
            existing_reg_obj = np.sum([user.reg_powers[TOU_idx[i]:].T @ (
                    user.Problem.TOU[TOU_idx[i]: user.Problem.N_reg] - user.price).reshape(-1, 1) for i, user
                                       in enumerate(users)])

        # Existing user charging profile summation
        reg_power_sum_profile = np.zeros(self.var_dim_constant)
        for i in range(len(self.station['ASAP_list'])):  # for all ASAP users
            user = self.station[user_keys[i]]
            TOU_idx = int(self.k / delta_t - user.Problem.user_time)
            reg_power_sum_profile[: user.Problem.N_reg - TOU_idx] += user.reg_powers[TOU_idx:].squeeze()

        num_flex = self.existing_user_info.shape[0]

        sch_power_sum_profile = cp.reshape(u,
                                           (self.var_dim_constant, num_flex)).T  # Row: # of user, Col: Charging Profile
        sch_power_sum_profile = cp.sum(sch_power_sum_profile[1:, :], axis=0)

        # New user charging profile(ASAP)

        N_reg_remainder = self.Problem.N_reg_remainder

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

        J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(
            reg_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
        J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(
            reg_power_sum_profile + sch_power_sum_profile + reg_new_user_profile)) * v[1]
        J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(
            reg_power_sum_profile + sch_power_sum_profile)) * v[2]

        J = J0 + J1 + J2

        return J

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
        num_sch_user = len(self.station['FLEX_list'])
        sch_user_info = np.zeros([num_sch_user, 5]) # [user_key, user_time, N_remain, user_duration, e_need]
        if user_keys:
            for i in range(num_sch_user):
                user = self.station[user_keys[i]].Problem
                start_time = user.user_time
                end_time = user.user_time + user.user_duration
                N_remain = int(end_time - self.k / self.Parameters.Ts) # Number of intervals left for the existing users
                TOU_idx = int(self.k / self.Parameters.Ts - start_time) # Current local time indices for User i
                # How much power we already charged?
                e_needed_now = user.e_need - np.sum(user.powers[: TOU_idx] * self.Parameters.eff * self.Parameters.Ts)
                sch_user_info[i, :] = [start_time, end_time, N_remain, TOU_idx, e_needed_now]

        ### Existing ASAP Users(check if they are still there)
        user_keys = self.station["ASAP_list"].copy()
        if user_keys:
            for user_key in user_keys:
                user = self.station[user_key].Problem
                end_time = user.user_time + user.N_reg
                N_remain = end_time - self.k / self.Parameters.Ts
                if N_remain <= 0:
                    del(self.station[user_key])
                    self.station["ASAP_list"].remove(user_key)

        # For the ASAP users, integrate their information in reg_user_info list.
        user_keys = self.station["ASAP_list"]
        num_reg_user = len(self.station['ASAP_list'])
        reg_user_info = np.zeros([num_reg_user, 5])
        if user_keys:
            for i in range(len(user_keys)):
                user = self.station[user_keys[i]].Problem
                start_time = user.user_time
                end_time = user.user_time + user.N_reg
                N_remain = int(end_time - self.k / self.Parameters.Ts) # Number of intervals left for the existing users
                TOU_idx = int(self.k / self.Parameters.Ts - start_time) # Current local time indices for User i
                reg_user_info[i, :] = [start_time, end_time, N_remain, TOU_idx, 0]


        ### New User information, incorporate it in self.existing_user_info
        new_user = self.Problem # The struct for the incoming user
        start_time = new_user.user_time
        existing_user_info = np.array([[start_time, -1, new_user.N_sch, 0, self.Problem.e_need]]) # Actually, all existing flex user info.
        existing_user_info = np.concatenate((existing_user_info, sch_user_info), axis = 0)
        self.existing_user_info = existing_user_info
        # Concatenate, and pick the largest interval as the Power Profile length.

        var_dim_constant = int(max(np.concatenate((existing_user_info, reg_user_info), axis = 0)[:, 2])) # chosen from maximum remaining duration for all EVs
        self.var_dim_constant = var_dim_constant

        # Initial values for uk
        uk_flex = self.Problem.power_rate * np.zeros([var_dim_constant * (num_sch_user + 1), 1]) # We are optimizing the FLEX profile, so the dimension is all possible flex users * dimension_con

        def J_func(z, u, v):
            # See the detailed comments of all J_func funcitons in self.argmin_u() (All of them are nearly identical)
            N_reg = self.Problem.N_reg
            TOU = self.Problem.TOU
            power_rate = self.Problem.power_rate
            delta_t = self.Parameters.Ts
            N_sch = self.Problem.N_sch

            user_keys = self.station['FLEX_list']
            existing_sch_obj = 0
            if user_keys:
                for i in range(1, self.existing_user_info.shape[0]): # EVs other than the new user
                    adj_constant = int(i * self.var_dim_constant)
                    # Every round of optimization we will update the "N_remain", i.e., here the duration is the time left
                    N_remain = int(self.existing_user_info[i, 2]) # N_remain, we do not modify any duration / number of intervals, however, we should modify all indices
                    TOU_idx = int(self.existing_user_info[i, 3]) # TOU_index
                    user = self.station[user_keys[i - 1]] # Here we need "i - 1", since the first row of existing_user_info is a new user
                    existing_sch_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user.Problem.TOU[TOU_idx:] - user.price).reshape(-1, 1) # No problem with indices

            user_keys = self.station['ASAP_list']
            existing_reg_obj = 0
            if user_keys:
                for i in range(len(user_keys)):
                    user = self.station[user_keys[i]]
                    TOU_idx = int(self.k / delta_t - user.Problem.user_time)  # The local indice for the duration(len(TOU) is duration)
                    existing_reg_obj += user.reg_powers[TOU_idx:].T @ (user.Problem.TOU[TOU_idx: user.Problem.N_reg] - user.price).reshape(-1, 1)

            # Existing user charging profile summation
            reg_power_sum_profile = np.zeros(self.var_dim_constant)
            for i in range(len(self.station['ASAP_list'])): # for all ASAP users
                user = self.station[user_keys[i]]
                TOU_idx = int(self.k / delta_t - user.Problem.user_time)
                reg_power_sum_profile[: user.Problem.N_reg - TOU_idx] += user.reg_powers[TOU_idx:].squeeze()

            num_flex = self.existing_user_info.shape[0]

            sch_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_flex)).T # Row: # of user, Col: Charging Profile
            sch_power_sum_profile = cp.sum(sch_power_sum_profile[1:, :], axis = 0)

            # New user charging profile(ASAP)

            N_reg_remainder = self.Problem.N_reg_remainder

            reg_new_user_profile = np.zeros(self.var_dim_constant)
            reg_new_user_profile[: N_reg - 1] = power_rate
            reg_new_user_profile[N_reg - 1] = (power_rate * N_reg_remainder) if N_reg_remainder > 0 else power_rate

            # Use cvxpy so all variables are cvxpy variables
            new_sch_obj = u[: N_sch].T @ (TOU[:N_sch] - z[0]).reshape(-1, 1) * delta_t
            new_reg_obj = (cp.sum(power_rate * (TOU[:N_reg - 1] - z[1])) + (power_rate * N_reg_remainder) * (TOU[N_reg - 1] - z[1])) * delta_t if N_reg_remainder > 0 else cp.sum(power_rate * (TOU[:N_reg] - z[1])) * delta_t
            new_leave_obj = 0

            J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_flex)).T, axis=0))) * v[0]
            J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + sch_power_sum_profile + reg_new_user_profile)) * v[1]
            J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * cp.max(reg_power_sum_profile + sch_power_sum_profile)) * v[2]

            return np.array([J0.value, J1.value, J2.value])
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

            uk_flex, e_deliveredk_flex = self.argmin_u(zk, vk)

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

            if count >= 50:
                print("Too much time for iteration(iteration times exceed 50)")
                break

        # # Iteration finished.
        # if zk[0] >= 30:
        #     zk = z_iter[:, 1]
        #     vk = v_iter[:, 1]

        print("After %d iterations," % count, "we got %f " % improve, "improvements, and claim convergence.")
        print("The prices are %f" %zk[0], "%f" %zk[1])


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

        ### Output the results
        opt = {}
        opt['e_need'] = self.Problem.e_need

        # Part 1: Prices
        opt["z"] = zk
        opt["z_hr"] = zk * self.Problem.power_rate
        # Add a self.price = chosen mode price in outer loop by calling all z values in optimizer
        opt["tariff_sch"] = zk[0]
        opt["tariff_reg"] = zk[1]
        opt["sch_centsPerHr"] = opt["z_hr"][0]
        opt["reg_centsPerHr"] = opt["z_hr"][1]

        # Part 2: Power Profiles
        opt["peak_pow"] = max(uk_flex)
        opt["sch_e_delivered"] = e_deliveredk_flex
        N_remain = int(self.Problem.user_duration)
        opt["sch_powers"] = uk_flex[: N_remain]
        self.Problem.powers = uk_flex[: N_remain]

        # For a possible "NEW" "ASAP" user, we assume that it's at the maximum for all ASAP intervals
        reg_powers = np.ones((self.Problem.N_reg, 1)) * self.Problem.power_rate
        if self.Problem.N_reg_remainder != 0: # For the last time slot, ASAP may not occupy the whole slot.
            reg_powers[self.Problem.N_reg - 1] = self.Problem.power_rate * self.Problem.N_reg_remainder
        opt["reg_powers"] = reg_powers
        self.reg_powers = reg_powers
        self.sch_powers = opt["sch_powers"]

        if self.Problem.assertion_flag == 1:
            opt["sch_powers"] = reg_powers

        # Part 3: Probability & Iteration Parameters

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
        opt["rev_flex"] = rev_flex[:count]
        opt["rev_asap"] = rev_asap[:count]
        opt["num_iter"] = count
        opt["time_start"] = self.Problem.user_time
        opt["time_end_flex"] = self.Problem.user_time + self.Problem.user_duration
        opt["time_end_asap"] = self.Problem.user_time + self.Problem.N_reg

        # Part 4: General Problem Space
        opt["prb"] = self.Problem
        opt["par"] = self.Parameters

        end = timeit.timeit()

        # Part 5: Station information(struct / class / dict?)
        station = self.station # We update the station struct every round.

        return station, opt