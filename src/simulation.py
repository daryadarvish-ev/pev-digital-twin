import seaborn as sns
import simpy
from optimizer_station import *
from station_pole_occupancy import *  # function for checking pole occupancy

# class driver should include -----------------------------
from driver import *  # |
from simulation_choice_function import *  # |
from user_info import *  # |
from choice_append import *  # |
# ---------------------------------------------------------

# class state_info should include -------------------------
from state_info import *  # |
from simulation_logger import SimulationLogger  # |
# from check_pole import *                               # |
from simulation_data_analysis import *  # |
from check_users_at_station import *  # |

# ---------------------------------------------------------

warnings.filterwarnings('ignore')
sns.set_theme()


class Simulator:
    """
    This class hold the simulation ...
    """

    def __init__(self, daily_sessions, total_day, input_df, input_real_df, number_of_pole):

        # SIMULATION PARAMETERS
        self.input_df = input_df
        self.input_real_df = input_real_df
        self.DAILY_SESSIONS = daily_sessions
        self.NUM_DAYS = total_day  # Arbitrarily changed this
        self.SIM_RUN_TIME = 1440 * self.NUM_DAYS
        self.CAR_ARR_TIME = 120
        self.CAR_STAY_TIME = 300
        self.FLEX_user = list()
        self.ASAP_user = list()
        self.LEAVE_user = list()
        self.station = {'FLEX_list': self.FLEX_user, 'ASAP_list': self.ASAP_user, 'LEAVE_list': self.LEAVE_user}
        self.user_choice = {}
        self.user_choice_price = {}
        self.user_arrival_time = {}
        self.user_departure_time = {}
        self.e_needed = {}
        self.number_of_pole = number_of_pole
        self.pole_occupancy = {}
        self.log = SimulationLogger()
        self.historical_data_sample_start_day = 1324

    def run_simulation(self, env, input_df, t_end):

        occupied_pole_num = []
        MM = range(t_end)
        MM = MM[::1]
        n = self.number_of_pole
        pole_list = [str(i) for i in range(1, n + 1)]

        poles = pd.DataFrame(columns=pole_list, index=MM)
        delta_t = 1
        # Initialize the previous day variable to the starting day
        day = -1
        prev_day = 0

        ev = Driver(data_path='../data/Sessions2_20221020.csv',
                    daily_sessions=self.DAILY_SESSIONS,
                    total_day=self.NUM_DAYS,
                    delta_t=delta_t, )
        # anlz = data_analyze(self.user_choice,
        #                     self.user_choice_price,
        #                     self.user_arrival_time,
        #                     self.user_departure_time)
        anlz = Station.data_analyze(self.user_choice,
                                    self.user_choice_price,
                                    self.user_arrival_time,
                                    self.user_departure_time)
        usr_behv_model, asap_quant, flex_quant = anlz.usr_behavior_clf(data_path='../data/Sessions2_20221020.csv',
                                               analysis_start=self.historical_data_sample_start_day)
        user_choice = ev.UserBehavior(usr_behv_model, asap_quant, flex_quant)
        # Start the simulation loop
        while True:

            time = int(env.now)
            print('this is time', time)
            yield env.timeout(delta_t)  # time increment is set as 1 min

            if (time % (24 * 60)) == 0:  # when a day is over, then count the day
                day += 1
                if day != prev_day:  # if the current day is different from the previous day
                    self.station = {'FLEX_list': self.FLEX_user, 'ASAP_list': self.ASAP_user,
                                    'LEAVE_list': self.LEAVE_user}  # add the user lists to the station dictionary
                    print("It's a new day!")
                    self.FLEX_user = []  # reset the user lists
                    self.ASAP_user = []
                    self.LEAVE_user = []
                    prev_day = day  # update the previous day variable to the current day

            # print('this is station from the start ', self.station)

            if time > t_end:  # if the time has passed the SIM_RUN_TIME, then break the loop

                data_analysis = data_analyze(self.user_choice, self.user_choice_price, self.user_arrival_time,
                                             self.user_departure_time)
                data_analysis.analysis()
                data_analysis.plot_generation()

                print(self.input_df)
                break

            # check -> new users and current users at the time
            new_user, current_users = CheckUsersAtStation(self.input_df, day,
                                                          time)  # at each time step, check if there is any user in the charging station
            print('new_user : ', new_user)
            print('current_users : ', current_users)

            ############################# if there is new user & no current user ##############################
            if new_user and not current_users:

                # current user list is not empty
                # user_value = int(new_user[0]) # current user
                for user_value in new_user:

                    # print(' this is the user value: ', user_value)
                    user = ev.user_info(self.input_df, time, user_value,
                                        delta_t)  # retrieve user information necessary for optimization

                    # always make the first one smaller than
                    par = Parameters(z0=np.array([20, 25, 1, 1]).reshape(4, 1),
                                     v0=np.array([0.3333, 0.3333, 0.3333]).reshape(3, 1),
                                     Ts=0.25,
                                     base_tarriff_overstay=1.0,
                                     eff=1.0,
                                     soft_v_eta=1e-4,
                                     opt_eps=0.0001,
                                     TOU=np.ones((96,)))

                    event = user.retrieve_user_information()

                    print(event)
                    # info = user_info(input_df, time, user, delta_t)

                    # get arrivalMinGlobal and departureMinGlobal
                    arrival_min_global = event['arrivalMinGlobal']
                    arrival_hour = event['arrival_hour']

                    # arrival_hour = info.input_df['arrivalHour'][info.user - 1]
                    # print('arrival_global', arrival_min_global)
                    # print('arrival_local', arrival_hour)
                    departure_min_global = event['departureMinGlobal']
                    # print('departure', departure_min_global)

                    prb = Problem(par=par, event=event)
                    opt = Optimization_charger(par, prb)
                    res = opt.run_opt()  # returns the current resolution from the optimizer

                    # get flex and asap tarrif
                    flex_price = res['tariff_flex']
                    asap_price = res['tariff_asap']

                    # Driver choice based on the tariff
                    choice, price = user_choice.basic_choice_function(asap_price, flex_price)
                    print('This is choice:', choice)

                    res['choice'] = choice

                    self.user_choice_price['EV' + str(user_value)] = price
                    print('This is the user_price dictionary: ', self.user_choice_price)

                    self.user_choice['EV' + str(user_value)] = choice
                    print('This is the user_choice dictionary: ', self.user_choice)

                    ev.append_choice(choice, user_value, current_users, self.user_choice, price, self.station, opt)

                    print('station', self.station)

                    arrival_min_global = self.input_real_df['arrivalMinGlobal'][user_value - 1]
                    self.user_arrival_time['EV' + str(user_value)] = arrival_min_global
                    print('This is the user_arrival dictionary: ', self.user_arrival_time)

                    departure_min_global = self.input_real_df['departureMinGlobal'][user_value - 1]
                    self.user_departure_time['EV' + str(user_value)] = departure_min_global
                    print('This is the user_departure dictionary: ', self.user_departure_time)

                    pole_dict, occupied_pole_num = check_pole(arrival_min_global, departure_min_global, t_end, poles,
                                                              occupied_pole_num, pole_number=self.number_of_pole)

                    self.pole_occupancy['EV' + str(user_value)] = occupied_pole_num[user_value - 1]
                    print('what is the pole occupancy dict', self.pole_occupancy)

                    # store historical values to the log info (energy needed, tariffs, power(soc))
                    self.log.add_data(time, res)

                    for key in self.pole_occupancy:
                        if self.pole_occupancy[key] == 'unavailability leave':
                            self.user_choice[key] = 'unavailability leave'
                            self.log.user_data.at[int(key.split('EV')[1]), 'choice'] = 'Leave'
                    print('This is the user_choice dictionary: ', self.user_choice)

                    e_needed = self.input_df['cumEnergy_kWh'][user_value - 1]
                    self.e_needed['EV' + str(user_value)] = e_needed
                    print('This is the e_needed dictionary: ', self.e_needed)

                    total_charging_revenue = anlz.total_revenue_calculate(self.user_choice_price,
                                                                          self.user_arrival_time,
                                                                          self.user_departure_time,
                                                                          self.e_needed,
                                                                          self.user_choice)
                    print('This is the total charging revenue: ', total_charging_revenue)

            ############################# if there is new user & current user ##############################
            elif new_user and current_users:
                # current user list is not empty

                for user_value in new_user:

                    # user_value # current user
                    print('This is the user value: ', user_value)
                    user = ev.user_info(self.input_df, time, user_value,
                                        delta_t)  # retrieve user information necessary for optimization

                    par = Parameters(z0=np.array([20, 20, 1, 1]).reshape(4, 1),
                                     # due to local optimality, we can try multiple starts
                                     eff=1.0,
                                     soft_v_eta=1e-4,
                                     opt_eps=0.0001,
                                     )

                    event = user.retrieve_user_information()

                    # get arrivalMinGlobal and departureMinGlobal
                    arrival_min_global = event['arrivalMinGlobal']
                    departure_min_global = event['departureMinGlobal']

                    prb = Problem(par=par, event=event)
                    arrival_hour = event['arrival_hour']
                    # print('this is what #####', self.station)
                    opt = Optimization_station(par, prb, self.station, arrival_hour)
                    res = opt.run_opt()

                    # get flex and asap tarrif
                    flex_price = res['tariff_flex']
                    asap_price = res['tariff_asap']

                    # Driver choice based on the tariff
                    choice, price = user_choice.basic_choice_function(asap_price, flex_price)
                    print('This is choice:', choice)

                    res['choice'] = choice

                    self.user_choice_price['EV' + str(user_value)] = price
                    print('This is the user_pirce dictionary: ', self.user_choice_price)

                    self.user_choice['EV' + str(user_value)] = (choice)
                    print('This is the user_choice dictionary: ', self.user_choice)

                    arrival_min_global = self.input_real_df['arrivalMinGlobal'][user_value - 1]
                    self.user_arrival_time['EV' + str(user_value)] = arrival_min_global
                    print('This is the user_arrival dictionary: ', self.user_arrival_time)

                    departure_min_global = self.input_real_df['departureMinGlobal'][user_value - 1]
                    self.user_departure_time['EV' + str(user_value)] = departure_min_global
                    print('This is the user_departure dictionary: ', self.user_departure_time)

                    pole_dict, occupied_pole_num = check_pole(arrival_min_global, departure_min_global, t_end, poles,
                                                              occupied_pole_num, pole_number=self.number_of_pole)

                    self.pole_occupancy['EV' + str(user_value)] = occupied_pole_num[user_value - 1]
                    print('what is the pole occupancy dict', self.pole_occupancy)

                    # store historical values to the log info (energy needed, tariffs, power(soc))
                    self.log.add_data(time, res)

                    for key in self.pole_occupancy:
                        if self.pole_occupancy[key] == 'unavailability leave':
                            self.user_choice[key] = 'unavailability leave'
                            self.log.user_data.at[int(key.split('EV')[1]), 'choice'] = 'Leave'
                    print('This is the user_choice dictionary: ', self.user_choice)


                    # Print the energy needed of users
                    e_needed = self.input_df['cumEnergy_kWh'][user_value - 1]
                    self.e_needed['EV' + str(user_value)] = e_needed
                    print('This is the e_needed dictionary: ', self.e_needed)

                    ev.append_choice(choice, user_value, current_users, self.user_choice, price, self.station, opt)

                    print('station', self.station)

                    # Calculate total charging revenue
                    total_charging_revenue = anlz.total_revenue_calculate(self.user_choice_price,
                                                                          self.user_arrival_time,
                                                                          self.user_departure_time,
                                                                          self.e_needed,
                                                                          self.user_choice)
                    print('This is the total charging revenue: ', total_charging_revenue)

                    print('#### end ####')

            ############################# if there is NO new user ##############################
            else:
                # print('station', self.station)
                print('#### nothing happened ####')
                pass
