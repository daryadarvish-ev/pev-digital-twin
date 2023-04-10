import matplotlib.pyplot as plt
import seaborn as sns
import simpy
from matplotlib.pyplot import figure

from optimizer_station import *
from Check_if_any_user_in_charging_station import *
from User_info import *
from run_optimization import *
from history import HistoricalData
from check_pole import *
from choice_append import *
from data_analysis import *

warnings.filterwarnings('ignore')
sns.set_theme()

class Simulator:
    """
    This class hold the simulation ...
    """
    def __init__(self, daily_sessions, total_day, input_df ,input_real_df , env):

        # SIMULATION PARAMETERS
        self.input_df = input_df
        self.input_real_df = input_real_df
        self.NUM_DAYS = total_day  # Arbitrarily changed this
        self.SIM_RUN_TIME = 1440 * self.NUM_DAYS
        self.CAR_ARR_TIME = 120
        self.CAR_STAY_TIME = 300
        self.env = simpy.Environment()
        self.FLEX_user = list()
        self.ASAP_user = list()
        self.LEAVE_user = list()
        self.station = {'FLEX_list': self.FLEX_user, 'ASAP_list': self.ASAP_user, 'LEAVE_list': self.LEAVE_user}
        self.user_choice = {}
        self.user_choice_price = {}
        self.user_arrival_time = {}
        self.user_departure_time = {}


    def run_simulation (self,env, input_df, t_end):

        delta_t = 1
        # Initialize the previous day variable to the starting day
        day = -1
        prev_day = 0

        # Start the simulation loop
        while True:

            time = int(env.now)
            print('this is time', time)
            yield env.timeout(delta_t)  # time increment is set as 1 min


            if (time % (24 * 60)) == 0:  # when a day is over, then count the day
                day += 1
                if day != prev_day:  # if the current day is different from the previous day
                    self.station = {'FLEX_list': self.FLEX_user, 'ASAP_list': self.ASAP_user,'LEAVE_list':self.LEAVE_user}  # add the user lists to the station dictionary
                    print("It's a new day!")
                    self.FLEX_user = []  # reset the user lists
                    self.ASAP_user = []
                    self.LEAVE_user = []
                    prev_day = day  # update the previous day variable to the current day

            # print('this is station from the start ', self.station)

            if time > t_end:  # if the time has passed the SIM_RUN_TIME, then break the loop

                data_analysis = data_analyze(self.user_choice, self.user_choice_price, self.user_arrival_time, self.user_departure_time)
                data_analysis.analysis()
                data_analysis.plot_generation()

                print(self.input_df)
                break

            # check -> new users and current users at the time
            new_user, current_users = Check_if_any_user_in_charging_station(self.input_df, day, time) # at each time step, check if there is any user in the charging station
            print('new_user : ', new_user)
            print('current_users : ', current_users)

            ############################# if there is new user & no current user ##############################
            if new_user and not current_users:

                # current user list is not empty
                # user_value = int(new_user[0]) # current user
                for user_value in new_user:

                    # print(' this is the user value: ', user_value)
                    user = user_info(self.input_df, time, user_value, delta_t) # retrieve user information necessary for optimization

                    # par = Parameters(z0=np.array([20, 20, 1, 1]).reshape(4, 1),
                    #                  # due to local optimality, we can try multiple starts
                    #                  eff=1.0,
                    #                  soft_v_eta=1e-4,
                    #                  opt_eps=0.0001,
                    #                  TOU=TOU_tariff
                    #                  )
                    par = Parameters(z0=np.array([20, 20, 1, 1]).reshape(4, 1),
                                     v0=np.array([0.3333, 0.3333, 0.3333]).reshape(3, 1),
                                     Ts=0.25,
                                     base_tarriff_overstay=1.0,
                                     eff=1.0,
                                     soft_v_eta=1e-4,
                                     opt_eps=0.0001,
                                     TOU = np.ones((96,)))


                    event = user.retrieve_user_information()
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
                    res = opt.run_opt()

                    dictionary = res
                    # print(dictionary)
                    # self.station['EV' + str(user_value)] = opt
                    # print('station', self.station)

                    # store historical values to the log info (energy needed, tariffs, power(soc))
                    log = HistoricalData()
                    log1 = log.add_data(time, res)

                    # print('this is log', log)
                    flex_price = log.get_tariff_flex(time)
                    asap_price = log.get_tariff_asap(time)
                    # print('flex price', flex_price)
                    # print('asap price', asap_price)

                    # Driver choice based on the tariff
                    choice, price = basic_choice_function(asap_price, flex_price)
                    print('this is choice:', choice)

                    self.user_choice_price['EV'+ str(user_value)] = price
                    print('what is this user_pirce dictionary: ', self.user_choice_price)

                    self.user_choice['EV'+ str(user_value)] = (choice)
                    print('what is this user_choice dictionary: ', self.user_choice)

                    append_choice(choice, user_value, current_users, self.user_choice, price, self.station , opt)

                    print('station', self.station)

                    arrival_min_global = self.input_real_df['arrivalMinGlobal'][user_value - 1]
                    self.user_arrival_time['EV' + str(user_value)] = arrival_min_global

                    print('what is this user_arrival dictionary: ', self.user_arrival_time)

                    departure_min_global = self.input_real_df['departureMinGlobal'][user_value - 1]
                    self.user_departure_time['EV' + str(user_value)] = departure_min_global

                    print('what is this user_departure dictionary: ', self.user_departure_time)




                #Charging station occupancy
                # cs = ChargingStation(num_poles=8)
                # cs.charge_car(user=user_value, arrival_time= arrival_min_global, departure_time= departure_min_global, user_choice = choice)

                # # Check which poles are available at time:
                # available_poles, occupied_poles = cs.get_available_poles(time)
                # print(f"Available poles: {available_poles}")
                # print(f"Occupied poles: {occupied_poles}")
                # print('#### end ####')

            ############################# if there is new user & current user ##############################
            elif new_user and current_users:
                # current user list is not empty

                for user_value in new_user:

                    # user_value # current user
                    print(' this is the user value: ', user_value)
                    user = user_info(self.input_df, time, user_value,
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
                    # print('this is what2 #####', self.station)


                    dictionary = res[1]
                    # print(dictionary)
                    # print(dictionary['tariff_asap'], dictionary['tariff_flex'])
                    # print(type(res[1]))
                    asap_price, flex_price = (dictionary['tariff_asap'], dictionary['tariff_flex'])
                    # flex_price = log1.get_tariff_flex(time)
                    # asap_price = log1.get_tariff_asap(time)

                    # Driver choice based on the tariff
                    choice, price = basic_choice_function(asap_price, flex_price)
                    print('this is choice:', choice)

                    # print('flex price', flex_price)
                    # print('asap price', asap_price)

                    self.user_choice_price['EV' + str(user_value)] = price
                    print('what is this user_pirce dictionary: ', self.user_choice_price)

                    self.user_choice['EV' + str(user_value)] = (choice)
                    print('what is this user_choice dictionary: ', self.user_choice)

                    arrival_min_global = self.input_real_df['arrivalMinGlobal'][user_value - 1]
                    self.user_arrival_time['EV' + str(user_value)] = arrival_min_global

                    print('what is this user_arrival dictionary: ', self.user_arrival_time)

                    departure_min_global = self.input_real_df['departureMinGlobal'][user_value - 1]
                    self.user_departure_time['EV' + str(user_value)] = departure_min_global

                    print('what is this user_departure dictionary: ', self.user_departure_time)


                    # print('this is the self.station when there is current user', self.station)
                    append_choice(choice, user_value, current_users, self.user_choice, price, self.station, opt)

                    print('station', self.station)


                    # Charging station occupancy
                    # cs = ChargingStation(num_poles=8)
                    # cs.charge_car(user=user_value, arrival_time=arrival_min_global, departure_time=departure_min_global,
                    #               user_choice=choice)

                    # # Check which poles are available at time:
                    # available_poles, occupied_poles = cs.get_available_poles(time)
                    # print(f"Available poles: {available_poles}")
                    # print(f"Occupied poles: {occupied_poles}")


                    print('#### end ####')


            ############################# if there is NO new user ##############################
            else:
                # print('station', self.station)
                print('#### nothing happened ####')
                pass



