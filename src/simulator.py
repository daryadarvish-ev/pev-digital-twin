import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import simpy
from matplotlib.pyplot import figure

from optimizer_station import *
from session_generator import *
from simulation_choice_function import choice_function

warnings.filterwarnings('ignore')
sns.set_theme()


class Simulator:
    """
    This class hold the simulation ...
    """

    def __init__(self):

        # SIMULATION PARAMETERS
        self.NUM_DAYS = 10  # Arbitrarily changed this
        self.SIM_RUN_TIME = 1440 * self.NUM_DAYS
        self.CAR_ARR_TIME = 120
        self.CAR_STAY_TIME = 300

        self.sorted_events_q = []

        # session output
        self.session = []
        # choice output
        self.user_choice = []
        # arrival time output
        self.arrival_time = []
        # departure time output
        self.departure_time = []
        # cartype output
        self.car_type = []
        # energy requested output
        self.energyreq = []
        # flex rate requested output
        self.rate_flex = []
        # asap rate requested output
        self.rate_asap = []

        self.station_in = dict()

        self.total_revenue = []
        self.total_cost = []
        self.total_dc = []

        self.e_need = []

        self.z_flex = []
        self.z_asap = []
        self.z_leave = []

        self.flex_tarrif = []
        self.asap_tarrif = []
        self.leave_tarrif = []

        self.v_flex = []
        self.v_asap = []
        self.v_leave = []

        self.prob_flex = []
        self.prob_asap = []
        self.prob_leave = []

        self.j_flex = []
        self.j_asap = []
        self.j_leave = []
        self.occupied_pole_num = []
        self.occupied_temp = 0
        self.arrival_day = []

        self.temp_asap = []
        self.temp_scheduled = []
        self.temp_leave = []
        self.temp_price = []
        self.choice = []
        self.models = ['Prius Prime', 'Model 3', 'Bolt EV', 'Kona', 'Model X 100 Dual', 'Clarity',
                       'Volt (second generation)',
                       'B-Class Electric Drive', 'Model S 100 Dual', 'Mustang Mach-E', 'Model S 90 Dual',
                       'Bolt EUV 2LT']

        self.station_df = pd.DataFrame()

        # poles
        self.MM = range(self.SIM_RUN_TIME)
        self.MM = self.MM[::15]
        self.YY = ['1', '2', '3', '4', '5', '6', '7', '8']

        self.poles = pd.DataFrame(columns=['1', '2', '3', '4', '5', '6', '7', '8'], index=self.MM)
        self.env = simpy.Environment()
        self.daily_sessions = [30] * 10
        self.session_generator = SessionGen(daily_sessions=self.daily_sessions, data_file='../data/Sessions2.csv',
                                            rnd_seeds=(4, 5, 30))
        self.input_df = self.session_generator.generate_session()
        self.env.process(self.first_process(self.env, self.input_df, self.SIM_RUN_TIME))

    # Todo rename this function
    def first_process(self, env, input_df, run_length):
        yield env.process(self._charger_station(env, input_df, run_length))

    def run_simulation(self):
        print("sim_run_tim", self.SIM_RUN_TIME)
        print("env", self.env)
        intervals = range(self.SIM_RUN_TIME)
        intervals = intervals[::15]
        print(intervals)
        self.env.run(self.SIM_RUN_TIME + 10)

    def _charger_station(self, env, input_df, run_time):

        user = 0

        next_leave_time = -1
        events_q = []

        # for plotting outputs
        leave_time_datalog = []
        num_flex = 0
        num_asap = 0
        num_leave_imm = 0
        num_leave_occ = 0
        ################################ building station dictionary ####
        station = {}
        station['FLEX_list'] = list()
        station['ASAP_list'] = list()
        station['Leave'] = list()
        station['Day'] = list()
        ###############################
        # Time until first user arrives:
        yield env.timeout(input_df['arrivalMinGlobal'][0])
        # print('first step', str(env.now()))

        while True:

            # Car arrives
            user += 1

            if user > input_df.shape[0]:
                break
            print(user)
            # cartype output
            car = random.choices(self.models, weights=(151, 110, 86, 51, 42, 42, 28, 24, 20, 19, 15, 14))
            self.car_type.append(car)
            # session output
            self.session.append(user)
            # arrival time output
            self.arrival_time.append(env.now)

            # energy asked by user (in Wh)
            desired_energy = input_df['cumEnergy_kWh'][user - 1]

            # energy required output
            self.energyreq.append(desired_energy)

            # print ("%s : Car %s arrived" % (str(env.now), user))
            events_q.append((int(env.now), "%s : Car %s arrived" % (str(env.now), user)))
            inst_dept = int(env.now) + 1

            # generate stay duration
            stay_duration = input_df['durationHour'][user - 1] * 60
            print('############################# start of iteration #############################')
            print('curr_time = ', int(env.now))
            print('departure_time = ', int(env.now) + int(stay_duration))
            print('requested_energy = ', desired_energy)

            ######################### Price getting ################

            # We define the timesteps in the APP as 15 minute
            delta_t = 0.25  # hour
            # print("For delta_t: ", delta_t, "max number of intervals:", 24 / delta_t)
            ################## Define the TOU Cost ##########################################
            ## the TOU cost is defined consid
            # ering the delta_t above, if not code raises an error.##

            # off-peak 0.175  cents / kwh
            TOU_tariff = np.ones((96,)) * 17.5
            ## 4 pm - 9 pm peak 0.367 cents / kwh
            TOU_tariff[64:84] = 36.7
            ## 9 am - 2 pm super off-peak 0.149 $ / kWh  to cents / kwh
            TOU_tariff[36:56] = 14.9

            par = Parameters(z0=np.array([20, 20, 1, 1]).reshape(4, 1),
                             # due to local optimality, we can try multiple starts
                             Ts=delta_t,
                             eff=1.0,
                             soft_v_eta=1e-4,
                             opt_eps=0.0001,
                             TOU=TOU_tariff)

            arrival_hour = input_df['arrivalHour']
            duration_hour = input_df['durationHour']
            e_need = input_df['cumEnergy_kWh']
            event = {
                "time": int(arrival_hour[user - 1] / delta_t),  # INTERVAL
                "e_need": e_need[user - 1],  # FLOAT
                "duration": int(duration_hour[user - 1] / delta_t),  # the slider allows 15 min increments INT
                "station_pow_max": 6.6,
                "user_power_rate": 6.6,
                "arrivalMinGlobal": input_df['arrivalMinGlobal'][user - 1],
                "departureMinGlobal": int(
                    input_df['arrivalMinGlobal'][user - 1] + input_df['durationHour'][user - 1] * 60)
            }

            print('Currently %d th user is using' % (user))
            ##################################################
            prb = Problem(par=par, event=event)

            # opt = Optimization_charger(par, prb)
            # res = opt.run_opt()

            #######################################################################
            if not station['FLEX_list'] and not station['ASAP_list']:
                opt = Optimization_charger(par, prb)
                res = opt.run_opt()
            else:
                opt = Optimization_station(par, prb, station, arrival_hour[user - 1])
                station, res = opt.run_opt()

            station["EV" + str(user)] = opt
            ################### RES GLOBAL ########################################

            self.e_need.append(res['e_need'])
            self.z_flex.append(res['z'][0])
            self.z_asap.append(res['z'][1])
            self.z_leave.append(res['z'][2])
            self.flex_tarrif.append(res['tariff_flex'])
            self.asap_tarrif.append(res['tariff_asap'])
            self.v_flex.append(res['v'][0])
            self.v_asap.append(res['v'][1])
            self.v_leave.append(res['v'][2])
            self.prob_flex.append(res['prob_flex'])
            self.prob_asap.append(res['prob_asap'])
            self.prob_leave.append(res['prob_leave'])
            self.arrival_day.append(input_df['arrivalDay'][user - 1])

            # Find the Optimized Price with the given arrival time & Energy requested & Departure
            asap_price, flex_price = (res['tariff_asap'], res['tariff_flex'])

            asap_power, flex_power = (res['asap_powers'], res['flex_powers'])
            N_asap, N_flex = (res['N_asap'], res['N_flex'])

            # Driver choice based on the tariff
            choice = choice_function(asap_price, flex_price)
            print("User's choice : ", choice[user - 1])

            start_ind = int(arrival_hour[user - 1] / delta_t)
            if choice == 1:
                total_revenue.append(asap_price * self.e_need[user - 1])
                total_cost.append(np.multiply(TOU_tariff[start_ind: start_ind + N_asap], asap_power * 0.25).sum())
            elif choice == 2:
                total_revenue.append(flex_price * self.e_need[user - 1])
                total_cost.append(np.multiply(TOU_tariff[start_ind: start_ind + N_flex], flex_power * 0.25).sum())

            self._check_pole(event['arrivalMinGlobal'], event['departureMinGlobal'])

            self.station_df = pd.DataFrame(list(
                zip(self.arrival_day, self.e_need, self.z_flex, self.z_asap, self.z_leave, self.flex_tarrif,
                    self.asap_tarrif,
                    self.v_flex, self.v_asap, self.v_leave, self.prob_flex, self.prob_asap, self.prob_leave,
                    self.car_type, self.occupied_pole_num, self.choice)),
                columns=['arrival_day', 'self.e_need', 'z_flex', 'z_asap', 'z_leave', 'flex_tarrif',
                         'asap_tarrif', 'v_flex', 'v_asap', 'v_leave', 'prob_flex', 'prob_asap',
                         'prob_leave', 'car_type', 'self.occupied_pole_num', 'Choice'])

            # Daily user Flex & Scheduled List
            self._asap_schedule_list(user, station, asap_price, flex_price)
            print('the current stat of station', station)

            print(" ############################# End of iteration #############################")
            # print('the current state of station_in', station_in)

            # rates output
            charge_time = 30 * round(stay_duration / 30)
            # flex rate requested output
            rate_flex.append(flex_price)
            # asap rate requested output
            rate_asap.append(asap_price)

            # terminal segment
            if env.now >= run_time - 30:
                events_q.sort(reverse=True)
                self.sorted_events_q = events_q

                # print timeline of events
                # while events_q:
                #     t = events_q.pop()
                #     print(t[1])

                # plot data
                x = np.array([x for x in range(len(leave_time_datalog))])

                figure(figsize=(10, 4))
                plt.plot(x, leave_time_datalog, linewidth=1)
                plt.xlabel("Number of arrivals")  # add X-axis label
                plt.ylabel("Stay duration")  # add Y-axis label
                plt.title("Number of Arrivals vs Stay duration")  # add title
                plt.show()

                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1])
                choice = ['Flex', 'ASAP', 'Leave', 'Leave\n(occupied)']
                frequency = [num_flex, num_asap, num_leave_imm, num_leave_occ]
                ax.bar(choice, frequency)
                plt.xlabel("Agent Choice")  # add X-axis label
                plt.ylabel("Frequency of choice")  # add Y-axis label
                plt.title("Agent Choice vs Frequency")
                # plt.rcParams["figure.autolayout"] = True
                # wrap_labels(ax, 500)
                # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                plt.show()

                break

                ## Todo what is the point of this code?
                # yield env.timeout(random.randint(30,self.CAR_ARR_TIME)) # random interval for next arrival
                yield env.timeout(input_df['arrivalMinGlobal'][user] - input_df['arrivalMinGlobal'][user - 1])

        return [session]

    def _check_pole(self, arrivalMinGlobal, departureMinGlobal):

        for num in self.YY:
            self.occupied_temp = (len(self.occupied_pole_num))
            if self.poles[num][arrivalMinGlobal] != "OCCUPIED":
                for time in range(arrivalMinGlobal, departureMinGlobal + 15, 15):
                    self.poles[num][time] = "OCCUPIED"
                self.occupied_pole_num.append(num)
            # available_pole_num = available_pole_num - len(self.occupied_pole_num)
            if len(self.occupied_pole_num) == self.occupied_temp + 1:
                break
            else:
                continue
        if (self.poles['1'][arrivalMinGlobal] == self.poles['2'][arrivalMinGlobal] == self.poles['3'][
            arrivalMinGlobal] == self.poles['4'][
            arrivalMinGlobal] == self.poles['5'][arrivalMinGlobal] == self.poles['6'][arrivalMinGlobal] ==
                self.poles['7'][
                    arrivalMinGlobal] == self.poles['8'][arrivalMinGlobal]):
            print("Every Poles are occupied at this moment")
            self.occupied_pole_num.append("unavailability leave")

    def _asap_schedule_list(self, user, station, asap_price, flex_price):

        total_day = 10

        for day in range(total_day):

            if self.station_df['arrival_day'][user - 1] == day:
                if user == 1:
                    day_temp = 0
                else:
                    day_temp = self.station_df['arrival_day'][user - 1]

                if self.station_df['Choice'][user - 1] == 'Regular':
                    self.temp_asap.append('EV' + str(user))
                    station["EV" + str(user)].price = asap_price
                    # temp_price.append(asap_price)
                elif self.station_df['Choice'][user - 1] == 'Scheduled':
                    self.temp_scheduled.append('EV' + str(user))
                    station["EV" + str(user)].price = flex_price
                    # temp_price.append(flex_price)
                elif self.station_df['Choice'][user - 1] == 'Leave':
                    self.temp_leave.append('EV' + str(user))
                    # station["EV" + str(user)].price  # arbitrary number for the leave price
                    # temp_price.append('700000')
                if user % 10 == 0:
                    self.temp_asap.clear()
                    self.temp_scheduled.clear()
                    self.temp_leave.clear()
                    station.clear()

                station['FLEX_list'] = self.temp_scheduled
                station['ASAP_list'] = self.temp_asap
                station['Leave'] = self.temp_leave
                station['Day'] = day_temp

        return station
