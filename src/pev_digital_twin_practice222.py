# Import libraries
import math
import random

import matplotlib.pyplot as plt
# from google.colab import drive
import seaborn as sns
import simpy
from matplotlib.pyplot import figure

from optimizer_station import *
# from optimizer_tugba_practice import *
from session_generation_practice import *

#from Station import *

# drive.mount('/content/gdrive', force_remount=True)
# %cd /content/gdrive/MyDrive/capstone/Optimizer/
# import optimizer_v2 as optimizer

warnings.filterwarnings('ignore')
# %matplotlib inline
sns.set_theme()


# SIMULATION PARAMETERS
NUM_DAYS = 10 # Arbitrarily changed this
SIM_RUN_TIME = 1440*NUM_DAYS
CAR_ARR_TIME = 120
CAR_STAY_TIME = 300

sorted_events_q = []

#session output
session = []
#choice output
user_choice = []
#arrival time output
arrival_time = []
#departure time output
departure_time = []
#cartype output
car_type = []
#energy requested output
energyreq = []
#flex rate requested output
rate_flex = []
#asap rate requested output
rate_asap = []
###############################
global temp_res
global e_NEED

global z_flex
global z_asap
global z_leave

global flex_tarrif
global asap_tarrif
global leave_tarrif

global v_flex
global v_asap
global v_leave

global prob_flex
global prob_asap
global prob_leave

global j_flex
global j_asap
global j_leave
global Choice
global occupied_pole_num
global occupied_temp
global arrival_day

global station_in
global temp_ASAP
global temp_Scheduled
global temp_leave
station_in = dict()

e_NEED = []

z_flex = []
z_asap = []
z_leave = []

flex_tarrif = []
asap_tarrif = []
leave_tarrif = []

v_flex = []
v_asap = []
v_leave = []

prob_flex = []
prob_asap = []
prob_leave = []

j_flex = []
j_asap = []
j_leave = []
Choice = []
occupied_pole_num = []
occupied_temp =0
arrival_day =[]

temp_ASAP = []
temp_Scheduled =[]
temp_leave = []
temp_price = []
models = ['Prius Prime','Model 3','Bolt EV', 'Kona','Model X 100 Dual','Clarity','Volt (second generation)','B-Class Electric Drive','Model S 100 Dual','Mustang Mach-E','Model S 90 Dual','Bolt EUV 2LT']

def charger_station(env, input_df, run_time):

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
    station['Leave'] = list( )
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

        # cartype output
        car = random.choices(models, weights=(151, 110, 86, 51, 42, 42, 28, 24, 20, 19, 15, 14))
        car_type.append(car)
        # session output
        session.append(user)
        # arrival time output
        arrival_time.append(env.now)

        # energy asked by user (in Wh)
        desired_energy = input_df['cumEnergy_kWh'][user - 1]

        # energy required output
        energyreq.append(desired_energy)

        # print ("%s : Car %s arrived" % (str(env.now), user))
        events_q.append((int(env.now), "%s : Car %s arrived" % (str(env.now), user)))
        inst_dept = int(env.now) + 1

        # generate stay duration
        stay_duration = input_df['durationHour'][user - 1] * 60

        print('curr_time = ', int(env.now))
        print('departure_time = ', int(env.now) + int(stay_duration))
        print('requested_energy = ', desired_energy)

        ######################### Price getting ################

        # We define the timesteps in the APP as 15 minute
        delta_t = 0.25  # hour
        print("For delta_t: ", delta_t, "max number of intervals:", 24 / delta_t)
        ################## Define the TOU Cost ##########################################
        ## the TOU cost is defined consid
        # ering the delta_t above, if not code raises an error.##

        # off-peak 0.175  cents / kwh
        TOU_tariff = np.ones((96,)) * 17.5
        ## 4 pm - 9 pm peak 0.367 cents / kwh
        TOU_tariff[64:84] = 36.7
        ## 9 am - 2 pm super off-peak 0.149 $ / kWh  to cents / kwh
        TOU_tariff[36:56] = 14.9

        par = Parameters(z0 = np.array([20, 20, 1, 1]).reshape(4, 1),
                         # due to local optimality, we can try multiple starts
                         Ts = delta_t,
                         eff = 1.0,
                         soft_v_eta = 1e-4,
                         opt_eps = 0.0001,
                         TOU = TOU_tariff)

        arrival_hour = input_df['arrivalHour']
        duration_hour = input_df['durationHour']
        e_need = input_df['cumEnergy_kWh']
        event = {
            "time": int(arrival_hour[user - 1] / delta_t),  # INTERVAL
            "e_need": e_need[user - 1],  # FLOAT
            "duration": int(duration_hour[user - 1] / delta_t),  # the slider allows 15 min increments INT
            "station_pow_max": 6.6,
            "user_power_rate": 6.6,
            "arrivalMinGlobal": input_df['arrivalMinGlobal'][user-1],
            "departureMinGlobal": int(input_df['arrivalMinGlobal'][user-1] + input_df['durationHour'][user-1] * 60)
        }

        print('Currently user %d is using' %(user))
        ##################################################
        prb = Problem(par=par, event=event)
        ######################################################
        # opt = Optimization_charger(par, prb)
        #
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

        e_NEED.append(res['e_need'])
        z_flex.append(res['z'][0])
        z_asap.append(res['z'][1])
        z_leave.append(res['z'][2])
        flex_tarrif.append(res['tariff_flex'])
        asap_tarrif.append(res['tariff_asap'])
        # leave_tarrif.append(res['tariff_overstay'])  ## Erased in the station_df too. if you want to add later, you have to add in the station_df too.
        v_flex.append(res['v'][0])
        v_asap.append(res['v'][1])
        v_leave.append(res['v'][2])
        prob_flex.append(res['prob_flex'])
        prob_asap.append(res['prob_asap'])
        prob_leave.append(res['prob_leave'])
        arrival_day.append(input_df['arrivalDay'][user-1])

        # Find the Optimized Price with the given arrival time & Energy requested & Departure
        asap_price, flex_price = (res['tariff_asap'], res['tariff_flex'])

        # Driver choice based on the tariff
        choice = choice_function(asap_price, flex_price)

        # If the choice is ASAP then, we change the duration
        #################################################THIS is when ASAP user departs right after charging is over ########################
        # if (choice == 1):
        #     input_df['departureMinGlobal'][user-1] = int(math.ceil((input_df['arrivalMinGlobal'][user-1] + (input_df['cumEnergy_kWh'][user-1]/event['station_pow_max'])*60)/15)*15)
        #     event['departureMinGlobal'] = int(math.ceil((input_df['arrivalMinGlobal'][user - 1] + (input_df['cumEnergy_kWh'][user - 1] / event['station_pow_max']) * 60) / 15) * 15)
        #     # print('this is departureMin global', input_df['departureMinGlobal'][user-1])
        # # if the choice is leave, THen remove the duration
        # if (choice == 3):
        #     input_df['departureMinGlobal'][user - 1] = input_df['arrivalMinGlobal'][user-1]
        #     event['departureMinGlobal'] = input_df['arrivalMinGlobal'][user - 1]
        ##############################################################################################################
        # Check the pole to build pole occupancy data
        # check_pole(event['arrivalMinGlobal'], input_df['departureMinGlobal'][user-1])
        # print(event['arrivalMinGlobal'], event['departureMinGlobal'])
        check_pole(event['arrivalMinGlobal'], event['departureMinGlobal'])

        global station_df
        station_df = pd.DataFrame(list(zip(arrival_day,e_NEED,z_flex,z_asap,z_leave,flex_tarrif,asap_tarrif,v_flex,v_asap,v_leave,prob_flex,prob_asap,prob_leave,car_type,occupied_pole_num,Choice)),
                                  columns=['arrival_day','e_need', 'z_flex', 'z_asap', 'z_leave', 'flex_tarrif', 'asap_tarrif', 'v_flex','v_asap', 'v_leave', 'prob_flex', 'prob_asap', 'prob_leave','car_type','occupied_pole_num','Choice'])

        ############### Daily user Flex & Scheduled List
        ASAP_Schedule_list(user, station, opt, asap_price, flex_price)
        print('the current stat of station', station)


        # for user in station_in['FLEX_list']:
        #     station["FLEX_list"].append("EV"+ str(user))
        #     station["EV" + str(user)].price = asap_price
        # for user in station_in['ASAP_list']:
        #     station["FLEX_list"].append("EV"+ str(user))
        #     station["EV" + str(user)].price = asap_price
        # for user in station_in['Leave']:
        #     station["FLEX_list"].append("EV"+ str(user))
        #     station["EV" + str(user)].price = asap_price

        # station["Day"].append(station_in['Day'])
        ################

        print('the current stat of station', station)
        # print('the current state of station_in', station_in)

        # if choice == 1:
        #     station["ASAP_list"].append("EV" + str(user))
        #     station["EV" + str(user)].price = asap_price
        # if choice == 2:
        #     station["FLEX_list"].append("EV" + str(user))
        #     station["EV" + str(user)].price = flex_price
        # elif choice == 3:
        #     station["LEAVE_list"].append("EV" + str(user))
        #     station["EV" + str(user)].price = 60000 # arbitrary number for the leave price

        # rates output
        charge_time = 30 * round((stay_duration) / 30)
        # flex rate requested output
        rate_flex.append(flex_price)
        # asap rate requested output
        rate_asap.append(asap_price)

        intervals = range(SIM_RUN_TIME)
        intervals = intervals[::15]

        # terminal segment
        if env.now >= run_time - 30 :
            events_q.sort(reverse = True)
            sorted_events_q = events_q

            # print timeline of events
            # while events_q:
            #     t = events_q.pop()
            #     print(t[1])

            # plot data
            x = np.array([x for x in range(len(leave_time_datalog))])

            figure(figsize = (10, 4))
            plt.plot(x, leave_time_datalog, linewidth=1)
            plt.xlabel("Number of arrivals")  # add X-axis label
            plt.ylabel("Stay duration")  # add Y-axis label
            plt.title("Number of Arrivals vs Stay duration")  # add title
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
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
            #yield env.timeout(random.randint(30,CAR_ARR_TIME)) # random interval for next arrival
            yield env.timeout(input_df['arrivalMinGlobal'][user] - input_df['arrivalMinGlobal'][user-1])  # index를 바꿔봐 앞에걸 user+1 뒤에걸 그냥 user

    return[session]

    # MAIN
    env = simpy.Environment()

    env.process(first_process(env, input_df, SIM_RUN_TIME)) ### input_df
    print("simu_run_tim",SIM_RUN_TIME)
    print("env", env)
    env.run(SIM_RUN_TIME + 10)
    # intervals = range(SIM_RUN_TIME)
    # intervals = intervals[::15]
    # print(intervals)


def choice_function(asap_price, flex_price):
    # choose lower price
    if random.uniform(0, 1) > 0.9:
        choice = 3
        Choice.append("Leave")
        return choice
    if asap_price > flex_price:
        choice = 1
        Choice.append("Regular")
        return choice
    else:
        choice = 2
        Choice.append("Scheduled")
        return choice
# def choice_check(choice):
#     if choice == "Regular"
#
#     else:
#         choice = 2
#         Choice.append("Scheduled")
#     return choice
#

# poles
global MM, YY, poles
MM = range(SIM_RUN_TIME)
MM = MM[::15]
YY = ['1', '2', '3', '4', '5', '6', '7', '8']

poles = pd.DataFrame(columns=['1', '2', '3', '4', '5', '6', '7', '8'], index=MM)


def check_pole(arrivalMinGlobal, departureMinGlobal):

    total_numb_poles = 8
    available_pole_num = 8

    # occupied_temp = 0
    count = 0

    for num in YY:
        occupied_temp = (len(occupied_pole_num))
        if (poles[num][arrivalMinGlobal] != "OCCUPIED"):
            for time in range(arrivalMinGlobal,departureMinGlobal+15,15):
                poles[num][time] = "OCCUPIED"
            occupied_pole_num.append(num)
        # available_pole_num = available_pole_num - len(occupied_pole_num)
        if (len(occupied_pole_num) == (occupied_temp) + 1):
            break
        else:
            continue
    if (poles['1'][arrivalMinGlobal] == poles['2'][arrivalMinGlobal]== poles['3'][arrivalMinGlobal]==poles['4'][arrivalMinGlobal]==poles['5'][arrivalMinGlobal]==  poles['6'][arrivalMinGlobal]== poles['7'][arrivalMinGlobal]== poles['8'][arrivalMinGlobal]):
        print ("Every Poles are occupied at this moment")
        occupied_pole_num.append("unavailability leave")

def ASAP_Schedule_list (user , station, opt, asap_price, flex_price):

    total_day = 10

    for day in range(total_day):

        if (station_df['arrival_day'][user-1] == day):
            if (user == 1):
                day_temp = 0
            else:
                # day_temp = station_df['arrival_day'][user-2]
                day_temp = station_df['arrival_day'][user-1]
            # if (user % 10 == 0):
            # if ( day_temp != day ):
            #     temp_ASAP.clear()
            #     temp_Scheduled.clear()
            #     temp_leave.clear()
            #     station.clear()
                # station['FLEX_list'] = []
                # station['ASAP_list']= []
                # station['Leave'] =[]
                # station['Day'] = []

            if(station_df['Choice'][user-1] == 'Regular'):
                temp_ASAP.append('EV' + str(user))
                station["EV" + str(user)].price = asap_price
                # temp_price.append(asap_price)
            elif(station_df['Choice'][user-1] == 'Scheduled'):
                temp_Scheduled.append('EV' + str(user))
                station["EV" + str(user)].price = flex_price
                # temp_price.append(flex_price)
            elif (station_df['Choice'][user - 1] == 'Leave'):
                temp_leave.append('EV' + str(user))
                # station["EV" + str(user)].price  # arbitrary number for the leave price
                # temp_price.append('700000')
            if (user % 10 == 0):
                temp_ASAP.clear()
                temp_Scheduled.clear()
                temp_leave.clear()
                station.clear()

            station['FLEX_list'] = temp_ASAP
            station['ASAP_list'] = temp_Scheduled
            station['Leave'] = temp_leave
            station['Day'] = day_temp

            # station['Price'] = temp_price
            # if ( user == 10 ):
            #     temp_ASAP.clear()
            #     temp_Scheduled.clear()
            #     temp_leave.clear()
            #     station.clear()

    return(station)

# station["EV" + str(user)] = opt
# station["ASAP_list"].append("EV" + str(user))
# station["EV" + str(user)].price = asap_price



def first_process(env, input_df, run_length):
    yield env.process(charger_station(env, input_df, run_length))

# # MAIN
env = simpy.Environment()

env.process(first_process(env, input_df, SIM_RUN_TIME)) ### input_df
print("simu_run_tim",SIM_RUN_TIME)
print("env", env)
env.run(SIM_RUN_TIME + 10)
# intervals = range(SIM_RUN_TIME)
# intervals = intervals[::15]
# print(intervals)