from optimizer_station import *
from simulation_choice_function import *
import pandas as pd

def run_optimization(par,prb,station,arrival_hour,user,input_df):

    e_NEED = []
    z_flex = []
    z_asap = []
    z_leave = []
    flex_tarrif = []
    asap_tarrif = []
    v_flex = []
    v_asap = []
    v_leave = []
    prob_flex = []
    prob_asap = []
    prob_leave = []
    arrival_day=[]
    arrival_time =[]

    if not station['FLEX_list'] and not station['ASAP_list']:
        opt = Optimization_charger(par, prb)
        res = opt.run_opt()
    else:
        opt = Optimization_station(par, prb, station, arrival_hour[user - 1])
        station, res = opt.run_opt()

    station["EV" + str(user)] = opt

    e_NEED.append(res['e_need'])
    z_flex.append(res['z'][0])
    z_asap.append(res['z'][1])
    z_leave.append(res['z'][2])
    flex_tarrif.append(res['tariff_flex'])
    asap_tarrif.append(res['tariff_asap'])

    v_flex.append(res['v'][0])
    v_asap.append(res['v'][1])
    v_leave.append(res['v'][2])

    prob_flex.append(res['prob_flex'])
    prob_asap.append(res['prob_asap'])
    prob_leave.append(res['prob_leave'])
    arrival_day.append(input_df['arrivalDay'][user - 1])
    arrival_time.append(input_df['arrivalMinGlobal'][user - 1])

    # Find the Optimized Price with the given arrival time & Energy requested & Departure
    asap_price, flex_price = (res['tariff_asap'], res['tariff_flex'])

    asap_power, flex_power = (res['asap_powers'], res['flex_powers'])
    N_asap, N_flex = (res['N_asap'], res['N_flex'])

    # Driver choice based on the tariff
    choice = basic_choice_function(asap_price, flex_price)
    print("User's choice : ", choice[user - 1])

    # start_ind = int(arrival_hour[user - 1] / delta_t)
    # if choice == 1:
    #     total_revenue.append(asap_price * e_need[user - 1])
    #     total_cost.append(np.multiply(TOU_tariff[start_ind: start_ind + N_asap], asap_power * 0.25).sum())
    # elif choice == 2:
    #     total_revenue.append(flex_price * e_need[user - 1])
    #     total_cost.append(np.multiply(TOU_tariff[start_ind: start_ind + N_flex], flex_power * 0.25).sum())
    #
    # check_pole(event['arrivalMinGlobal'], event['departureMinGlobal'], SIM_RUN_TIME)

    station_df = pd.DataFrame(list(
        zip(arrival_time, arrival_day, e_NEED, z_flex, z_asap, z_leave, flex_tarrif, asap_tarrif, v_flex, v_asap,
            v_leave, prob_flex, prob_asap, prob_leave, choice)),
                              columns=['arrival_time', 'arrival_day', 'e_need', 'z_flex', 'z_asap', 'z_leave',
                                       'flex_tarrif', 'asap_tarrif', 'v_flex', 'v_asap', 'v_leave', 'prob_flex',
                                       'prob_asap', 'prob_leave', 'Choice'])

                # removed car type from station_df