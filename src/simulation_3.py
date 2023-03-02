
from optimizer_station import *
import simpy
# from simulation import *
# from simulation import input_df
# from simulation import df
import pandas as pd
from TOU_2 import *
from driver import *
from session_generation_practice import input_df




file_path = r'C:\Users\seungyun\Desktop\df_3.csv'
df = pd.read_csv(file_path)
print(df)


env = simpy.Environment()
t_end = 1440

def simulator (env, df, t_end):

    dict = {"ASAP_list": [], "FLEX_list": []}

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

    while True:

        time = int(env.now)
        # print("simulation time now :", env.now)
        yield env.timeout(15)
        if (time > t_end):
            break # break when now is at t_end

        ### THIS is one I use
        users_reg = [int(user) for user in df.loc[df.index == time / 15, 'regular'].item().split(',') if user != '0']
        users_flex = [int(user) for user in df.loc[df.index == time / 15, 'scheduled'].item().split(',') if
                      user != '0']

        users_new_reg = [int(user) for user in str(df.loc[df.index == time / 15, 'new_regular'].item()).split(',') if
                         user != '0']
        users_new_flex = [int(user) for user in str(df.loc[df.index == time / 15, 'new_scheduled'].item()).split(',') if
                          user != '0']

        ####
        # Convert the lists to sets
        users_reg_set = set(users_reg)
        users_flex_set = set(users_flex)
        users_new_reg_set = set(users_new_reg)
        users_new_flex_set = set(users_new_flex)

        # Find the duplicates and remove them from users_reg_set and users_flex_set
        duplicates_reg = users_reg_set.intersection(users_new_reg_set)
        duplicates_flex = users_flex_set.intersection(users_new_flex_set)
        users_reg_set -= duplicates_reg
        users_flex_set -= duplicates_flex

        # Convert the sets back to lists
        users_reg = list(users_reg_set)
        users_flex = list(users_flex_set)


        print('this is time', time)
        print('this is regular users', users_reg)
        print('this is flex users', users_flex)
        print('this is new regular users', users_new_reg)
        print('this is new flex users', users_new_flex)

        store = {"ASAP_list": [], "Flex_list": [], "users_new_reg": [], "users_new_flex": []}

        # if not users_reg and not users_flex:
        #     print("Both lists are empty")
        # else:
        store["ASAP_list"] = users_reg
        store["Flex_list"] = users_flex
        store["users_new_reg"] = users_new_reg
        store["users_new_flex"] = users_new_flex

        print(store)


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
                         TOU=TOU_tariff)  # retrieve the

        # print(par2)
        ##################################
        arrival_hour = input_df['arrivalHour']

        ### where the optimization and simulation will be linked to each other.

        drv = driver(input_df, 0.25)





        if (not store["users_new_reg"] and not store["users_new_flex"] and not store["users_new_reg"] and not store["users_new_flex"] ):
            print("###### NOTHING HERE #####")
            continue


        if (len(store["users_new_reg"]) > 0 or len(store["users_new_flex"]) > 0) and not store["ASAP_list"] and not store["Flex_list"]:
            print("##### THIS IS FIRST ONE #####")
            for user in store["users_new_reg"] + store["users_new_flex"]:
                event = drv.generate_random_user(user)
                prb = Problem(par=par, event=event)

                opt = Optimization_charger(par, prb)
                res = opt.run_opt()

                dict["EV" + str(user)] = opt

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

                # Find the Optimized Price with the given arrival time & Energy requested & Departure
                asap_price, flex_price = (res['tariff_asap'], res['tariff_flex'])

                asap_power, flex_power = (res['asap_powers'], res['flex_powers'])
                N_asap, N_flex = (res['N_asap'], res['N_flex'])

                # Driver choice based on the tariff
                # choice = choice_function(asap_price, flex_price)
                # print("User's choice : ", choice[user - 1])
                #
                # start_ind = int(arrival_hour[user - 1] / delta_t)
                # if choice == 1:
                #     total_revenue.append(asap_price * e_need[user - 1])
                #     total_cost.append(np.multiply(TOU_tariff[start_ind: start_ind + N_asap], asap_power * 0.25).sum())
                # elif choice == 2:
                #     total_revenue.append(flex_price * e_need[user - 1])
                #     total_cost.append(np.multiply(TOU_tariff[start_ind: start_ind + N_flex], flex_power * 0.25).sum())

        else:
            print("##### THIS IS SECOND ONE #####")
            for user in store["ASAP_list"]:
                dict["ASAP_list"].append("EV" + str(user))
            for user in store["Flex_list"]:
                dict["FLEX_list"].append("EV" + str(user))

            for user in (store["users_new_reg"] + store["users_new_flex"]):
                event = drv.generate_random_user(user)
                prb = Problem(par=par, event=event)

                opt = Optimization_station(par, prb, dict, arrival_hour[user - 1])
                dict, res = opt.run_opt()

                dict["EV" + str(user)] = opt

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

# # MAIN
env = simpy.Environment()

env.process(simulator(env, df, t_end)) ### input_df
print("simu_run_tim",t_end)
print("env", env)
env.run(until= t_end + 10)

