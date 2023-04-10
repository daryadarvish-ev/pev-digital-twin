from session_generator import *
import random
# random.seed(4) # before ->51

# random.seed(3) # before
# random.seed(4) # before ->60
random.seed(1)# Regular 43, Scheduled 40, Leave 17
# random.seed(2) # 52/ 35/ 13

# daily_sessions = [30] * 10
# data_file = '../data/Sessions2.csv'

def run_session_generator(daily_sessions, total_day, data_file):

    # daily_sessions = [daily_sessions*3]*total_day

    input_df = SessionGen(daily_sessions, total_day, data_file)
    input_df, input_real_df = input_df.generate_session()


    # print('this is the pre-processed data,' , input_df)
    #
    # input_df['arrivalHour'] = input_df['arrivalHour'].apply(lambda x: round(x / 0.25) * 0.25 + 0.25)
    # input_df['arrivalMinGlobal'] = input_df['arrivalMinGlobal'].apply(lambda x: round(x / 15) * 15 + 15)
    # # input_df['arrivalMinGlobal'] = input_df['arrivalMinGlobal'].apply(lambda x: round(x))
    #
    # input_df['durationMin'] = input_df['durationMin'].apply(lambda x: round(x / 25) * 25 + 25)  ### convert the charge druation in termss of session which is 25. 0 remander, also changed to math ceil method
    # input_df['durationHour'] = input_df['durationHour'].apply(lambda x: round(x / 0.25) * 0.25 + 0.25)
    # input_df['cumEnergy_kWh'] = input_df['cumEnergy_kWh'].apply(lambda x: round(x / 10) * 10 + 10)  # Insert the kWH
    # # input_df['cumEnergy_kWh'] = input_df['cumEnergy_kWh'].apply(lambda x: (x)) # Insert the kWH
    #
    # # adding departure hours
    # input_df['departureHour'] = (input_df['arrivalMin'].apply(lambda x: x) + input_df['durationMin'].apply(lambda x: x))
    # input_df['departureHour'] = input_df['departureHour'].apply(lambda x: int(x // 60))
    # input_df["departureMinGlobal"] = (input_df['arrivalMinGlobal'].apply(lambda x: x) + input_df['durationHour'].apply(lambda x: x) * 60)
    #
    # # filter out impossible charing Scenario (durationMin*6.6kWh(maxpower) > cumEnergy_Wh)
    # input_df = input_df[input_df['durationHour'] * 6.6 > input_df['cumEnergy_kWh']]
    #
    # ## filter out overnight charging check
    # input_df = input_df[input_df['arrivalMin'] + input_df['durationMin'] < 1440]
    #
    # # filter out not charging scenario -> even though there are requested amount of energy from the user, there is instance that the charger didn't porivde energy
    # input_df = input_df[input_df['cumEnergy_kWh'] > 0]
    #
    # # show the max columns and rows
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    # # ascending order
    # input_df= input_df.sort_values(['arrivalDay', 'arrivalHour', 'arrivalMinGlobal'], ascending=[True, True, True])
    #
    # # set the index again
    # input_df = input_df.reset_index(drop=True)


    # ##### After filtering out the data , Keep only daily sessions needed ##
    #
    # # how many session per day
    # day_session = 10
    #
    # # currently how many sessions are in one day
    # data_len = input_df['arrivalDay'].value_counts()
    #
    # # list of days
    # days = list(range(0, day_session, 1))
    #
    # # dictionary for each day with index
    # d = {day: [] for day in days}
    #
    # for i in range(len(input_df)):
    #     for j in range(len(data_len)):
    #         if (input_df.at[i, 'arrivalDay'] == j):
    #             d[j].append(input_df.index[i])
    #
    # # If each day has more than day_session( for this case it is 10 sessions per day), then randomly remove the surplus sessions from each day
    # Val_list = []
    # for i in range(len(d)):
    #     # how many sessions needs to be removed
    #     to_remove = len(d[i]) - day_session
    #     to_remove = int(to_remove)
    #     # randomly choose the number of how many to remove from the data_index
    #     # remove_values = random.sample(d[i], to_remove)
    #     remove_values = random.sample(d[i], min(to_remove, len(d[i]))) if to_remove >= 0 else []
    #
    #     # create a list of values to keep
    #     to_keep = [value for value in d[i] if value not in remove_values]
    #     # replace d[i] with the list of values to keep
    #     d[i] = to_keep
    #     Val_list += to_keep
    #
    # input_df = input_df.loc[input_df.index.isin(Val_list)]
    #
    # # print(input_df['arrivalDay'].value_counts())
    # input_df = input_df.dropna()
    # input_df = input_df.reset_index(drop=True)
    #

    return (input_df, input_real_df)
