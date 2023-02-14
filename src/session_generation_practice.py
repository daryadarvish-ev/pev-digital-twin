import math

from session_generator2 import *
import random

# List with number of sessions for different days ([number of sessions on day 1, ..., number of sessions on the last day])
daily_sessions = [30]*10

input_gen = InputGen(daily_sessions=daily_sessions, data_file='../data/Sessions2.csv', rnd_seeds=(4,5,30))
# input_gen = InputGen(daily_sessions=daily_sessions, data_file='../data/ChargePointEV.csv', rnd_seeds=(4,5,30))

# Remove outliers:
input_gen.data = input_gen.data[(input_gen.data['DurationHrs'] < 15) & (input_gen.data['DurationHrs'] > 1/6)]
input_gen.data = input_gen.data[input_gen.data['cumEnergy_Wh'] / input_gen.data['DurationHrs'] <= 6700]

# Generate Data (arrival/duration/energy needed)
input_gen.arrival_gen()
input_gen.duration_gen()
input_gen.energy_gen()

# Produce input_df dataframe with the generated data
input_df = input_gen.df
input_df_without_preprocess = input_gen.df

################### Rounding up / (15mins based) #################

input_df['arrivalHour'] = input_df['arrivalHour'].apply(lambda x: round(x/0.25)*0.25 + 0.25)
input_df['arrivalMinGlobal'] = input_df['arrivalMinGlobal'].apply(lambda x: round(x/15)*15 +15)
input_df['durationMin'] = input_df['durationMin'].apply(lambda x: round(x/25)*25 + 25)    ### convert the charge druation in termss of session which is 25. 0 remander, also changed to math ceil method
input_df['durationHour'] = input_df['durationHour'].apply(lambda x: round(x/0.25)*0.25 + 0.25)
input_df['cumEnergy_kWh'] = input_df['cumEnergy_kWh'].apply(lambda x: round(x/10)*10 + 10)  # Insert the kWH
#input_df['cumEnergy_kWh'] = input_df['cumEnergy_kWh'].apply(lambda x: (x)) # Insert the kWH

# adding departure hours
input_df['departureHour'] = (input_df['arrivalMin'].apply(lambda x: x) + input_df['durationMin'].apply(lambda x: x))
input_df['departureHour'] = input_df['departureHour'].apply(lambda x: int(x//60))
input_df["departureMinGlobal"] = ((input_df['arrivalMinGlobal'].apply(lambda x:x) + input_df['durationHour'].apply(lambda x: x) * 60))

#filter out impossible charing Scenario (durationMin*6.6kWh(maxpower) > cumEnergy_Wh)
input_df = input_df[input_df['durationHour'] * 6.6 > input_df['cumEnergy_kWh']]

## filter out overnight charging check
input_df = input_df[input_df['arrivalMin'] + input_df['durationMin'] < 1440]

# show the max columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ascending order
input_df = input_df.sort_values(['arrivalDay', 'arrivalHour','arrivalMinGlobal'], ascending = [True,True, True])

# set the index again
input_df = input_df.reset_index(drop=True)

##### After filtering out the data , Keep only daily sessions needed ##

# how many session per day
day_session = 10

# currently how many sessions are in one day
data_len = input_df['arrivalDay'].value_counts()

# list of days
days = list(range(0,day_session,1))

# dictionary for each day with index
d = {day: [] for day in days}

for i in range(len(input_df)):
    for j in range(len(data_len)):
        if (input_df.at[i, 'arrivalDay'] == j):
            d[j].append(input_df.index[i])

# If each day has more than day_session( for this case it is 10 sessions per day), then randomly remove the surplus sessions from each day
Val_list = []
for i in range(len(d)):
    # how many sessions needs to be removed
    to_remove = len(d[i]) - day_session
    # randomly choose the number of how many to remove from the data_index
    remove_values = random.sample(d[i], to_remove)
    # create a list of values to keep
    to_keep = [value for value in d[i] if value not in remove_values]
    # replace d[i] with the list of values to keep
    d[i] = to_keep
    Val_list += to_keep

input_df = input_df.loc[input_df.index.isin(Val_list)]

print(input_df['arrivalDay'].value_counts())
input_df = input_df.dropna()
input_df = input_df.reset_index(drop=True)
print(input_df)
print(input_df.shape)



