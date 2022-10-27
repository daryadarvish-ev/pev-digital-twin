import math

from session_generator2 import *


# List with number of sessions for different days ([number of sessions on day 1, ..., number of sessions on the last day])
daily_sessions = [10]*10

input_gen = InputGen(daily_sessions=daily_sessions, data_file='../data/Sessions2.csv', rnd_seeds=(4,5,30))
# Remove outliers:
input_gen.data = input_gen.data[(input_gen.data['DurationHrs'] < 15) & (input_gen.data['DurationHrs'] > 1/6)]
input_gen.data = input_gen.data[input_gen.data['cumEnergy_Wh'] / input_gen.data['DurationHrs'] <= 6700]

input_gen.arrival_gen()
print(input_gen.arrival_gen())
input_gen.duration_gen()
input_gen.energy_gen()

input_df = input_gen.df

#input_df['arrivalMin'] = input_df['arrivalMin'].apply(lambda x: round(x/30)*30 + 30)
#input_df['arrivalMinGlobal'] = input_df['arrivalMinGlobal'].apply(lambda x: round(x/30)*30 + 30)
input_df['durationMin'] = input_df['durationMin'].apply(lambda x: math.ceil(x/15)*15)    ### convert the charge druation in termss of session which is 15. 0 remander, also changed to math ceil method
input_df['durationHour'] = input_df['durationHour'].apply(lambda x: math.ceil(x/15)*15)//60
input_df['cumEnergy_kWh'] = input_df['cumEnergy_kWh'].apply(lambda x: round(x/10)*10 + 10)  # Insert the kWH
#input_df['cumEnergy_kWh'] = input_df['cumEnergy_kWh'].apply(lambda x: (x)) # Insert the kWH

# adding departure hours
input_df['departureHour'] = (input_df['arrivalMin'].apply(lambda x: x) + input_df['durationMin'].apply(lambda x: x))
input_df['departureHour'] = input_df['departureHour'].apply(lambda x: int(x//60))

#filter out impossible charing Scenario (durationMin*6.6kWh(maxpower) > cumEnergy_Wh)
input_df = input_df[input_df['durationMin']/60 * 6.6 > input_df['cumEnergy_kWh']]

## filter out overnight charging check
input_df = input_df[input_df['arrivalMin'] + input_df['durationMin'] < 1440]

# input_df.head(10)
pd.set_option('display.max_columns', None)
print(input_df)
input_df = input_df.reset_index(drop=True)  # indexing 다시하기

print(input_df)
print(input_df.shape)
