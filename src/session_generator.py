import pandas as pd
import numpy as np
import random


class SessionGen:

    def __init__(self, daily_sessions, data_file, rnd_seeds=(100, 200, 300)):
        self.ses = daily_sessions
        self.data = pd.read_csv(data_file, parse_dates=['connectTime', 'startChargeTime', 'Deadline', 'lastUpdate'])
        self.df = pd.DataFrame(columns=['arrivalDay','arrivalHour', 'arrivalMin', 'arrivalMinGlobal'])
        self.rnd_seeds = rnd_seeds

    def arrival_gen(self):
        self.data['arrivalMin'] = self.data['connectTime'].apply(lambda x: x.hour * 60 + x.minute)
        self.data['arrivalHour'] = self.data['connectTime'].apply(lambda x: x.hour)
        
        for i in range(len(self.ses)):
            np.random.seed(self.rnd_seeds[0] + i)
            quantiles = sorted(np.random.rand(self.ses[i]))
            aux_df = pd.DataFrame()
            aux_df['arrivalDay'] = [i]*self.ses[i]
            aux_df['arrivalHour'] = (np.quantile(self.data['arrivalMin'], quantiles))/60   # added arrival hours
            aux_df['arrivalMin'] = np.quantile(self.data['arrivalMin'], quantiles)
            aux_df['arrivalMinGlobal'] = aux_df['arrivalDay']*24*60 + aux_df['arrivalMin']
            self.df = pd.concat([self.df, aux_df])
        self.df.reset_index(inplace=True, drop=True)
        self.df['arrivalMin'] = self.df['arrivalMin'].apply(lambda x: int(x))
        self.df['arrivalHour'] = self.df['arrivalHour'].apply(lambda x: (x))
        self.df['arrivalMinGlobal'] = self.df['arrivalMinGlobal'].apply(lambda x: int(x))

    def duration_gen(self, bins=(0, 472, 654, 1440)):
        
        self.data['arrivalPeriod'] = pd.cut(self.data['arrivalMin'],
                                            bins=bins,
                                            labels=['night', 'morning', 'afternoon'])
        self.df['arrivalPeriod'] = pd.cut(self.df['arrivalMin'],
                                          bins=bins,
                                          labels=['night', 'morning', 'afternoon'])

        np.random.seed(self.rnd_seeds[1])
        quantiles = np.random.rand(self.df.shape[0])
        durations = []
        for i in range(self.df.shape[0]):
            if self.df['arrivalPeriod'][i] == 'night':
                durations.append(
                    np.quantile(self.data[self.data['arrivalPeriod'] == 'night']['DurationHrs'], quantiles[i]) * 60)
            elif self.df['arrivalPeriod'][i] == 'morning':
                durations.append(
                    np.quantile(self.data[self.data['arrivalPeriod'] == 'morning']['DurationHrs'], quantiles[i]) * 60)
            elif self.df['arrivalPeriod'][i] == 'afternoon':
                durations.append(
                    np.quantile(self.data[self.data['arrivalPeriod'] == 'afternoon']['DurationHrs'], quantiles[i]) * 60)

        self.df.drop('arrivalPeriod', axis=1, inplace=True)
        self.df['durationMin'] = durations
        self.df['durationMin'] = self.df['durationMin'].apply(lambda x: int(x))
        self.df['durationHour'] = durations                                          # added duration Hour
        self.df['durationHour'] = self.df['durationMin'].apply(lambda x:(x))/60     # added duration Hour

    def energy_gen(self, bins=(0, 217, 443, 1440)):
        self.data['durationType'] = pd.cut(self.data['DurationHrs']*60,
                                           bins=bins,
                                           labels=['short', 'medium', 'long'])
        self.data['averagePower'] = self.data['cumEnergy_Wh'] / self.data['DurationHrs']
        self.df['durationType'] = pd.cut(self.df['durationMin'],
                                           bins=bins,
                                           labels=['short', 'medium', 'long'])

        np.random.seed(self.rnd_seeds[2])
        quantiles = np.random.rand(self.df.shape[0])
        avg_pow = []
        for i in range(self.df.shape[0]):
            if self.df['durationType'][i] == 'short':
                avg_pow.append(
                    np.quantile(self.data[self.data['durationType'] == 'short']['averagePower'], quantiles[i]))
            if self.df['durationType'][i] == 'medium':
                avg_pow.append(
                    np.quantile(self.data[self.data['durationType'] == 'medium']['averagePower'], quantiles[i]))
            if self.df['durationType'][i] == 'long':
                avg_pow.append(
                    np.quantile(self.data[self.data['durationType'] == 'long']['averagePower'], quantiles[i]))

        # for i in range(len(avg_pow)): # this Seungyun added to constrain the max power rate (6600W)
        #     if (avg_pow[i] < 6600):
        #         self.df['averagePower'] = avg_pow[i]

        self.df['averagePower'] = avg_pow
        self.df['cumEnergy_kWh'] = self.df.apply(lambda x: int(x['averagePower']*x['durationMin']/(60*1000)), axis=1) # I have changed the column name from cumEnergy_Wh to kWh
        self.df.drop(['averagePower', 'durationType'], axis=1, inplace=True)

    def generate_session(self):

        # List with number of sessions for different days ([number of sessions on day 1, ..., number of sessions on the last day])

        # Remove outliers:
        self.data = self.data[(self.data['DurationHrs'] < 15) & (self.data['DurationHrs'] > 1 / 6)]
        self.data = self.data[self.data['cumEnergy_Wh'] / self.data['DurationHrs'] <= 6700]

        # Generate Data (arrival/duration/energy needed)
        self.arrival_gen()
        self.duration_gen()
        self.energy_gen()

        # Produce input_df dataframe with the generated data
        input_df = self.df
        input_df_without_preprocess = self.df

        ################### Rounding up / (15mins based) #################

        input_df['arrivalHour'] = input_df['arrivalHour'].apply(lambda x: round(x / 0.25) * 0.25 + 0.25)
        input_df['arrivalMinGlobal'] = input_df['arrivalMinGlobal'].apply(lambda x: round(x / 15) * 15 + 15)
        input_df['durationMin'] = input_df['durationMin'].apply(lambda x: round(
            x / 25) * 25 + 25)  ### convert the charge druation in termss of session which is 25. 0 remander, also changed to math ceil method
        input_df['durationHour'] = input_df['durationHour'].apply(lambda x: round(x / 0.25) * 0.25 + 0.25)
        input_df['cumEnergy_kWh'] = input_df['cumEnergy_kWh'].apply(lambda x: round(x / 10) * 10 + 10)  # Insert the kWH
        # input_df['cumEnergy_kWh'] = input_df['cumEnergy_kWh'].apply(lambda x: (x)) # Insert the kWH

        # adding departure hours
        input_df['departureHour'] = (
                    input_df['arrivalMin'].apply(lambda x: x) + input_df['durationMin'].apply(lambda x: x))
        input_df['departureHour'] = input_df['departureHour'].apply(lambda x: int(x // 60))
        input_df["departureMinGlobal"] = (
        (input_df['arrivalMinGlobal'].apply(lambda x: x) + input_df['durationHour'].apply(lambda x: x) * 60))

        # filter out impossible charing Scenario (durationMin*6.6kWh(maxpower) > cumEnergy_Wh)
        input_df = input_df[input_df['durationHour'] * 6.6 > input_df['cumEnergy_kWh']]

        ## filter out overnight charging check
        input_df = input_df[input_df['arrivalMin'] + input_df['durationMin'] < 1440]

        # show the max columns and rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # ascending order
        input_df = input_df.sort_values(['arrivalDay', 'arrivalHour', 'arrivalMinGlobal'], ascending=[True, True, True])

        # set the index again
        input_df = input_df.reset_index(drop=True)

        ##### After filtering out the data , Keep only daily sessions needed ##

        # how many session per day
        day_session = 10

        # currently how many sessions are in one day
        data_len = input_df['arrivalDay'].value_counts()

        # list of days
        days = list(range(0, day_session, 1))

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
        
        return input_df
    