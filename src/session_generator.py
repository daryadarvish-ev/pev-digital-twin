import pandas as pd
import numpy as np

class InputGen:

    def __init__(self, daily_sessions, data_file, rnd_seeds=(100, 200, 300)):
        self.ses = daily_sessions
        self.data = pd.read_csv(data_file, parse_dates=['connectTime', 'startChargeTime', 'Deadline', 'lastUpdate'])
        self.df = pd.DataFrame(columns=['arrivalDay', 'arrivalMin', 'arrivalMinGlobal'])
        self.rnd_seeds = rnd_seeds

    def arrival_gen(self):
        self.data['arrivalMin'] = self.data['connectTime'].apply(lambda x: x.hour * 60 + x.minute)
        for i in range(len(self.ses)):
            np.random.seed(self.rnd_seeds[0] + i)
            quantiles = sorted(np.random.rand(self.ses[i]))
            aux_df = pd.DataFrame()
            aux_df['arrivalDay'] = [i]*self.ses[i]
            aux_df['arrivalMin'] = np.quantile(self.data['arrivalMin'], quantiles)
            aux_df['arrivalMinGlobal'] = aux_df['arrivalDay']*24*60 + aux_df['arrivalMin']
            self.df = pd.concat([self.df, aux_df])
        self.df.reset_index(inplace=True, drop=True)
        self.df['arrivalMin'] = self.df['arrivalMin'].apply(lambda x: int(x))
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

        self.df['averagePower'] = avg_pow
        self.df['cumEnergy_Wh'] = self.df.apply(lambda x: int(x['averagePower']*x['durationMin']/(60*1000)), axis=1)
        self.df.drop(['averagePower', 'durationType'], axis=1, inplace=True)