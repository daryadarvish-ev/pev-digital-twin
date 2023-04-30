from session_generator import SessionGen
from simulation import *
import simpy

random.seed(1)

# Average number of users during ###
# Spring Weekday: 8.39      Summer Weekday: 8.06        Fall Weekday: 10.69     Winter Weekday: 6.35    All seasons Weekday: 8.39
# Spring Weekend: 1.62      Summer Weekend: 2.50        Fall Weekend: 2.12      Winter Weekend: 1.67    All seasons Weekend: 1.97
# Spring Week: 6.48         Summer Week: 6.49           Fall Week: 8.24         Winter Week: 4.94       All seasons Week: 6.54


# User Choice ##########################
data_file = '../data/Sessions2.csv'         # user can change data
daily_sessions = 7                          # user can choose daily number of charging sessions
total_day = 10                              # user can choose total number of simulated days
number_of_pole = 8                          # user can choose the number of charging poles to be simulated
season = 'All season'                             # user can choose between (Spring, Summer, Fall, Winter, All season) to simulate
day_type = 'Week'                        # user can choose between (Weekday, Weekend, Week) to simulate
########################################

endtime = 60 * 24 * total_day       # this is simulation endtime
input_df = SessionGen(daily_sessions, total_day, data_file, season, day_type) # this part runs the session generation using the real data
input_df, input_real_df = input_df.generate_session() # this parts splits generated sessions into two dataframe

print('this is post processed data', input_df)

env = simpy.Environment() # set up simpy environment
Sim = Simulator(daily_sessions=daily_sessions,
                total_day=total_day,
                input_df=input_df,
                input_real_df=input_real_df,
                number_of_pole=number_of_pole)

# # MAIN
env.process(Sim.run_simulation(env, input_df, endtime))
print("simu_run_tim", endtime)
print("env", env)
env.run(until=endtime + 10)


print(Sim.log.user_data)
Sim.log.process_station_power()
print(Sim.log.station_power)
Sim.log.compute_aggregate_metrics()
print(Sim.log.aggregate_metrics)
print('total_cost:', round(Sim.log.aggregate_metrics['utility cost'].sum(), 2), 'total_revenue', round(Sim.log.aggregate_metrics['revenue'].sum(), 2))
