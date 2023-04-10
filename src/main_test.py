from run_session_generator import *
from New_simulation import *
import simpy

# User Choice #
data_file = '../data/Sessions2.csv'
daily_sessions = 10
total_day = 10
endtime = 60 * 24 * total_day


input_df, input_real_df = run_session_generator(daily_sessions, total_day, data_file) # here we can select the number
# print('this is post processed data', input_df)


# print(input_df)
#
env = simpy.Environment() # set up simpy environment
Sim = Simulator(daily_sessions, total_day, input_df, input_real_df, env)

# # MAIN
env.process(Sim.run_simulation(env, input_df, endtime)) ### input_df
print("simu_run_tim",endtime)
print("env", env)
env.run(until= endtime + 10)


Sim = Simulator(daily_sessions, total_day, input_df, input_real_df, env)
Sim1 = Sim.run_simulation(env, input_df, endtime)

