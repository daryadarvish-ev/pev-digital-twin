from session_generator import SessionGen
from simulation import *
import simpy

# User Choice #
data_file = '../data/Sessions2.csv'
daily_sessions = 10
total_day = 10
endtime = 60 * 24 * total_day

input_df = SessionGen(daily_sessions, total_day, data_file)
input_df, input_real_df = input_df.generate_session()

print('this is post processed data', input_df)

env = simpy.Environment() # set up simpy environment
Sim = Simulator(daily_sessions, total_day, input_df, input_real_df, env)

# # MAIN
env.process(Sim.run_simulation(env, input_df, endtime)) ### input_df
print("simu_run_tim",endtime)
print("env", env)
env.run(until= endtime + 10)


Sim = Simulator(daily_sessions, total_day, input_df, input_real_df, env)
Sim1 = Sim.run_simulation(env, input_df, endtime)
