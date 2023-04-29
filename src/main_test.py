from session_generator import SessionGen
from simulation import *
import simpy

random.seed(1)

# User Choice #
data_file = '../data/Sessions2.csv'
daily_sessions = 10
total_day = 3
endtime = 60 * 24 * total_day
number_of_pole = 4

input_df = SessionGen(daily_sessions, total_day, data_file)
input_df, input_real_df = input_df.generate_session()

print('this is post processed data', input_df)

env = simpy.Environment() # set up simpy environment
Sim = Simulator(daily_sessions=daily_sessions,
                total_day=total_day,
                input_df=input_df,
                input_real_df=input_real_df,
                number_of_pole=number_of_pole)

# # MAIN
env.process(Sim.run_simulation(env, endtime)) ### input_df
print("simu_run_tim", endtime)
print("env", env)
env.run(until=endtime + 10)
print(Sim.log.user_data)
Sim.log.process_station_power()
print(Sim.log.station_power)
Sim.log.compute_aggregate_metrics()
print(Sim.log.aggregate_metrics)
print('total_cost:', round(Sim.log.aggregate_metrics['utility cost'].sum(), 2), 'total_revenue', round(Sim.log.aggregate_metrics['revenue'].sum(), 2))
