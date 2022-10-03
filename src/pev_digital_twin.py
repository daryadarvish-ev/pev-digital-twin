# Import libraries

from datetime import datetime, date, time
from typing_extensions import runtime
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
# from google.colab import drive
import seaborn as sns
import pandas as pd
import numpy as np
import statistics
import warnings
import random
import simpy
import math
import os
from optimization import *

# drive.mount('/content/gdrive', force_remount=True)
# %cd /content/gdrive/MyDrive/capstone/Optimizer/
# import optimizer_v2 as optimizer

warnings.filterwarnings('ignore')
# %matplotlib inline
sns.set_theme()


# SIMULATION PARAMETERS
NUM_DAYS = 10 # Arbitrarily changed this
SIM_RUN_TIME = 1440*NUM_DAYS
CAR_ARR_TIME = 120
CAR_STAY_TIME = 300

sorted_events_q = []

#session output
session = []
#choice output
user_choice = []
#arrival time output
arrival_time = []
#departure time output
departure_time = []
#cartype output
car_type = []
#energy requested output
energyreq = []
#flex rate requested output
rate_flex = []
#asap rate requested output
rate_asap = []

models = ['Prius Prime','Model 3','Bolt EV', 'Kona','Model X 100 Dual','Clarity','Volt (second generation)','B-Class Electric Drive','Model S 100 Dual','Mustang Mach-E','Model S 90 Dual','Bolt EUV 2LT']

# import price lookup tables
# table_flex = pd.read_csv('/Users/areinaud/Desktop/table_flex.csv', index_col=0)
# table_asap = pd.read_csv('/Users/areinaud/Desktop/table_asap.csv', index_col=0)

def charger_station(env, input_df, run_time):
    
    user = 0
    next_leave_time = -1
    events_q = []

    # for plotting outputs
    leave_time_datalog = []
    num_flex = 0
    num_asap = 0
    num_leave_imm = 0
    num_leave_occ = 0

    # Time until first user arrives:
    yield env.timeout(input_df['arrivalMinGlobal'][0])
    # print('first step', str(env.now()))

    while True:
      
      # Car arrives 
      user += 1

      if user > input_df.shape[0]:
        break

      #cartype output        
      car = random.choices(models, weights = (151, 110, 86, 51, 42, 42, 28, 24, 20, 19, 15, 14))
      car_type.append(car)
      #session output        
      session.append(user)
      #arrival time output
      arrival_time.append(env.now)

      #energy asked by user (in Wh)    
      desired_energy = input_df['cumEnergy_Wh'][user-1]

      #energy required output
      energyreq.append(desired_energy)

      # print ("%s : Car %s arrived" % (str(env.now), user))
      events_q.append((int(env.now), "%s : Car %s arrived" % (str(env.now), user))) 
      inst_dept = int(env.now) + 1
    
      # generate stay duration
      stay_duration = input_df['durationMin'][user-1]
    
      print('curr_time = ', int(env.now))
      print('departure_time = ', int(env.now)+int(stay_duration))
      print('requested_energy = ', desired_energy)
    
      asap_price, flex_price = optimizer_main(curr_time=int(env.now), departure_time=int(env.now)+int(stay_duration), requested_energy=desired_energy, pwr_rate=6.6) #Divide by 1000 to have kWh

      #rates output
      charge_time = 30 * round((stay_duration)/30)
      #flex rate requested output
      rate_flex.append(flex_price)
      #asap rate requested output
      rate_asap.append(asap_price)

      
      if arrival_time[-1] > next_leave_time:
          leaveTime = int(env.now) + int(stay_duration)
          # print("leave time = ", leaveTime)
          choice = choice_function(asap_price, flex_price)
          if choice == 1:
            choice_name = 'SCHEDULED'
          elif choice == 2:
            choice_name ='REGULAR'
          else:
            choice_name = 'LEAVE'
          # print("choice = ", choice)
          #choice output
          user_choice.append(choice_name)


          if choice == 1:
              # print("User %s chose flex" % (user))
              events_q.append((arrival_time[-1]+3, "%s : User %s chose flex" % (arrival_time[-1]+3, user)))
              num_flex += 1  
              leave_time_datalog.append(stay_duration)
              next_leave_time = leaveTime
              # print ("%s : Car %s left" % (next_leave_time, user))
              events_q.append((next_leave_time, "%s : Car %s left" % (next_leave_time, user)))
#               yield env.timeout(3)
              #departure time output
              departure_time.append(next_leave_time)  
          elif choice == 2:
              # print("User %s chose ASAP" % (user))
              events_q.append((arrival_time[-1]+3, "%s : User %s chose ASAP" % (arrival_time[-1]+3, user))) 
              num_asap += 1
              
              leave_time_datalog.append(stay_duration)
              next_leave_time = leaveTime
              # print ("%s : User %s left" % (next_leave_time, user))
              events_q.append((next_leave_time, "%s : Car %s left" % (next_leave_time, user)))
#               yield env.timeout(3)
              #departure time output
              departure_time.append(next_leave_time) 

          elif choice == 3:
              #departure time output 
              departure_time.append(inst_dept) 
              
              # print("User %s chose to leave without charging" % (user))
              events_q.append((inst_dept, "%s : User %s chose to leave without charging" % (inst_dept, user))) 
              num_leave_imm += 1
#               yield env.timeout(3)

      else:

          #choice output
          choice_name = 'OCCUPIED'
          user_choice.append(choice_name)
          #departure time output 
          departure_time.append(inst_dept) # (env.now) 

          # print("User %s left because charger is occupied" % (user))
          events_q.append((inst_dept, "%s : User %s left because charger is occupied" % (inst_dept, user)))
          num_leave_occ += 1
#           yield env.timeout(3)
    
      # terminal segment
      if env.now >= run_time - 30 :
          events_q.sort(reverse = True)
          sorted_events_q = events_q
          
          # print timeline of events
          # while events_q:
          #     t = events_q.pop()
          #     print(t[1])
          
          # plot data
          x = np.array([x for x in range(len(leave_time_datalog))])

          figure(figsize = (10, 4))
          plt.plot(x, leave_time_datalog, linewidth=1)
          plt.xlabel("Number of arrivals")  # add X-axis label
          plt.ylabel("Stay duration")  # add Y-axis label
          plt.title("Number of Arrivals vs Stay duration")  # add title
          plt.show()

          fig = plt.figure()
          ax = fig.add_axes([0,0,1,1])
          choice = ['Flex', 'ASAP', 'Leave', 'Leave\n(occupied)']
          frequency = [num_flex, num_asap, num_leave_imm, num_leave_occ]
          ax.bar(choice, frequency)
          plt.xlabel("Agent Choice")  # add X-axis label
          plt.ylabel("Frequency of choice")  # add Y-axis label
          plt.title("Agent Choice vs Frequency")
          # plt.rcParams["figure.autolayout"] = True
          # wrap_labels(ax, 500)
          # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
          plt.show()
          
          break
    
          #yield env.timeout(random.randint(30,CAR_ARR_TIME)) # random interval for next arrival
      yield env.timeout(input_df['arrivalMinGlobal'][user] - input_df['arrivalMinGlobal'][user-1])

    return[session]
    # return[user_choice]
    # return[arrival_time]
    # return[departure_time]
    # return[car_type]
    # return[energyreq]
    # return[rate_flex]
    # return[rate_asap]


def choice_function(asap_price, flex_price):
    # print("leave time = ", leave_time)
    # print("curr time = ", curr_time)
#     charge_time = 30 * round((leave_time - curr_time)/30)
    # print("charge_time = ", charge_time)
    # print('charge level = ', charge_level)
    # read price from tables
    # print(table_asap)
#     asap_price = table_asap[str(desired_energy)][charge_time]
#     flex_price = table_flex[str(desired_energy)][charge_time]
    # choose lower price
    if asap_price > flex_price:
      choice = 1
    else: 
      choice = 2
    if random.uniform(0, 1) > 0.9:
      choice = 3
    return choice



def first_process(env, input_df, run_length):
    yield env.process(charger_station(env, input_df, run_length))
