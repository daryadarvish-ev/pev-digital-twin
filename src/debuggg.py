# %%
# import modules
import datetime #, timedelta
import time
import json
import pandas as pd
import numpy as np
import decimal
from matplotlib import pyplot as plt
import seaborn as sns
import math
import scienceplots
import matplotlib 

# from plotly.subplots import make_subplots
import plotly.graph_objects as go

import boto3
from boto3.dynamodb.conditions import Key, Attr

# Obtain secret key for AWS
f = open("secret.txt")
text = f.readlines()

# Access dynamodb on AWS
dynamodb = boto3.resource(
    "dynamodb",
    aws_access_key_id=text[0][:-1],
    aws_secret_access_key=text[1][:-1],
    region_name="us-east-2",
)

    
plt.style.use(['science','ieee','high-vis'])
# plt.rcParams['axes.linewidth'] = 1
# plt.rcParams['figure.dpi'] =300

w = 3.3 
h = 2.5

plt.rcParams['figure.constrained_layout.use'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
high_vis_map = ["#0d49fb", "#e6091c", "#26eb47", "#8936df", "#fec32d", "#25d7fd"]

# %%
# Scan through all Sessions2 data to get session items
table = dynamodb.Table('Sessions2')
params = {'ProjectionExpression': "dcosId,userId,user_email,vehicle_model,vehicle_maxChgRate_W,siteId,stationId,connectTime,startChargeTime,Deadline,energyReq_Wh,estCost,reg_centsPerHr,sch_centsPerHr,sch_centsPerKwh,sch_centsPerOverstayHr,#Dur,DurationHrs,choice,regular,scheduled,cumEnergy_Wh,peakPower_W,power,lastUpdate",
          'ExpressionAttributeNames': {"#Dur":"Duration"}}

# Repeat scan until LastEvaluatedKey is None
start = time.time()
done = False
start_key = None
temp = []
while not done:
    if start_key:
        params['ExclusiveStartKey'] = start_key
    response = table.scan(**params)
    temp.extend(response.get('Items', []))
    print("Length of Scanned Items is {0} items".format(len(temp)))
    start_key = response.get('LastEvaluatedKey', None)
    done = start_key is None

# Print elapsed time
end = time.time()
elapsed_time_min = np.floor((end-start)/60)
elapsed_time_sec = (end-start) % 60
elapsed_time = "Elapsed time: %d minutes, %d seconds" % (elapsed_time_min, elapsed_time_sec)
print(elapsed_time)

# Create dataframe
session_df = pd.DataFrame(temp)
del temp


# %% [markdown]
# ## Expected Energy Demand, Duration

# %%
session_df['connectTime']  = pd.to_datetime(session_df['connectTime'] )
session_df['startChargeTime']  = pd.to_datetime(session_df['startChargeTime'] )
session_df['Deadline']  = pd.to_datetime(session_df['Deadline'] )
session_df['lastUpdate']  = pd.to_datetime(session_df['lastUpdate'])
session_df = session_df.sort_values(by='connectTime')
session_df = session_df[session_df['siteId'] == 25]
session_df['interArrivalTime_min'] = session_df['connectTime'].diff().dt.seconds / 60
plot_df = session_df[session_df['connectTime'].dt.year >= 2022]
session_df['arrHour']=session_df['connectTime'].dt.hour

session_df['cumEnergy_Wh']=session_df['cumEnergy_Wh'].astype(float)
session_df['cumEnergy_KWh']=session_df['cumEnergy_Wh']/1000
session_df['DurationHrs'] = session_df['DurationHrs'].astype(float)

# %%
# 1. peakPower_W = 0 and cumEnergy_W = 0: delete  
session_df = session_df[(session_df["peakPower_W"]!=0) & (session_df["cumEnergy_Wh"]!=0)]

# 2. Fix the user peak_powers 
# some users have historical peak power > 6.6 / 7 
user_df = session_df[['userId','peakPower_W','vehicle_maxChgRate_W']].groupby('userId').max()
user_df['session_counts'] = session_df[['userId','peakPower_W']].groupby('userId').count()

validate_users = user_df[(user_df['session_counts']==1) & (user_df['peakPower_W']<=6000) ].index.to_list()
user_df.reset_index(drop=False,inplace=True)

user_df.rename(columns={"peakPower_W":'historical_peakPower_w'},inplace=True)
session_df = session_df.merge(user_df[['userId','historical_peakPower_w']])

# %%
optimizer_time = pd.Timestamp(2023, 2, 7, 8)
session_df['endTime'] = session_df['startChargeTime'] + pd.to_timedelta(session_df['Duration'])
cols = ['connectTime','choice','power','endTime','Duration', 'userId', 'Deadline', 'startChargeTime','DurationHrs','dcosId', 'lastUpdate' ]
session_df[(session_df['startChargeTime']>optimizer_time)].sort_values(by='connectTime')[cols]

# %%
high_power_idx = list(session_df[session_df['historical_peakPower_w'] >= 5000].index)
low_power_idx = list(session_df[session_df['historical_peakPower_w'] < 5000].index)

session_df['highPower'] = pd.Series(dtype=int)

session_df.loc[high_power_idx ,'highPower'] = 1
session_df.loc[low_power_idx ,'highPower'] = 0

session_df.loc[high_power_idx ,'userPower_kW'] = 6.6
session_df.loc[low_power_idx ,'userPower_kW'] = 3.3

# %%
plt.rcParams['figure.dpi'] =300

# %%
fig,ax= plt.subplots(2,1,figsize=( w*2, h*1.5 ))

sns.boxplot(
    data=session_df, x="arrHour", y="DurationHrs",hue='highPower',
    notch=True, showcaps=True,
    flierprops={"marker": "x"},
    medianprops={"color": "coral"},ax=ax[0]
)

sns.boxplot(
    data=session_df, x="arrHour", y="cumEnergy_KWh",hue='highPower',
    notch=True, showcaps=True,
    flierprops={"marker": "x"},
    medianprops={"color": "coral"},ax=ax[1]
)

for i in [0,1]:
    ax[i].grid(alpha=0.3)
    ax[i].set_xlabel("Arrival Hour")

# %%
iterables = [[0,1], range(0,24)]
idx = pd.MultiIndex.from_product(iterables, names=['highPower','arrHour'])
# pd.DataFrame(index=idx,columns = ['DurationHrs', 'cumEnergy_Wh', 'interArrivalTime_min', 'arrivalHour','cumEnergy_KWh', 'count'])
expected_demand = session_df.groupby(['highPower','arrHour']).mean()
expected_demand['count'] = session_df.groupby(['highPower','arrHour']).count()['dcosId']
expected_demand = expected_demand.reindex(idx).reset_index(drop=False)
expected_demand.rename(columns = {'cumEnergy_Wh':"estEnergyWh", 
                                  'cumEnergy_KWh':"estEnergykWh", 
                                  "DurationHrs":"estDurationHrs"},inplace=True)


expected_demand['count'].fillna(0,inplace=True)
# expected_demand.fillna(method='ffill',inplace=True)

# %%
for i in range(0, len(expected_demand), 24):
    group = expected_demand.iloc[i:i+24]
    # if group.iloc[0].isna().any():
    #     group.iloc[0].fillna(method='bfill', inplace=True)
    group.fillna(method='ffill', inplace=True)
    group.fillna(method='bfill', inplace=True)

# %%
import optimizer_station as opt
# We define the timesteps in the APP as 15 minute 
delta_t = 0.25 #hour 
print("For delta_t: ",delta_t, "max number of intervals:",24/delta_t)
################## Define the TOU Cost ##########################################
## the TOU cost is defined considering the delta_t above, if not code raises an error.##

# off-peak 0.175  cents / kwh 
TOU_tariff = np.ones((96,)) * 17.5
## 4 pm - 9 pm peak 0.367 cents / kwh 
TOU_tariff[64:84] = 36.7
## 9 am - 2 pm super off-peak 0.49 $ / kWh  to cents / kwh
TOU_tariff[36:56] = 14.9

# %%
optimizer_time = pd.Timestamp(2023, 3, 6, 7, 0, 0)
print(optimizer_time)
print(pd.to_datetime(optimizer_time).value)

# %%
def unixTime(timeStamp):
    return int(pd.to_datetime(timeStamp).timestamp())

def dummyUserPower(choice):
    
    """ Returns Array but this can also be a dictionary """

    ### Yifei: In the optimizer, we require strictly that HIGH power corresponds to 6.6 kW and LOW power corresponds to 3.3 kW.
    ### Aka, for REG, we require the first several intervals to be 6.6 kW (must satisfy, cannot be other values like 6, 6, 6...)
    
    ts = pd.date_range(start = pd.Timestamp(2023, 3, 6, 7, 45, 0), periods=10, freq="15min")
    ts = [unixTime(t) for t in ts]
    
    p0 = 6600
    p1 = 6600
    p2 = 6600
    p3 = 6600
    p4 = 6600
    p5 = 3300

    if choice == "REG":
        
        ## Can you clarify up to what are we recording for the REG option? Is it the N_ASAP? 

        ## Yifei: Yes, the length of REG_powers should be the same with N_ASAP.
        powers = np.array([[ts[0], p0], 
                           [ts[1], p1], 
                           [ts[2], p2], 
                           [ts[3], p3], 
                           [ts[4], p4],
                           [ts[5], 3300]])

    elif choice == "SCH":
        powers = np.array([ [ts[0], 0], [ts[1], p1], [ts[2], p2], [ts[3], p3], [ts[4], p4],
                           [ts[5], 6600], [ts[6], 0], [ts[7], 0], [ts[8], 0], [ts[9], 0]])
    else:
        powers = np.zeros(shape=(6,2))
    return powers

def dummyStateRecord():
    """ 
    Returns: List of Dictionaries
    Each entry is a state dictionary with keys: 
    
    monthlyPeak, int
    timeStamp, int
    sessions, list of dictionaries with keys
    dcosId, int
    choice, string
    powerRate, string
    energyNeeded, float
    deadline, int
    optPower, array 
        """
    
    stateRecord = [
        {"monthlyPeak":18, 
         "timeStamp":unixTime(pd.Timestamp(2023, 3, 6, 7, 45, 0)), ## Last record TS(decision of the last vehicle)
         "sessions": [ 
             {
                "dcosId" : 1001,
                "choice": "SCH",
                "powerRate": "HIGH",
                "energyNeeded" : 8250,
                "deadline" : unixTime(pd.Timestamp(2023, 3, 6, 9, 45, 0)), # So here the values are all discretized to 15 min intervals?
                "optPower" : dummyUserPower("SCH") 
             }, 
             {
                "dcosId" : 1002,
                "choice": "REG",
                "powerRate": "HIGH",
                "energyNeeded" : 9075,
                "deadline" : unixTime(pd.Timestamp(2023, 3, 6, 10, 0, 0)),
                "optPower" : dummyUserPower("REG") 
             }, 
         ]
        }
    ]
    print(stateRecord[0]["sessions"][0]["optPower"][:,1].sum() * 0.25)
#     assert  == 3750
    
    return stateRecord

def arrHourList(arrHour, optHorizon):
    """ arrHour, int: current optimization hour
        optHorizon, int: how long to optimize in hours """ 
    if arrHour <= (24-optHorizon):
        overnight = [False for _ in range(optHorizon)]
        return list(range(arrHour,(arrHour+optHorizon))), overnight
    else: 
        lst = list(range(arrHour,24))
        overnight = [False for _ in range(24-arrHour)]
        lst.extend(list(range(0 , ((arrHour+optHorizon)-24) )))
        overnight.extend([True for _ in range((arrHour+optHorizon)-24)])
        ## Overnight charging: like [23, 0, 1, 2]

        return lst, overnight

# %% [markdown]
# ## Input format convertion(from raw input data to the format of the optimizer)

# %%
import datetime
import copy
def data_format_convertion(stateRecords, opt_hour, delta_t, overnight_label, optTime):
    timezone = datetime.timezone(datetime.timedelta(hours=0))
    stateRecord = copy.deepcopy(stateRecords[0]["sessions"])
    if not stateRecord:
        return None
    # num_users = len(stateRecord)
    res = []
    for user in stateRecord:
        try: 
            user["optPower"] = user["optPower"].tolist()
        except AttributeError: 
            user["optPower"] = user["optPower"]
            
        user["power_rate"] = 6.6 if user["powerRate"] == "HIGH" else 3.3

        start_time_obj = datetime.datetime.fromtimestamp(int(user["optPower"][0][0]), timezone) # the timestamp of the first time slot
        user["start_time"] = float(start_time_obj.hour + start_time_obj.minute / 60)

        end_time_obj = datetime.datetime.fromtimestamp(int(user["optPower"][-1][0]), timezone) # Or retrieve the last time slot??
        user["end_time"] = float(end_time_obj.hour + end_time_obj.minute / 60) + delta_t
        # unix end time: int(user["optPower"][-1][0]) + 60 * 60 * delta_t
        unix_end_time = int(user["optPower"][-1][0] + 60 * 60 * delta_t)
        opt_time = optTime
        if overnight_label:
            opt_time += datetime.timedelta(days=1)
            opt_time = opt_time.replace(hour=opt_hour, minute=0, second=0, microsecond=0)
        else:
            opt_time = opt_time.replace(hour=opt_hour, minute=0, second=0, microsecond=0)
        unix_opt_time = unixTime(opt_time)

        if unix_end_time <= unix_opt_time:
            continue

        user["optPower"] = np.round(np.array([x[1] for x in user["optPower"]]) / 1000, 2)
        user["price"] = 25 if user["choice"] == "SCH" else 30
        user["energyNeeded"] = float(user["energyNeeded"]) / 1000
        del(user["deadline"])
        res.append(user)

    return res

def convertOutput(stateRecords, station_info, res, hour, user_choice, optTime):
    """ Convert the output to the original format """
    new_state = copy.deepcopy(stateRecords[0])

    new_state["monthlyPeak"] = round(res["new_peak_sch"][0], 2) if user_choice == "SCH" else round(res["new_peak_reg"][0], 2)

    # Update the timestamp. Tugba: What should be the timestamp here?
    if optTime.hour > hour: # If it is overnight.
        new_TimeStamp = optTime + datetime.timedelta(days=1)
        new_TimeStamp = new_TimeStamp.replace(hour=hour, minute=0, second=0)
    else:
        new_TimeStamp = optTime.replace(hour=hour, minute=0, second=0)
    new_state["timeStamp"] = unixTime(new_TimeStamp)

    # new_state["timeStamp"] = unixTime(pd.Timestamp(2023, 3, 6, hour, 0 , 0))
    finishing_list = []
    if new_state["sessions"]:
        for i, user in enumerate(new_state["sessions"]):
            # Changed this because we need an array for code to not break.. 
            # We can agree on a convention later
            user["optPower"] = recover_json_serialized_power_array(user["optPower"])
            timezone = datetime.timezone(datetime.timedelta(hours=0))
            end_time_obj = datetime.datetime.fromtimestamp(int(user["optPower"][-1][0]), timezone) # Or retrieve the last time slot??
            end_time = float(end_time_obj.hour + end_time_obj.minute / 60) + delta_t
            if end_time <= hour:
                finishing_list.append(i)
                continue
            user_update = [d for d in station_info if d["dcosId"] == user["dcosId"]][0] # The updated user info from opt output
            TOU_idx = user_update["TOU_idx"]
            user["optPower"][TOU_idx:, 1] = (np.ceil(user_update["optPower"][TOU_idx:] * 1000)).astype(int) # Retaining the UNIXTIME and updating the power
    if finishing_list:
        # Remove the index in finishing_list from new_state["sessions"]
        new_state["sessions"] = [user for i, user in enumerate(new_state["sessions"]) if i not in finishing_list]
    new_user = dict()
    new_user["dcosId"] = "dummyUser"
    new_user["choice"] = user_choice  # This choice and OPT power / price are decided outside the optimizer
    new_user["powerRate"] = "HIGH" if res["power_rate"] == 6.6 else "LOW"
    new_user["energyNeeded"] = int(1000 * res['e_need'])
    new_user["optPower"], new_state["aggPower"], new_user["deadline"] = powerOutput(res, user_choice, optTime)
    ## TO-DO: How to get the deadline? Is it the rounded and discretized time or the actual time? For example, 8:45(Rounded) / 8:47(Actual)?
    new_state["sessions"].append(new_user)

    return new_state

def powerOutput(res, user_choice, optTime):
    """ Convert the output to the original format """
    start_timestamp = res["time_start"] * delta_t
    start_timestamp_hour = int(start_timestamp)
    start_timestamp_minute = int((start_timestamp % 1) * 60)
    start_time = optTime.replace(hour=start_timestamp_hour, minute=start_timestamp_minute, second=0) # Start_time == optTime here.
    # start_time = datetime.datetime(2023, 3, 6, start_timestamp_hour, start_timestamp_minute, 0)
    t0 = unixTime(start_time)

    end_timestamp = res["time_end_SCH"] * delta_t if user_choice == "SCH" else res["time_end_REG"] * delta_t
    end_timestamp_hour = int(end_timestamp)
    end_timestamp_minute = int((end_timestamp % 1) * 60)
    if end_timestamp_hour >= 24:
        end_timestamp_hour -= 24
        end_time = optTime + datetime.timedelta(days=1)
        end_time = end_time.replace(hour=end_timestamp_hour, minute=end_timestamp_minute, second=0)
    else:
        end_time = optTime.replace(hour=end_timestamp_hour, minute=end_timestamp_minute, second=0)
    # end_time = datetime.datetime(2023, 3, 6, end_timestamp_hour, end_timestamp_minute, 0)
    t1 = unixTime(end_time)

    timestamps = np.arange(t0, t1, delta_t * 60 * 60).astype(int)   # In seconds, for example, 0.25 * 60 = 15min in seconds
    optPower = copy.deepcopy(res["sch_powers"]) * 1000 if user_choice == "SCH" else copy.deepcopy(res["reg_powers"]) * 1000
    optPower = optPower.astype(int)
    output_power = np.concatenate((timestamps.reshape(-1, 1), optPower.reshape(-1, 1)), axis=1)

    ## Here the max_timestamp is used to calculate the aggregated power(decide which interval has power)
    max_timestamp = res["time_max"] * delta_t
    max_timestamp_hour = int(max_timestamp)
    max_timestamp_minute = int((max_timestamp % 1) * 60)
    if max_timestamp_hour >= 24:
        max_timestamp_hour -= 24
        max_time = optTime + datetime.timedelta(days=1)
        max_time = max_time.replace(hour=max_timestamp_hour, minute=max_timestamp_minute, second=0)
    else:
        max_time = optTime.replace(hour=max_timestamp_hour, minute=max_timestamp_minute, second=0)
    # max_time = datetime.datetime(2023, 3, 6, max_timestamp_hour, max_timestamp_minute, 0)
    t2 = unixTime(max_time)

    timestamps_agg = np.arange(t0, t2, delta_t * 60 * 60).astype(int)
    aggPower = copy.deepcopy(res["sch_agg"]) * 1000 if user_choice == "SCH" else copy.deepcopy(res["reg_agg"]) * 1000
    aggPower = aggPower.astype(int)
    output_power_agg = np.concatenate((timestamps_agg.reshape(-1, 1), aggPower.reshape(-1, 1)), axis=1)

    return output_power, output_power_agg, t1

# %% [markdown]
# 
# ## Generate the expected demand price table

# %%
from collections import defaultdict

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def recover_json_serialized_power_array(json_serialized_item):
    return np.asarray(json_serialized_item)
    
def generateOptPricePowerFromDummyInput(expectedDemand, optTime):
    
    unixOptTime = unixTime(optTime)  ## Convert the optTime to unix time
    arrival_hour = optTime.hour

    # We only take the next 4 hours of the price table

    ## read the stateRecords last entry
    stateRecords = dummyStateRecord()

    try:
        expected_demand.set_index(['arrHour','highPower'], inplace=True)
    except KeyError:
        print("")
        
    States, expectedDemand = generateOptPricePower(stateRecords, expectedDemand, optTime, arrival_hour)
        
    return States, expectedDemand
    

def generateOptPricePower(stateRecords, 
                          expectedDemand,
                          optTime,
                          arrival_hour,
                          opt_horizon=4,):
    """
    Generate the expected demand price table
    optTime: The time when the re-optimization is triggered. pd.Timestamp.
    arrival_hour: The hour of optTime
    """

    States = defaultdict(dict)
    optHours, overnight = arrHourList(arrival_hour, optHorizon=opt_horizon)

    ## read the expected demand table
    for highPower in [0,1]:
        for i, hour in enumerate(optHours):

            ## Here we are converting the optimization time to the arrival time
            hr = optTime.hour
            minute =  optTime.minute / 60

            arrival_time = hr + minute
            duration_hour = expected_demand.loc[(hour, highPower), 'estDurationHrs']
            e_need = expected_demand.loc[(hour, highPower), 'estEnergykWh']

            event = {
                "time": int(hour / delta_t), # Hour or Arrival_hour?
                "e_need": e_need,
                "duration": int(duration_hour / delta_t),
                "station_pow_max": 6.6,
                "user_power_rate": expected_demand.loc[(hour, highPower), 'userPower_kW'],
                "limit_reg_with_sch": False,
                "limit_sch_with_constant": False,
                "sch_limit": 0,
                "historical_peak": 10
            }

            stateRecord = data_format_convertion(stateRecords, hour, delta_t, overnight[i], optTime)
            par = opt.Parameters(z0 = np.array([30, 30, 1, 1]).reshape(4, 1),
                         Ts = delta_t,
                         eff = 1.0,
                         soft_v_eta = 1e-4,
                         opt_eps = 0.0001,
                         TOU = TOU_tariff,
                         demand_charge_cost=12)

            prb = opt.Problem(par=par, event=event)

            obj = opt.Optimization_station(par, prb, stateRecord, hour)
            station_info, res = obj.run_opt()

            # One of the indexes power level 
            # index hierarchy: hour -> powerLevel -> SCH 
            # json.dumps(States['hour8']['SCH'])
            States["hour" + str(hour) + "-" + str(highPower)]["SCH"] = convertOutput(stateRecords, station_info, res, hour, "SCH", optTime)
            States["hour" + str(hour) + "-" + str(highPower)]["REG"] = convertOutput(stateRecords, station_info, res, hour, "REG", optTime)

            ### Heyy so this is for US to be able to test. We won't do this type of output in the server
            # However we will send the Right hand side of the below lines.
            # RHS should be a state record, in the server it will be indexed by hour and power and choice
            expected_demand.loc[(hour, highPower), 'SCH_expected_power_W']= json.dumps(States["hour" + str(hour) + "-" + str(highPower)]['SCH'],
                                                                                 cls=NumpyEncoder)

            expected_demand.loc[(hour, highPower), 'REG_expected_power_W']= json.dumps(States["hour" + str(hour) + "-" + str(highPower)]['REG'],
                                                                                 cls=NumpyEncoder)

            ### How do we update the expected demand table? Make a slice for 4 hours or take the whole?
            reg_centsPerHr, sch_centsPerHr = res["reg_centsPerHr"], res['sch_centsPerHr']
            expected_demand.loc[(hour, highPower), 'sch_centsPerHr'] = sch_centsPerHr
            expected_demand.loc[(hour, highPower), 'reg_centsPerHr'] = reg_centsPerHr


    return States, expectedDemand

# %%
## Dummy inputs 
cols =  ['sch_centsPerHr','reg_centsPerHr',
        'connectTime','Duration', 'Deadline', 'startChargeTime', 
     'sch_centsPerKwh','sch_centsPerOverstayHr', 'choice','DurationHrs',
       'dcosId', 'lastUpdate', 'energyReq_Wh','cumEnergy_Wh',  
         'interArrivalTime_min','arrHour', 'cumEnergy_KWh', 
         'historical_peakPower_w','userPower_kW','highPower']

optTime = pd.Timestamp(2023, 3, 6, 8, 0, 0)

subset = session_df[session_df['startChargeTime'] > optTime][cols].sort_values(by="startChargeTime")
subset.reset_index(drop=False,inplace=True)

_, expected_demand = generateOptPricePowerFromDummyInput(expected_demand, 
                                          optTime = pd.Timestamp(2023, 3, 6, 8, 0, 0))

# %%
def get_new_state(expected_demand, new_session_start):
    new_State = expected_demand.loc[(new_session_start.hour, 0),
                                   "SCH_expected_power_W"]
    new_State = json.loads(new_State)

    for session in new_State['sessions']:
        if session['dcosId'] == 'dummyUser':
            new_State['sessions'].remove(session)

        if session['deadline'] <= unixTime(new_session_start):
            print(pd.to_datetime(session['deadline'])," <= ",unixTime(new_session_start))
            print("removed session:",session['dcosId']) ## Check if it is removing correct or wrong session!!
            new_State['sessions'].remove(session)
    return new_State

# %%
# Real inputs

## Row: 0, 9, 17
last_opt_time = pd.Timestamp(2023, 3, 6, 8, 0, 0)


for row in subset.index[:]:
    new_session_start = subset.loc[row,"startChargeTime"]
    new_session_choice = subset.loc[row,"choice"]
    new_session_id = subset.loc[row,"dcosId"]
    new_session_power = subset.loc[row,"highPower"]
    
    arrHour = new_session_start.hour
    curr_opt_time = new_session_start.replace(minute=0, second=0)
    opt_horizon = 4
    while curr_opt_time - last_opt_time > pd.Timedelta(hours=opt_horizon):
        new_State = get_new_state(expected_demand, last_opt_time)
        last_opt_time += pd.Timedelta(hours=opt_horizon) # Re-optimize every 4 hours
        stateRecords = [new_State] # States at the last opt hour
        States, expected_demand = generateOptPricePower(stateRecords,
                              expected_demand,
                              last_opt_time,
                              last_opt_time.hour)

    arrHour = new_session_start.hour
    new_State = expected_demand.loc[(new_session_start.hour, new_session_power),
                                   new_session_choice[:3]+ "_expected_power_W"]

    new_State = json.loads(new_State)

    for session in new_State['sessions']:
        if session['dcosId'] == 'dummyUser':
            session['dcosId'] = str(new_session_id)

        if session['deadline'] <= unixTime(new_session_start):
            print(pd.to_datetime(session['deadline'])," <= ",unixTime(new_session_start))
            print("removed session:",session['dcosId']) ## Check if it is removing correct or wrong session!!
            new_State['sessions'].remove(session)


    ### Identify the users at the Station
    optTime = new_session_start

    unixOptTime = unixTime(optTime)
    arrival_hour = optTime.hour

    # We only take the next 4 hours of the price table

    optHours = arrHourList(arrival_hour, optHorizon=4)
    ## read the stateRecords last entry
    stateRecords = [new_State]

    try:
        expected_demand.set_index(['arrHour','highPower'], inplace=True)
    except KeyError:
        print("")
    
    States, expected_demand = generateOptPricePower(stateRecords, 
                              expected_demand,
                              optTime, 
                              arrival_hour)
    last_opt_time = optTime


