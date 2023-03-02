import optimizer_station as opt

import numpy as np

def main():

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

    par = opt.Parameters(z0 = np.array([25, 30, 1, 1]).reshape(4, 1),
                            Ts = delta_t,
                            eff = 1.0,
                            soft_v_eta = 1e-4,
                            opt_eps = 0.0001,
                            TOU = TOU_tariff)


    arrival_hour = 8
    duration_hour = 4
    e_need = 7.3

    ### Yifei: Also do we define the event here or in the optimizer?
    event = {
        "time": int(arrival_hour / delta_t),
        "e_need": e_need,
        "duration": int(duration_hour / delta_t),
        "station_pow_max": 6.6,
        "user_power_rate": 6.6,
        "limit_reg_with_sch": False,
        "limit_sch_with_constant": False,
        "sch_limit": 0,
        "historical_peak":4
    }

    prb = opt.Problem(par=par, event=event)

    # Yifei: The station object, here we assume no ongoing sessions. The form of this object is not decided yet. Dict or Class?
    station_info = None

    obj = opt.Optimization_station(par, prb, station_info, arrival_hour)
    station, res = obj.run_opt()

    reg_centsPerHr, sch_centsPerHr = res["reg_centsPerHr"], res['sch_centsPerHr']
    print(reg_centsPerHr, sch_centsPerHr )

if __name__ == "__main__":
    main()