import pandas as pd


def check_pole(arrivalMinGlobal, departureMinGlobal, SIM_RUN_TIME, poles, occupied_pole_num, pole_number):
    n = pole_number
    YY = [str(i) for i in range(1, n + 1)]

    # Convert arrivalMinGlobal and departureMinGlobal to integers
    arrivalMinGlobal = int(arrivalMinGlobal)
    departureMinGlobal = int(departureMinGlobal)

    all_poles_occupied = True
    for pole_number in YY:
        if poles[pole_number][arrivalMinGlobal] != "OCCUPIED":
            all_poles_occupied = False
            break

    if all_poles_occupied:
        print("Every Poles are occupied at this moment")
        occupied_pole_num.append("unavailability leave")


    for num in YY:
        occupied_temp = (len(occupied_pole_num))
        if (poles[num][arrivalMinGlobal] != "OCCUPIED"):
            for time in range(arrivalMinGlobal,departureMinGlobal,1):
                poles[num][time] = "OCCUPIED"
            occupied_pole_num.append(num)
        # available_pole_num = available_pole_num - len(occupied_pole_num)
        if (len(occupied_pole_num) == ((occupied_temp) + 1)):
            break
        else:
            continue


    return poles, occupied_pole_num
