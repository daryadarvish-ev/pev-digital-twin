
def append_choice(choice, new_user, current_user, user_choice, price, station, opt ):
    FLEX_user = list()  # reset the user lists
    ASAP_user = list()
    LEAVE_user = list()

    print(new_user)
    print(current_user)


    for user in current_user:
        if (user_choice['EV' + str(user)] == "Regular" ):
            ASAP_user.append('EV' + str(user))
        elif (user_choice['EV' + str(user)] == 'Scheduled'):
            FLEX_user.append('EV' + str(user))
        elif (user_choice['EV' + str(user)] == 'Leave'):
            LEAVE_user.append('EV' + str(user))
        else:
            None

    if (choice == "Regular"):
        ASAP_user.append('EV' + str(new_user))
        # station['EV' + str(new_user)].price = price
    elif (choice == 'Scheduled'):
        FLEX_user.append('EV' + str(new_user))
        # station['EV' + str(new_user)].price = price
    elif (choice == 'Leave'):
        LEAVE_user.append('EV' + str(new_user))
        # station['EV' + str(new_user)].price = price
    else:
        None

    station['FLEX_list'] = FLEX_user
    station['ASAP_list'] = ASAP_user
    station['LEAVE_list'] = LEAVE_user
    station['EV' + str(new_user)] = opt
    station['EV' + str(new_user)].price = price


    for ev in station['ASAP_list'].copy():
        if ev not in station:
            station['ASAP_list'].remove(ev)

    for ev in station['FLEX_list'].copy():
        if ev not in station:
            station['FLEX_list'].remove(ev)

    # Get a list of all EV numbers in the station dictionary
    station_ev_numbers = [int(key[2:]) for key in station.keys() if key.startswith("EV")]

    print(station_ev_numbers)
    # # Delete any keys in the station dictionary that are not in current_user
    # for ev_number in station_ev_numbers:
    #     if ev_number not in current_user:
    #         del station[f"EV{ev_number}"]
    #

    # for ev in station['ASAP_list']:
    #     print('this is ASAP_power', station[ev]["asap_powers"])

    # for ev in station['FLEX_list']:
    #     print('this is FLEX_price_obj', station[ev]["flex_powers"])