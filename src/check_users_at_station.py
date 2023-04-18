import numpy as np


def CheckUsersAtStation(input_df, day, time):
    new_users = []
    current_users = []

    arrivals_day = input_df['arrivalDay'].values
    arrivals_min = input_df['arrivalMinGlobal'].values
    departure_min = input_df['departureMinGlobal'].values

    # find the indices where the condition is true
    indices = np.where((arrivals_day == day) & (arrivals_min == time))[0]

    indices2 = np.where((arrivals_day == day) & (arrivals_min <= time) & (time <= departure_min))[0]

    # append the users to the list
    new_users += list(indices + 1)
    current_users += list(indices2 + 1)

    # remove new_users that are already in current_users
    current_set = set(current_users)
    new_set = set(new_users)
    common_set = current_set.intersection(new_set)
    current_users = list(current_set - common_set)

    # return users
    return new_users, current_users