import random
import numpy as np

def random_leave(asap_price, flex_price, asap_quantiles, flex_quantiles):
    """Create a function to randomly assign the user to take a new option 'Leave'"""
    if min(asap_price, flex_price) < min(asap_quantiles[0.25], flex_quantiles[0.25]):
        leave_probability = 0.05
    elif min(asap_price, flex_price) < min(asap_quantiles[0.50], flex_quantiles[0.50]):
        leave_probability = 0.075
    elif min(asap_price, flex_price) < min(asap_quantiles[0.75], flex_quantiles[0.75]):
        leave_probability = 0.1
    else:
        leave_probability = 0.125

    leave = np.random.poisson(leave_probability)
    return 'Leave' if leave > 0 else None


def basic_choice_function(asap_price, flex_price):
    """Basic choice function which chooses the lowest price"""
    if random.uniform(0, 1) > 0.9:
        return "Leave", 9999
    if asap_price > flex_price:
        return "Scheduled", flex_price
    else:
        return "Regular", asap_price


