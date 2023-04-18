import random


def basic_choice_function(asap_price, flex_price):
    """Basic choice function which chooses the lowest price"""

    if random.uniform(0, 1) > 0.9:
        return "Leave", 9999
    if asap_price > flex_price:
        return "Scheduled", flex_price
    else:
        return "Regular", asap_price
