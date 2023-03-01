import random


def basic_choice_function(asap_price, flex_price):
    """Basic choice function which chooses the lowest price"""

    if random.uniform(0, 1) > 0.9:
        return "Leave"
    if asap_price > flex_price:
        return "Scheduled"
    else:
        return "Regular"
