import random
global Choice
Choice = []

def choice_function(asap_price, flex_price):
    # choose lower price
    if random.uniform(0, 1) > 0.9:
        choice = 3
        Choice.append("Leave")
        return Choice
    if asap_price > flex_price:
        choice = 1
        Choice.append("Scheduled")
        return Choice
    else:
        choice = 2
        Choice.append("Regular")
        return Choice