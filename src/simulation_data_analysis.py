import matplotlib.pyplot as plt
import numpy as np


class data_analyze:

    def __init__(self, choice_dict, price_dict, arrival_dict , departure_dict ):
        self.choice = choice_dict
        self.price = price_dict
        self.arrival = arrival_dict
        self.departure = departure_dict

    # Convert dictionary to lists of keys and values
    def analysis(self):

        # Count the number of regular, scheduled, and leave users
        regular_count = 0
        scheduled_count = 0
        leave_count = 0

        for user, status in self.choice.items():
            if status == 'Regular':
                regular_count += 1
            elif status == 'Scheduled':
                scheduled_count += 1
            else:
                leave_count += 1

        print(f"Regular users: {regular_count}")
        print(f"Scheduled users: {scheduled_count}")
        print(f"Leave users: {leave_count}")

        # Compare prices for regular users
        regular_prices = []
        for user, status in self.choice.items():
            if status == 'regular':
                regular_prices.append(self.price[user])

        # if regular_count > 0:
        #     print(f"Min price for regular users: {min(regular_prices)}")
        #     print(f"Max price for regular users: {max(regular_prices)}")
        # else:
        #     print("There are no regular users.")

        # Plot the data as a bar graph
        x = ['Regular', 'Scheduled', 'Leave']
        y = [regular_count, scheduled_count, leave_count]

        for i, v in enumerate(y):
            plt.text(i, v + 0.5, str(v), ha='center')

        plt.bar(x, y)
        plt.title("User Status")
        plt.xlabel("Status")
        plt.ylabel("Number of Users")
        plt.show()

    def plot_generation(self):
        # Create lists to store data
        regular_prices = []
        scheduled_prices = []
        regular_arrival = []
        scheduled_arrival = []
        arrival_times = []
        ev_labels = []

        # Loop through each EV in the input dictionaries
        for ev, price in self.price.items():
            choice = self.choice.get(ev, None)
            arrival = self.arrival[ev]
            ev_labels.append(ev)

            # Add data to the appropriate list based on the charging option
            if choice == 'Regular':
                regular_prices.append(price)
                regular_arrival.append(arrival)
            elif choice == 'Scheduled':
                scheduled_prices.append(price)
                scheduled_arrival.append(arrival)

        # Set the x-axis limits to the user's arrival time
        x_min = min(min(regular_arrival), min(scheduled_arrival))
        x_max = max(max(regular_arrival), max(scheduled_arrival))

        # Set the y-axis limits to the range of regular prices
        y_min = min(min(regular_prices), min(scheduled_prices)) - 0.1
        y_max = max(max(regular_prices), max(scheduled_prices)) + 0.1

        # Create the dot plot for regular prices
        plt.scatter(regular_arrival, regular_prices, label='Regular')

        # Create the dot plot for scheduled prices
        plt.scatter(scheduled_arrival, scheduled_prices, label='Scheduled')

        # Set the x-axis and y-axis limits
        plt.xlim(x_min-10, x_max+10)
        plt.ylim(y_min-1, y_max+1)

        # Add labels and legend to the plot
        plt.xlabel('Arrival Time (minutes)')
        plt.ylabel('Price (Cents($)/kWh)')
        plt.legend()

        # Show the plot
        plt.show()






