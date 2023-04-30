
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer

import warnings
warnings.filterwarnings("ignore")

class data_analyze:

    def __init__(self, choice_dict, price_dict, arrival_dict, departure_dict):
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
        plt.xlim(0, x_max + 50)
        plt.ylim(y_min - 1, y_max + 1)

        # Combine regular and scheduled data
        time = regular_arrival + scheduled_arrival
        price = regular_prices + scheduled_prices

        # Create a list of tuples containing (arrival_time, price) pairs
        data = list(zip(time, price))

        # Sort the data by arrival time
        data.sort(key=lambda x: x[0])

        # Separate the sorted data into arrival times and prices
        sorted_time, sorted_price = zip(*data)

        # Create the line plot for prices using only the specified range of data
        plt.plot(sorted_time, sorted_price, linestyle='-')

        # Add labels and legend to the plot
        plt.xlabel('Arrival Time (minutes)')
        plt.ylabel('Price (Cents($)/kWh)')
        plt.legend()

        # Show the plot
        plt.show()

    def revenue_calculate(self, price, energy, duration):
        revenue = (price * 0.01) * energy
        return revenue

    def total_revenue_calculate(self, price_dict, arr_dict, dep_dict, e_needed_dict, user_choice_dict):
        total_revenue = 0
        for key in user_choice_dict.keys():
            if user_choice_dict[key] == 'Regular' or user_choice_dict[key] == 'Scheduled':
                price = price_dict[key]
                energy = e_needed_dict[key]
                duration = dep_dict[key] - arr_dict[key]
                total_revenue += self.revenue_calculate(price, energy, duration)
        return total_revenue



    def usr_behavior_clf(self, data_path, analysis_start=1324):
        data = pd.read_csv(data_path)
        # Data after 1324th row is more stable and closer to current situation
        X = data[['vehicle_model',
                  'stationId',
                  'startChargeTime',
                  'reg_centsPerHr',
                  'sch_centsPerHr',
                  'sch_centsPerOverstayHr']][analysis_start:]
        # Create a KNNImputer with k=3
        imputer = KNNImputer(n_neighbors=3)

        # Fit the imputer and transform the dataset
        X[['reg_centsPerHr', 'sch_centsPerHr', 'sch_centsPerOverstayHr']] = imputer.fit_transform(X[['reg_centsPerHr',
                                                                                                     'sch_centsPerHr',
                                                                                                     'sch_centsPerOverstayHr']])
        # Convert categorical features into numerical values
        X['vehicle_model'] = X['vehicle_model'].astype('category').cat.codes
        # Convert 'startChargeTime' column to datetime
        X['startChargeTime'] = pd.to_datetime(X['startChargeTime'])
        # Convert 'startChargeTime' column to timestamp
        X['startChargeTime'] = X['startChargeTime'].apply(lambda x: x.timestamp())
        # Split the data into features (X) and labels (y)
        y = data['choice'][analysis_start:]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            # random_state=42
                                                            )

        # Define classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100,
                                                    # random_state=2023
                                                    ),
            'Logistic Regression': LogisticRegression(solver='liblinear',
                                                      # random_state=2023
                                                      ),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(
                # random_state=2023
            )
        }

        # Train and evaluate each classifier
        accuracies = {}
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[name] = accuracy
            print(f"{name}:")
            print(f"  Accuracy: {accuracy:.2f}")
            print("  Classification Report:")
            print(classification_report(y_test, y_pred))

        # Visualize classifier accuracies
        plt.figure(figsize=(10, 6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.xlabel('Classifiers')
        plt.ylabel('Accuracy')
        plt.title('Classifier Accuracies Comparison')
        plt.show()
        # for i in range(len(y_pred)):
        #     print(y_pred[i], y_test.reset_index(drop=True)[i])





