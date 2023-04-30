import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt


# Read the CSV file and convert the 'connectTime' column to datetime
df = pd.read_csv('../data/Sessions2_20221020.csv')
df['connectTime'] = pd.to_datetime(df['connectTime'])

# Create a new column 'season' based on the month
seasons = {
    'Spring': [[(3, 1), (5, 31)]],
    'Summer': [[(6, 1), (8, 31)]],
    'Fall': [[(9, 1), (11, 30)]],
    'Winter': [[(12, 1), (12, 31)], [(1, 1), (2, 28)]]
}

def get_season(dt):
    for season, date_ranges in seasons.items():
        for date_range in date_ranges:
            (start_month, start_day), (end_month, end_day) = date_range
            if start_month <= dt.month <= end_month and start_day <= dt.day <= end_day:
                return season
    return None

df['season'] = df['connectTime'].apply(get_season)

# Create a new column 'day_of_week' based on the day of the week
def get_day_of_week(dt):
    if dt.weekday() < 5:
        return 'Weekday'
    else:
        return 'Weekend'

df['day_of_week'] = df['connectTime'].apply(get_day_of_week)

def get_total_users(season, day_of_week):
    if day_of_week == 'Week':
        return df[df['season'] == season].shape[0]
    else:
        return df[(df['season'] == season) & (df['day_of_week'] == day_of_week)].shape[0]

def count_days_in_season(season, day_type):
    days = 0
    for date_range in seasons[season]:
        start_date = date(2022, *date_range[0])
        end_date = date(2022, *date_range[1])

        while start_date <= end_date:
            if day_type == 'Week':
                days += 1
            elif day_type == 'Weekday' and start_date.weekday() < 5:
                days += 1
            elif day_type == 'Weekend' and start_date.weekday() >= 5:
                days += 1
            start_date += timedelta(days=1)

    return days

# Loop over seasons and day_types
# Loop over seasons and day_types
seasons_list = ['Spring', 'Summer', 'Fall', 'Winter', 'All season']
day_types = ['Weekday', 'Weekend', 'Week']

for season in seasons_list:
    if season == 'All season':
        for day_type in day_types:
            total_users = get_total_users('Spring', day_type) + get_total_users('Summer', day_type) + \
                          get_total_users('Fall', day_type) + get_total_users('Winter', day_type)
            total_days = count_days_in_season('Spring', day_type) + count_days_in_season('Summer', day_type) + \
                         count_days_in_season('Fall', day_type) + count_days_in_season('Winter', day_type)
            average_users = total_users / total_days if total_days > 0 else 0
            print(f"Average number of users during All seasons {day_type}: {average_users:.2f}")
    else:
        for day_type in day_types:
            total_users = get_total_users(season, day_type)
            total_days = count_days_in_season(season, day_type)
            average_users = total_users / total_days if total_days > 0 else 0
            print(f"Average number of users during {season} {day_type}: {average_users:.2f}")

plot_data = []
for season in seasons_list:
    season_data = []
    for day_type in day_types:
        if season == 'All season':
            total_users = get_total_users('Spring', day_type) + get_total_users('Summer', day_type) + \
                          get_total_users('Fall', day_type) + get_total_users('Winter', day_type)
            total_days = count_days_in_season('Spring', day_type) + count_days_in_season('Summer', day_type) + \
                         count_days_in_season('Fall', day_type) + count_days_in_season('Winter', day_type)
        else:
            total_users = get_total_users(season, day_type)
            total_days = count_days_in_season(season, day_type)

        average_users = total_users / total_days if total_days > 0 else 0
        season_data.append(average_users)
    plot_data.append(season_data)

# Create the bar plot
fig, ax = plt.subplots()
x = range(len(seasons_list))

# Set the width of the bars
width = 0.25

# Plot the data for each day type
for i, day_type in enumerate(day_types):
    ax.bar([elem + width * i for elem in x], [data[i] for data in plot_data], width, label=day_type)

# Customize the plot
ax.set_xticks([elem + width for elem in x])
ax.set_xticklabels(seasons_list)
ax.set_ylabel('Average Users')
ax.set_title('Average Users by Season and Day Type')
ax.legend()

# Show the plot
plt.show()
