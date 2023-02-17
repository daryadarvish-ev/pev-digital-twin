import csv
from datetime import datetime, timedelta
from math import ceil

class CAISOData:
    def __init__(self, csv_file_path):
        self.data = []
        with open(csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Parse date and time string to datetime object
                row_time = datetime.strptime(row['date'], '%m/%d/%Y %I:%M:%S %p')
                # Round up to nearest 15-minute interval
                row_time += timedelta(minutes=15 - row_time.minute % 15)
                row['date'] = row_time
                row['price'] = float(row['price'])
                self.data.append(row)

    def get_price_at_time(self, target_time):
        # Round up target time to nearest 15-minute interval
        target_time += timedelta(minutes=15 - target_time.minute % 15)
        target_price = None
        for row in self.data:
            row_time = row['date']
            if row_time == target_time:
                target_price = row['price']
                break
            elif row_time > target_time:
                break
        return target_price
