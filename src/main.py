from TOU import *

data = CAISOData('C:/Users/seungyun/OneDrive/SlrpEv Capstone Project/SlrpEv/pev-digital-twin/data/20200101-20230223_CAISO_Average_Price.csv')
target_time = datetime(2020, 1, 1, 0, 10)
target_price = data.get_price_at_time(target_time)
print(f"The price at {target_time} is {target_price:.2f}")

