import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


def register_user():
    register_url = 'https://api2.watttime.org/v2/register'
    params = {'username': 'slrpev',
              'password': 'slrpev-sim-2023',
              'email': 'darya_darvish@berkeley.edu',
              'org': 'slrpev-world'}
    rsp = requests.post(register_url, json=params)
    print(rsp.text)


def login():
    login_url = 'https://api2.watttime.org/v2/login'
    rsp = requests.get(login_url, auth=HTTPBasicAuth('slrpev', 'slrpev-sim-2023'))
    print(rsp.text)
    return rsp.json()['token']


def determine_grid_region(token, latitude, longitude):
    region_url = 'https://api2.watttime.org/v2/ba-from-loc'
    headers = {'Authorization': 'Bearer {}'.format(token)}
    params = {'latitude': latitude, 'longitude': longitude}
    rsp = requests.get(region_url, headers=headers, params=params)
    print(rsp.text)
    return rsp.json()


def get_emissions(token, starttime, endtime):
    data_url = 'https://api2.watttime.org/v2/data'
    headers = {'Authorization': 'Bearer {}'.format(token)}
    params = {'ba': 'CAISO_NORTH',
              'starttime': starttime,
              'endtime': endtime}
    rsp = requests.get(data_url, headers=headers, params=params)
    return pd.DataFrame(rsp.json())


# register_user()
token = login()

# latitude and longitude for UC Berkeley
latitude = '37.87'
longitude = '-122.25'
starttime = '2022-11-16T20:30:00-0800'
endtime = '2022-11-16T20:45:00-0800'
determine_grid_region(token, latitude, longitude)
emissions_df = get_emissions(token, starttime, endtime)
