from parameters import *
import requests
from requests.structures import CaseInsensitiveDict
from utility.util import format_print

def get_top_parking_by_occupancy_model(ts_park_couples,  day_of_week):

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"

    ret = {}

    for (ts, parking_lot) in ts_park_couples:

        # normalize timeslot (remove seconds + add unnecessary 0)
        ts = ts[:-3]

        if len(ts) == 4:
            ts = '0' + ts

        # format string
        payload = '{'
        payload += f'"WEEKDAY": {day_of_week},'
        payload += f'"WEEK_SHIFT": 0,'
        payload += f'"PKLOT_ID": {parking_lot}'
        payload += '}'

        resp = requests.post(OCCUPANCY_PREDICTOR_URL, headers=headers, data=payload)

        if resp.status_code != 200:
            format_print(f"Error: {resp.status_code}", "red")
            return None
        
        resp_json = resp.json()
        
        # store the delta to the target occupancy
        ret[(ts, parking_lot)] = (abs(float(resp_json[ts]) - OCCUPANCY_PREDICTOR_TARGET_OCCUPANCY), float(resp_json[ts]))
    

    # sort ret in increasing order
    sorted_ret = sorted(ret.items(), key=lambda kv: kv[1][0])

    return sorted_ret[:OCCUPANCY_PREDICTOR_TOP_N]
