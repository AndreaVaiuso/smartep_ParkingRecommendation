from pyexpat import model
from Model.SuggeritoreFasceOrarie.SuggeritoreFasceOrarie_recommendation import SuggeritoreFasceOrarie_recommendation
from Model.SuggeritoreFasceOrarieDaParcheggio.SuggeritoreFasceOrarieDaParcheggio_recommendation import SuggeritoreFasceOrarieDaParcheggio_recommendation
from Model.SuggeritoreParcheggi.SuggeritoreParcheggi_recommendation import SuggeritoreParcheggi_recommendation

from OccupancyPredictorInterface.OccupancyPredictor_recommendation import get_top_parking_by_occupancy_model

from parameters import *
import datetime
from Model.ModelHandler import *
from utility.util import *


CACHED_USER_DB = {}


def get_recommended_parking(model, user_db_filename, user_id, time_slots=None, day_of_week=datetime.datetime.today().weekday()):
    global CACHED_USER_DB

    if user_db_filename not in CACHED_USER_DB:
        CACHED_USER_DB[user_db_filename] = load_dataset(user_db_filename)

    user_db = CACHED_USER_DB[user_db_filename]

    user_db = load_dataset(user_db_filename)

    format_print(
        f"Getting recommended parking for user {user_id}", "bold rgb(255,145,0)")
    if time_slots is None:
        format_print("No time slots provided, computing recommended time slots", "bold rgb(255,145,0)")
        # retrieve the best time slot for the user
        time_slots = SuggeritoreFasceOrarie_recommendation(model, user_id, day_of_week, user_db)

    ts_park_couples = []
    for time_slot in time_slots:
        if len(time_slot)  == 2:
            time_slot = time_slot[0]
        format_print("Computing recommended parkings for time slot: " +
                     time_slot, "bold rgb(255,145,0)")
        # get the top n items for the user in the given time slot
        parkings = SuggeritoreParcheggi_recommendation(model, user_id, day_of_week, time_slot, user_db)
    
        for parking in parkings:
            ts_park_couples.append((time_slot, parking))

    ts_park_couples = [(x[0], x[1][0]) for x in ts_park_couples]

    # retrieve the best parking according to the predicted occupancy in the time slot
    if OCCUPANCY_PREDICTOR_ENABLED:
        recommendations = get_top_parking_by_occupancy_model(ts_park_couples, day_of_week)
        recommendations = [[x[0][0] + ":00", x[0][1]] for x in recommendations]
    else:
        recommendations = ts_park_couples

    format_print("Recommended parking: " + str(recommendations), "bold rgb(255,145,0)")

    return recommendations


def get_recommended_timeslot(model, user_db_filename, user_id, parking_lots, day_of_week=datetime.datetime.today().weekday()):

    global CACHED_USER_DB

    if user_db_filename not in CACHED_USER_DB:
        CACHED_USER_DB[user_db_filename] = load_dataset(user_db_filename)

    user_db = CACHED_USER_DB[user_db_filename]

    format_print(f"Getting recommended time slot for user {user_id}", "bold rgb(255,145,0)")

    ts_park_couples = []

    # retrieve the best time slot for the user for each parking lot
    for parking_lot in parking_lots:
        parking_lot = str(parking_lot)
        format_print("Computing recommended time slot for parking lot: " +
                     str(parking_lot), "bold rgb(255,145,0)")
        time_slots = SuggeritoreFasceOrarieDaParcheggio_recommendation(model, user_id, day_of_week, parking_lot, user_db)

        for time_slot in time_slots:
            ts_park_couples.append((time_slot, parking_lot))
    ts_park_couples = [(x[0][0], x[1]) for x in ts_park_couples]
    if OCCUPANCY_PREDICTOR_ENABLED:
        recommendations = get_top_parking_by_occupancy_model(ts_park_couples, day_of_week)
        recommendations = [[x[0][0] + ":00", x[0][1]] for x in recommendations]
    else:
        recommendations = ts_park_couples

    return recommendations

