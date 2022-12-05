import pandas as pd
from parameters import *
import numpy as np
import datetime
from utility.MatrixFactorizationEngine import *
from utility.util import *


def SuggeritoreParcheggi_recommendation(model, user_id, day_of_week, time_slot, user_db):

    model = model["SuggeritoreParcheggi_recommender"][day_of_week]
    model = model[time_slot]

    # load the user from the database
    user_row = load_user_from_db(user_id, time_slot, day_of_week, model, user_db)

    # if the user is new, get the ranking of the top parkings
    if user_row is None or user_row.shape[0] == 0:
        format_print(f"New user for {day_of_week} {time_slot}", "yellow")
        return [(model["labels"][i] , -1) for i in model["ranking"]]

    # convert user_row in a (index, value) list for non-zero elements
    user_tuples = [(i, user_row[0, i]) for i in range(len(user_row[0])) if user_row[0, i] > 0]

    # only use PARKING_USER_INPUTS_TO_MODEL parkings
    user_tuples = sorted(user_tuples, key=lambda x: x[1], reverse=True)
    user_tuples = user_tuples[:SUGGPARCHEGGI_USER_INPUTS_TO_MODEL]

    # compute the recommendation
    output = recommending_for_user(user_tuples, model["p_matrix"], model["q_matrix"], SUGGPARCHEGGI_LEARNING_RATE, SUGGPARCHEGGI_EPOCHS)[0]

    # get sorted TOP_N indexes
    top_n_indexes = np.argsort(output)[::-1][:SUGGPARCHEGGI_TOP_N].tolist()

    # print top_n_indexes with values
    return [(model["labels"][i], output[i]) for i in top_n_indexes if output[i] > SUGGPARCHEGGI_TOP_N_THRESHOLD]


def load_user_from_db(user_id, time_slot, day_of_week, model, user_db):
    # filter by user
    df = user_db[user_db['user'] == user_id]

    # if new user, return None
    if df is None or df.shape[0] == 0:
        return None

    # filter by day of week
    df_filter = filter_by_day_of_week(df, day_of_week)

    # if there are no records for the user in the day of week, don't use the filters
    if df_filter is None or df_filter.shape[0] == 0:
        df_filter = df

    # filter the dataset by time slot
    df_filter = df_filter.between_time(time_slot, str(datetime.timedelta(hours=int(time_slot.split(
        ":")[0])) + datetime.timedelta(minutes=int(time_slot.split(":")[1]) + TIMESLOT_RES_MIN) - datetime.timedelta(seconds=1)))

    # if there are no records for the user in the day of week and time slot, don't use the filters
    if df_filter is None or df_filter.shape[0] == 0:
        df_filter = df

    # retrieve list of parkings
    parkings = model['labels']

    # build matrix of None
    matrix = np.zeros((1, len(parkings)))

    # build matrix for each user count the number of parkings in each time slot of resolution TIME_SLOT_MODEL_RES_H
    for j, parking in enumerate(parkings):
        # count the number of parkings in each time slot for each user
        matrix[0, j] = df_filter[df_filter['parking_lot'] == parking].shape[0]

    # normalize matrix
    matrix = decoupling_normalization(matrix)

    return matrix
