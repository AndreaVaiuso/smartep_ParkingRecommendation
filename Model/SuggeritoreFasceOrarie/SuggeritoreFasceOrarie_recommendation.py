import pandas as pd
import datetime
from parameters import *
import numpy as np
from utility.MatrixFactorizationEngine import *
from utility.util import *


def SuggeritoreFasceOrarie_recommendation(model, user_id, day_of_week, user_db):

    model = model["SuggeritoreFasceOrarie_recommender"][day_of_week]

    # load the user from the database
    user_row = load_user_from_db(user_id, day_of_week, model, user_db)

    # if the user is new, get the ranking of the top time_slots
    if user_row is None or user_row.shape[0] == 0:
        format_print(f"New user for {day_of_week}", "yellow")
        return [(model["labels"][i] , -1)  for i in model["ranking"]]

    # convert user_row in a (index, value) list for non-zero elements
    user_tuples = [(i, user_row[0, i]) for i in range(len(user_row[0])) if user_row[0, i] > 0]


    # only use TIME_SLOT_USER_INPUTS_TO_MODEL time slots
    user_tuples = sorted(user_tuples, key=lambda x: x[1], reverse=True)
    user_tuples = user_tuples[:SUGGFASCEORARIE_USER_INPUTS_TO_MODEL]

    # compute the recommendation
    output = recommending_for_user(user_tuples, model["p_matrix"], model["q_matrix"], SUGGFASCEORARIE_LEARNING_RATE, SUGGFASCEORARIE_EPOCHS)[0]

    # get sorted TOP_N indexes
    top_n_indexes = np.argsort(output)[::-1][:SUGGFASCEORARIE_TOP_N].tolist()
    
    # return top_n_indexes with values
    return [(model["labels"][x], output[x]) for x in top_n_indexes if output[x] > SUGGFASCEORARIE_TOP_N_THRESHOLD]


def load_user_from_db(user_id, day_of_week, model, user_db):

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

    # get number of time slots
    num_time_slots = len(model["labels"])

    # compute resolution
    res = 24 * 60 // num_time_slots

    # build matrix
    matrix = np.zeros((1, num_time_slots))

    # build matrix for each user count the number of parkings in each time slot of resolution TIME_SLOT_MODEL_RES_H
    for j in range(0, num_time_slots):
        # convert i to hours
        start_slot = datetime.timedelta(seconds=0) + datetime.timedelta(minutes=(j * res))
        end_slot = datetime.timedelta(seconds=0) + datetime.timedelta(minutes=((j + 1) * res)) + datetime.timedelta(seconds=-1)

        # filter the dataset by time slot
        df_slot = df_filter.between_time(str(start_slot), str(end_slot))

        # count the number of parkings in each time slot for the user
        matrix[0, j] = df_slot.shape[0]

    # normalize matrix
    matrix = decoupling_normalization(matrix)

    return matrix