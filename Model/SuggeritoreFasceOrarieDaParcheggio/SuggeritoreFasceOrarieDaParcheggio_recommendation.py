from parameters import *
import numpy as np
import datetime
from utility.MatrixFactorizationEngine import *
from utility.util import *


def SuggeritoreFasceOrarieDaParcheggio_recommendation(model, user_id, day_of_week, parking_lot, user_db):

    model = model["SuggeritoreFasceOrarieDaParcheggio_recommender"][day_of_week][parking_lot]

    # load the user from the database
    user_row = load_user_from_db(user_id, parking_lot, day_of_week, model, user_db)

    # if the user is new, get the ranking of the top parkings
    if user_row is None or user_row.shape[0] == 0:
        format_print(f"New user for {day_of_week} {parking_lot}", "yellow")
        return [(model["labels"][i] , -1)  for i in model["ranking"]]

    # convert user_row in a (index, value) list for non-zero elements
    user_tuples = [(i, user_row[0, i]) for i in range(len(user_row[0])) if user_row[0, i] > 0]

    # only use PARKING_USER_INPUTS_TO_MODEL parkings
    user_tuples = sorted(user_tuples, key=lambda x: x[1], reverse=True)
    user_tuples = user_tuples[:SUGGFASCEORARIEPARCHEGGIO_USER_INPUTS_TO_MODEL]

    # compute the recommendation
    output = recommending_for_user(user_tuples, model["p_matrix"], model["q_matrix"], SUGGFASCEORARIEPARCHEGGIO_LEARNING_RATE, SUGGFASCEORARIEPARCHEGGIO_EPOCHS)[0]

    # get sorted TOP_N indexes
    top_n_indexes = np.argsort(output)[::-1][:SUGGFASCEORARIEPARCHEGGIO_TOP_N].tolist()

    # print top_n_indexes with values
    return [(model["labels"][i], output[i]) for i in top_n_indexes if output[i] > SUGGFASCEORARIEPARCHEGGIO_TOP_N_THRESHOLD]


def load_user_from_db(user_id, parking_lot, day_of_week, model, user_db):
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

    # filter the dataset by parking lot
    df_filter = df_filter.loc[df_filter['parking_lot'] == parking_lot]

    # if there are no records for the user in the day of week and time slot, don't use the filters
    if df_filter is None or df_filter.shape[0] == 0:
        df_filter = df

    # retrieve list of time slots
    timeslots = model['labels']

    # build matrix of None
    matrix = np.zeros((1, len(timeslots)))

    # build matrix for each user count the number of parkings in each time slot
    for j, time_slot in enumerate(timeslots):
        
        end = str(datetime.timedelta(hours=int(time_slot.split(
        ":")[0])) + datetime.timedelta(minutes=int(time_slot.split(":")[1]) + TIMESLOT_RES_MIN) - datetime.timedelta(seconds=1))
        
        df_timeslot = df_filter.between_time(time_slot, end)
        matrix[0, j] = df_timeslot.shape[0]       

    # normalize matrix
    matrix = decoupling_normalization(matrix)

    return matrix
