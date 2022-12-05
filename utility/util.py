import pandas as pd
import numpy as np
import datetime
from rich import console
from parameters import VERBOSE, RECORD_METRICS, TIMESLOT_RES_MIN

console = console.Console()

METRIC_FILE = None
DUMP_FILE = None

def init_metric_dump_files(dataset_file_name):
    global METRIC_FILE, DUMP_FILE

    normalized_file_name = normalize_dataset_file_name(dataset_file_name)

    METRIC_FILE = open(f"out/{normalized_file_name}_metrics.csv", "a")
    DUMP_FILE = open(f"out/{normalized_file_name}_dump.txt", "a")


def normalize_dataset_file_name(dataset_file_name):
    return dataset_file_name.split('/')[1].split('.')[0] if dataset_file_name is not None else "none"

def load_dataset(dataset_file):
    # load dataset
    df = pd.read_csv(dataset_file)

    # add day of week column
    df['day_of_week'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").weekday())

    # put timeslots for each row in the dataframe
    dt = pd.to_datetime(df['time'])
    df['timeslot'] = ((dt.dt.second + dt.dt.minute * 60 + dt.dt.hour * 3600) // (TIMESLOT_RES_MIN * 60) * (TIMESLOT_RES_MIN * 60 ))
    # convert seconds to HH:MM:SS format
    df['timeslot'] = df['timeslot'].apply(lambda x: str(datetime.timedelta(seconds=x)))

    # apply DateTimeIndex on time field
    df['time'] = pd.DatetimeIndex(df['time'])
    df.set_index('time', inplace=True)

    return df


def filter_by_day_of_week(df, day_of_week, fail_on_missing=True):

    # filter the dataset by the given day of week
    df = df[df['day_of_week'] == day_of_week]

    if(fail_on_missing and df.shape[0] == 0):
       return None

    return df


def record_metric(label, metric, value):
    global METRIC_FILE
    if RECORD_METRICS:
        METRIC_FILE.write(str(datetime.datetime.now()) + ";" + str(label) + ";" + str(metric) + ";" + str(value) + "\n")
        METRIC_FILE.flush()


def dump_to_file(text):
    global DUMP_FILE
    if VERBOSE:
        DUMP_FILE.write(str(datetime.datetime.now()) + "," + str(text) + "\n")
        DUMP_FILE.flush()


def format_print(text, style_color):
    if VERBOSE:
        console.print(text, style=style_color)


def build_ratings_matrix_SuggeritoreFasceOrarie(df, users, num_time_slots):

    # build matrix
    matrix = np.zeros((len(users), num_time_slots))
    # remember the labels for each time slot
    time_slot_labels = []
    # build matrix for each user count the number of parkings in each time slot of resolution TIME_SLOT_MODEL_RES_H
    for j in range(0, 24 * 60 // TIMESLOT_RES_MIN):
        # convert i to hours
        start_slot = datetime.timedelta(seconds=0) + datetime.timedelta(
            minutes=(j * TIMESLOT_RES_MIN)
        )
        end_slot = (
            datetime.timedelta(seconds=0)
            + datetime.timedelta(minutes=((j + 1) * TIMESLOT_RES_MIN))
            + datetime.timedelta(seconds=-1)
        )
        # append element in time_slot_labels
        time_slot_labels.append(str(start_slot))
        # filter the dataset by time slot
        df_slot = df.between_time(str(start_slot), str(end_slot))
        # count the number of parkings in each time slot for each user
        for i, user in enumerate(users):
            # put the number of parkings in the matrix
            matrix[i, j] = df_slot[df_slot["user"] == user]["user"].count()

    return matrix, time_slot_labels


def build_ratings_matrix_SuggeritoreParcheggi(df, users, parkings):

    # build matrix
    matrix = np.zeros((len(users), len(parkings)))
    # build matrix for each user count the number of parkings in each time slot of resolution TIME_SLOT_MODEL_RES_H
    for j, parking in enumerate(parkings):
        # count the number of parkings in each time slot for each user
        for i, user in enumerate(users):
            # put the number of parkings in the matrix
            matrix[i, j] = df[
                (df["user"] == user)
                & (df["parking_lot"] == parkings[j])
            ].shape[0]

    return matrix


def build_ratings_matrix_SuggeritoreFasceOrarieDaParcheggio(df, users, timeslots):

    # build matrix
    matrix = np.zeros((len(users), len(timeslots)))
    # build matrix for each user count the number of parkings in each time slot of resolution TIME_SLOT_MODEL_RES_H
    for j, time_slot in enumerate(timeslots):
        # count the number of parkings in each time slot for each user
        for i, user in enumerate(users):
            # put the number of parkings in the matrix
            df_timeslot = df.between_time(time_slot, str(datetime.timedelta(hours=int(time_slot.split(
                ":")[0])) + datetime.timedelta(minutes=int(time_slot.split(":")[1]) + TIMESLOT_RES_MIN) - datetime.timedelta(seconds=1)),)
            matrix[i, j] = df_timeslot[df_timeslot["user"] == user].shape[0]

    return matrix


def build_users_relevant_items(user_db_filepath):
    df = load_dataset(user_db_filepath)
    
    table = {}

    users = df["user"].unique()

    for user in users:
        user_table = {"p":{}, "fo":{}, "pfo":{}}
        df_user = df[df["user"] == user]

        for day in range(0, 7):
            df_day = filter_by_day_of_week(df_user, day)
            if df_day is None:
                user_table["p"][day] = []
                user_table["fo"][day] = []
                user_table["pfo"][day] = []
            else:                
                user_table["p"][day] = df_day["parking_lot"].unique()
                user_table["fo"][day] = df_day["timeslot"].unique()
                # pfo is unique couples of parking lot and timeslot
                user_table["pfo"][day] = list(set(list(zip(df_day["timeslot"], df_day["parking_lot"]))))
        
        table[user] = user_table

    return table

                
