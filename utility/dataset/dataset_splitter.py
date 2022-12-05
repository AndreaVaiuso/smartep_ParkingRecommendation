import numpy as np
import pandas as pd
from util.util import *
from parameters import *

OLD_DATA_PERC = 0.75
NEW_DATA_PERC = 0.2
VALID_DATA_PERC = 1 - OLD_DATA_PERC - NEW_DATA_PERC


# Splits the dataset into old, new and validation sets. Merges old and new sets to create the training set.
# 
#                              DATASET
#
# +-------------------------------------+-------------+---------------+
# |               old (75%)             |  new (20%)  | valid (0.05%) |
# +-------------------------------------+-------------+---------------+
# 
# |                                                   |
# +---------------------------------------------------+
#                       training set

def split(dataset_file_name):

    # load dataset
    df = pd.read_csv(dataset_file_name)
    format_print(f"Splitting dataset {dataset_file_name} in {100*OLD_DATA_PERC}% = {df.shape[0]*OLD_DATA_PERC} records (old data), {100*NEW_DATA_PERC}% = {df.shape[0]*NEW_DATA_PERC} records (new data), {100*VALID_DATA_PERC}% = {df.shape[0]*VALID_DATA_PERC} records (validation data)", "white")

    # split df into old, new and validation data
    df_old = df[:int(df.shape[0]*OLD_DATA_PERC)]
    df_new = df[int(df.shape[0]*OLD_DATA_PERC):int(df.shape[0]*(OLD_DATA_PERC + NEW_DATA_PERC))]
    df_valid = df[int(df.shape[0]*(OLD_DATA_PERC+NEW_DATA_PERC)):]

    # select 2 random users from old set and move them to new set
    validation_users = np.random.choice(df.user.unique(), size=2, replace=False)
    for user in validation_users:
        # move the user to validation from old
        df_new = df_new.append(df_old[df_old.user == user])
        df_old = df_old[df_old.user != user]

    # select 2 random users from old set and move them to validation
    validation_users = np.random.choice(df.user.unique(), size=2, replace=False)
    for user in validation_users:
        # move the user to validation from old
        df_valid = df_valid.append(df_old[df_old.user == user])
        df_old = df_old[df_old.user != user]

    # write data to csv
    df_old.to_csv(f"data/{normalize_dataset_file_name(dataset_file_name)}_old.csv", index=False)
    df_new.to_csv(f"data/{normalize_dataset_file_name(dataset_file_name)}_new.csv", index=False)
    df_valid.to_csv(f"data/{normalize_dataset_file_name(dataset_file_name)}_valid.csv", index=False)

    # build old+new set
    df_train = df_old.append(df_new)

    # write data to csv
    df_train.to_csv(f"data/{normalize_dataset_file_name(dataset_file_name)}_train.csv", index=False)    


if __name__ == "__main__":
    split(DATASET_FILE_NAME)