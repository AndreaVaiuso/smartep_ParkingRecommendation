from parameters import *
from utility.util import *

from Model.SuggeritoreFasceOrarie.SuggeritoreFasceOrarie_building import build_SuggeritoreFasceOrarie_recommendation_model
from Model.SuggeritoreFasceOrarie.SuggeritoreFasceOrarie_sml_update import sml_update_SuggeritoreFasceOrarie

from Model.SuggeritoreFasceOrarieDaParcheggio.SuggeritoreFasceOrarieDaParcheggio_building import build_SuggeritoreFasceOrarieDaParcheggio_recommendation_model
from Model.SuggeritoreFasceOrarieDaParcheggio.SuggeritoreFasceOrarieDaParcheggio_sml_update import sml_update_SuggeritoreFasceOrarieDaParcheggio

from Model.SuggeritoreParcheggi.SuggeritoreParcheggi_building import build_SuggeritoreParcheggi_recommendation_model
from Model.SuggeritoreParcheggi.SuggeritoreParcheggi_sml_update import sml_update_SuggeritoreParcheggi

import json
import pandas as pd
import numpy as np
import multiprocessing as mp
import psutil as ps

MODEL = {
    "SuggeritoreFasceOrarie_recommender": [],
    "SuggeritoreParcheggi_recommender": [],
    "SuggeritoreFasceOrarieDaParcheggio_recommender": [],
}


def reset_model():
    global MODEL
    MODEL = {
        "SuggeritoreFasceOrarie_recommender": [],
        "SuggeritoreParcheggi_recommender": [],
        "SuggeritoreFasceOrarieDaParcheggio_recommender": [],
    }


def init_model(model_filepath, dataset_filepath, output_model_filepath):

    init_metric_dump_files(output_model_filepath if model_filepath is None else model_filepath)

    reset_model()

    record_metric("Start model build", "cpu", ps.cpu_percent(percpu=True))
    record_metric("Start model build", "ram", ps.virtual_memory())

    format_print("Initializing model", "bold green")

    global MODEL

    if model_filepath is not None:
        format_print("Loading model from file: " + model_filepath, "green")
        load_model(model_filepath)
    else:
        build_model(dataset_filepath)
        save_model(output_model_filepath)

    return MODEL


def load_model(filename):

    global MODEL

    format_print("Loading model", "bold green")

    with open(filename) as json_file:
        MODEL = json.load(json_file)

    # convert lists to numpy arrays
    # prepare the model by converting each numpy array to list
    for i in range(0, len(MODEL["SuggeritoreFasceOrarie_recommender"])):
        MODEL["SuggeritoreFasceOrarie_recommender"][i]["p_matrix"] = np.array(
            MODEL["SuggeritoreFasceOrarie_recommender"][i]["p_matrix"]
        )
        MODEL["SuggeritoreFasceOrarie_recommender"][i]["q_matrix"] = np.array(
            MODEL["SuggeritoreFasceOrarie_recommender"][i]["q_matrix"]
        )

    for i in range(0, len(MODEL["SuggeritoreParcheggi_recommender"])):
        for j in MODEL["SuggeritoreParcheggi_recommender"][i]:
            MODEL["SuggeritoreParcheggi_recommender"][i][j]["p_matrix"] = np.array(
                MODEL["SuggeritoreParcheggi_recommender"][i][j]["p_matrix"]
            )
            MODEL["SuggeritoreParcheggi_recommender"][i][j]["q_matrix"] = np.array(
                MODEL["SuggeritoreParcheggi_recommender"][i][j]["q_matrix"]
            )

    for i in range(0, len(MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"])):
        for j in MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"][i]:
            MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"][i][j]["p_matrix"] = np.array(
                MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"][i][j]["p_matrix"]
            )
            MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"][i][j]["q_matrix"] = np.array(
                MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"][i][j]["q_matrix"]
            )

    format_print("Model loaded", "green")


# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def save_model(filename):
    global MODEL

    format_print("Saving model", "bold green")

    with open(filename, "w") as json_file:
        json.dump(MODEL, json_file, cls=NumpyEncoder)

    format_print("Model saved", "green")


def build_model(dataset_filepath):

    global MODEL

    format_print("Building model", "bold green")
    format_print(
        f"Parallelizing model building with {CPU_CORES} cores", "bold green")

    MODEL["SuggeritoreFasceOrarie_recommender"] = build_SuggeritoreFasceOrarie_recommendation_model(
        dataset_filepath)

    time_slots = MODEL["SuggeritoreFasceOrarie_recommender"][0]["labels"]

    MODEL["SuggeritoreParcheggi_recommender"] = build_SuggeritoreParcheggi_recommendation_model(
        time_slots, dataset_filepath)

    MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"] = build_SuggeritoreFasceOrarieDaParcheggio_recommendation_model(
        time_slots, dataset_filepath)

    format_print("Model built", "green")


def model_update(new_data_filename, model_updated_output_path, combination_coefficient):

    global MODEL

    SuggeritoreFasceOrarie_update = sml_update_SuggeritoreFasceOrarie(
        MODEL["SuggeritoreFasceOrarie_recommender"], new_data_filename, combination_coefficient)

    # update SuggeritoreFasceOrarie_recommender
    for i, week_model in enumerate(MODEL["SuggeritoreFasceOrarie_recommender"]):
        day_of_week = i
        # find model in time slot model
        for j, model in enumerate(SuggeritoreFasceOrarie_update):
            if model[0] == day_of_week:
                MODEL["SuggeritoreFasceOrarie_recommender"][i]["p_matrix"] = model[1]
                MODEL["SuggeritoreFasceOrarie_recommender"][i]["q_matrix"] = model[2]
                MODEL["SuggeritoreFasceOrarie_recommender"][i]["ranking"] = model[3]
                break

    # update SuggeritoreParcheggi_recommender
    SuggeritoreParcheggi_update = sml_update_SuggeritoreParcheggi(
        MODEL["SuggeritoreParcheggi_recommender"], new_data_filename, combination_coefficient)

    for i, week_model in enumerate(MODEL["SuggeritoreParcheggi_recommender"]):
        for time_slot in week_model:
            for j, model in enumerate(SuggeritoreParcheggi_update):
                if model[0] == i and model[1] == time_slot:
                    MODEL["SuggeritoreParcheggi_recommender"][i][time_slot]["p_matrix"] = model[2]
                    MODEL["SuggeritoreParcheggi_recommender"][i][time_slot]["q_matrix"] = model[3]
                    MODEL["SuggeritoreParcheggi_recommender"][i][time_slot]["ranking"] = model[4]
                    break

    # update SuggeritoreFasceOrarieDaParcheggio_recommender
    SuggeritoreFasceOrarieDaParcheggio_update = sml_update_SuggeritoreFasceOrarieDaParcheggio(
        MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"], new_data_filename, combination_coefficient)

    for i, week_model in enumerate(MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"]):
        for parking_lot in week_model:
            for j, model in enumerate(SuggeritoreFasceOrarieDaParcheggio_update):
                if model[0] == i and model[1] == parking_lot:
                    MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"][i][parking_lot]["p_matrix"] = model[2]
                    MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"][i][parking_lot]["q_matrix"] = model[3]
                    MODEL["SuggeritoreFasceOrarieDaParcheggio_recommender"][i][parking_lot]["ranking"] = model[4]
                    break

    save_model(model_updated_output_path)

    format_print("Model updated", "green")

    return MODEL
