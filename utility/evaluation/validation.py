from utility.recommendations import get_recommended_parking, get_recommended_timeslot
import numpy as np
import pandas as pd
from Model.ModelHandler import reset_model

from utility.util import *
from parameters import *
import multiprocessing as mp
import datetime

import matplotlib.pyplot as plt

RELEVANT_ITEMS = None
METRICS = {
    "total_predictions": 0,

    "accuracy_points": [0, 0,0,0], 
    "hit_rate_points": [0, 0,0,0], 

    "satisfaction_sum": [0, 0,0,0], 
    "precision_sum": [0, 0,0,0], 
    "recall_sum": [0, 0,0,0],
    "coverage_elems": [[], [], [], []]
}



def time_diff_secs(timestr1, timestr2):
    dt0 = datetime.datetime.timestamp(datetime.datetime.strptime(timestr1, "%H:%M:%S"))
    dt1 = datetime.datetime.timestamp(datetime.datetime.strptime(timestr2, "%H:%M:%S"))
    return dt1 - dt0

def satisfaction(delta, min_delta, n=5, m=3):
    normalized_delta = abs(delta)/min_delta
    return 1/((normalized_delta/m)**n + 1)


def append(x):
    global METRICS
    global RELEVANT_ITEMS

    if x is None:
        return

    recs, relevant_in_recs, correct_in_recs, satisfaction_in_recs, relevant_len = x   

    # truncate length to 4
    recs = recs[:4]
    relevant_in_recs = relevant_in_recs[:4]
    correct_in_recs = correct_in_recs[:4]
    satisfaction_in_recs = satisfaction_in_recs[:4]


    METRICS["total_predictions"] += 1

    for i in range(4):
        n = i + 1
        recs_sub = recs[:n]
        relevant_in_recs_sub = relevant_in_recs[:n]
        correct_in_recs_sub = correct_in_recs[:n]
        satisfaction_in_recs_sub = satisfaction_in_recs[:n]

        METRICS["accuracy_points"][i] += 1 if sum(correct_in_recs_sub) > 0 else 0
        METRICS["hit_rate_points"][i] += correct_in_recs_sub[0]

        precision = 0
        relevant_found = 0
        for j, e in enumerate(relevant_in_recs_sub):
            if e == 1:
                relevant_found += 1
                precision += relevant_found/(j+1)
        
        METRICS["precision_sum"][i] += precision/len(relevant_in_recs_sub)
        METRICS["recall_sum"][i] += relevant_found/relevant_len
        METRICS["satisfaction_sum"][i] += max(satisfaction_in_recs_sub) if satisfaction_in_recs_sub is not None else 0

        for elem in recs_sub:
            if elem not in METRICS["coverage_elems"][i]:
                METRICS["coverage_elems"][i].append(elem)

    format_print(f"Progress: {100* round( METRICS['total_predictions']/ METRICS['total_rows'], 4)}%", "yellow")
    format_print(f"current accuracy: {100*(METRICS['accuracy_points'][3]/METRICS['total_predictions'], 4)}%", "yellow")
    format_print(f"current satisfaction: {100*round(METRICS['satisfaction_sum'][3]/METRICS['total_predictions'], 4)}%", "yellow")

def task(model, user_db_filename, user, ts_actual, day_of_week, p_actual, type):
    # count how many ':' in time_slot
    time_slot_parts = ts_actual.split(":")
    if len(time_slot_parts) == 2:
        ts_actual += ":00"
    try:
        my_name=f"{user}_{ts_actual}_{day_of_week}_{p_actual}"

        if type == "p": # ts --> parking lot
        
            recs = [x[1] for x in get_recommended_parking(model,  user_db_filename, user, [ts_actual], day_of_week)]
            
            relevant_in_recs = [1 if x in RELEVANT_ITEMS[user]["p"][day_of_week] else 0 for x in recs]
            correct_in_recs = [1 if x == p_actual else 0 for x in recs]
            satisfaction_in_recs = [0 for _ in recs]
            relevant_len = len(RELEVANT_ITEMS[user]["p"][day_of_week]) 

        elif type == "fo": # parking lot --> ts
        
            recs = [x[0] for x in get_recommended_timeslot(model, user_db_filename, user, [p_actual], day_of_week)]
            
            relevant_in_recs = [1 if x in RELEVANT_ITEMS[user]["fo"][day_of_week] else 0 for x in recs]
            correct_in_recs = [1 if x == ts_actual else 0 for x in recs]
            satisfaction_in_recs = [satisfaction(time_diff_secs(x, ts_actual), TIMESLOT_RES_MIN * 60) for x in recs]
            relevant_len = len(RELEVANT_ITEMS[user]["fo"][day_of_week])

        elif type == "pfo": # nothing --> ts, parking lot

            recs = [(x[0], x[1]) for x in get_recommended_parking(model,  user_db_filename, user,  None, day_of_week)]

            relevant_in_recs = [1 if (x[0], x[1]) in RELEVANT_ITEMS[user]["pfo"][day_of_week] else 0 for x in recs]
            correct_in_recs = [1 if (x[0], x[1]) == (ts_actual, p_actual) else 0 for x in recs]
            satisfaction_in_recs = [0 for _ in recs] # [satisfaction(time_diff_secs(x[0], ts_actual), TIMESLOT_RES_MIN * 60) for x in recs]
            relevant_len = len(RELEVANT_ITEMS[user]["pfo"][day_of_week])
    

        return (recs, relevant_in_recs, correct_in_recs, satisfaction_in_recs, relevant_len)

    except Exception as e:

        console.print(f"{my_name} ==> {e}", style="red")
        dump_to_file(f"{my_name} ==> {e}")
        return None


def validation(model, user_db_filepath, validation_set_filepath, type):

    format_print("validation", "yellow")

    global RELEVANT_ITEMS
    global METRICS

    parkings = []
    for day in range(len(model["SuggeritoreFasceOrarieDaParcheggio_recommender"])):
        for key in model["SuggeritoreFasceOrarieDaParcheggio_recommender"][day]:
            if key not in parkings:
                parkings.append(key)


    num_parkings = len(parkings)

    num_timeslots = 60 * 24 // (TIMESLOT_RES_MIN)

    # load dataset and shuffle
    df_valid = pd.read_csv(validation_set_filepath)
    #df_valid = df_valid.sample(frac=0.01)
    METRICS["total_rows"] = df_valid.shape[0]
    df_valid = df_valid.sample(frac=1).reset_index(drop=True)

    #build table of relevant items
    RELEVANT_ITEMS = build_users_relevant_items(validation_set_filepath)


    pool = mp.Pool(CPU_CORES)
    for i, record in df_valid.iterrows():
        user = record.user
        time_slot = record.time
        day_of_week = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}[record.weekday]
        parking_lot = record.parking_lot

        pool.apply_async(task, args=(model, user_db_filepath, user, time_slot, day_of_week, parking_lot, type), callback=append)

    pool.close()    
    pool.join()


    results = {"accuracy": [METRICS["accuracy_points"][i]/METRICS["total_predictions"] for i in range(0, 4)],
               "hit_rate": [METRICS["hit_rate_points"][i]/METRICS["total_predictions"] for i in range(0, 4)],
               "precision": [METRICS["precision_sum"][i]/METRICS["total_predictions"] for i in range(0, 4)],
               "recall": [METRICS["recall_sum"][i]/METRICS["total_predictions"] for i in range(0, 4)],
               "satisfaction": [METRICS["satisfaction_sum"][i]/METRICS["total_predictions"] for i in range(0, 4)]}
    try:    
        results["f1-score"] = [2*results["precision"][i]*results["recall"][i]/(results["precision"][i]+results["recall"][i]) for i in range(0, 4)]
    except Exception as e:
        results["f1-score"] = -1

    if type == "pfo":
        results["coverage"] = [len(METRICS["coverage_elems"][i])/(num_timeslots * num_parkings) for i in range(0, 4)]
    elif type == "p":
        results["coverage"] = [len(METRICS["coverage_elems"][i])/(num_parkings) for i in range(0, 4)]
    elif type == "fo":
        results["coverage"] = [len(METRICS["coverage_elems"][i])/(num_timeslots) for i in range(0, 4)]

    
    # reset
    METRICS = {
        "total_predictions": 0,

        "accuracy_points": [0, 0,0,0], 
        "hit_rate_points": [0, 0,0,0], 

        "satisfaction_sum": [0, 0,0,0], 
        "precision_sum": [0, 0,0,0], 
        "recall_sum": [0, 0,0,0],
        "coverage_elems": [[], [], [], []]
    }

    reset_model()

    return results


def validation_plots(results, out_plots_dirpath):
    if out_plots_dirpath[-1] != "/":
        out_plots_dirpath += "/"

    map_keys_labels = {
        "accuracy": "Accuracy",
        "hit_rate": "Hit Rate",
        "precision": "Precision",
        "recall": "Recall",
        "satisfaction": "Satisfaction",
        "f1-score": "F1-Score",
        "coverage": "Coverage"
    }

    # for each metric, one bar for pfo, one for p and one for fo
    for metric in results["pfo"]:
        title = map_keys_labels[metric] if metric not in ["precision", "recall"] else "Mean Average " + map_keys_labels[metric]
        plt.figure()
        plt.title(title)
        plt.xlabel("Use case")
        plt.ylabel(map_keys_labels[metric])
        plt.xticks(range(3), ["PFO", "P", "FO"])

        plt.ylim(0, 1)

        plt.bar(0, results["pfo"][metric][-1], color="blue", label="PFO")
        plt.bar(1, results["p"][metric][-1], color="red", label="P")
        plt.bar(2, results["fo"][metric][-1], color="green", label="FO")
        #plt.legend()
        plt.savefig(out_plots_dirpath + title.lower().replace(" ", "_") + ".pdf")
        plt.close()


    # precision, satisfaction, f1-score when the index varies
    for metric in ["precision", "satisfaction", "f1-score"]:
        plt.figure()
        title = map_keys_labels[metric] + "@k"
        plt.title(title)
        plt.xlabel("k")
        plt.ylabel(map_keys_labels[metric])

        plt.xticks([ x + 0.2 for x in range(1, len(results["pfo"][metric]) + 1)], range(1, len(results["pfo"][metric]) + 1))
        
        for i, z in enumerate(["pfo", "p", "fo"]):
            xs = [1 + 0.2 * i + 1 * x for x in range(len(results[z][metric]))]
            plt.bar(xs, [results[z][metric][k] for k in range(len(results[z][metric]))], width=0.2, label=z.upper())

        plt.ylim(0, 1)

        plt.legend()
        plt.savefig(out_plots_dirpath + metric + "_at_k" + ".pdf")
        plt.close()
