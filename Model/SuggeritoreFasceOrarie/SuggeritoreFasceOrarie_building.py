from utility.util import *
from parameters import *
from utility.MatrixFactorizationEngine import *
import multiprocessing as mp
import psutil as ps

TIMESLOT_WEEK_CACHE = None
timeslot_model = []

def append_to_model(elem):
    global timeslot_model
    timeslot_model.append(elem)

def task(day_of_week, df, users, num_time_slots):

    global TIMESLOT_WEEK_CACHE

    my_name = f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}"

    format_print(f"{my_name}: building started", "green")

    df_filter_week = filter_by_day_of_week(df, day_of_week)
    
    cache = False

    # if there are no data in the dataset for the specified day of week, reload the model without filters
    if df_filter_week is None or df_filter_week.shape[0] == 0:
        format_print(f"{my_name}: no data in the dataset for the specified day of week", "yellow")

        if TIMESLOT_WEEK_CACHE is None:
            df_filter_week = df.copy()
            format_print(f"{my_name}: reloading model without filters", "yellow")
            cache = True
        else:
            format_print(f"{my_name}: using cached data", "purple")
            record_metric(f"{my_name} end", "cpu", ps.cpu_percent(percpu=True))
            record_metric(f"{my_name} end", "ram", ps.virtual_memory())
            
            elem = TIMESLOT_WEEK_CACHE.copy()
            elem["day_of_week"] = day_of_week
            
            return elem

    matrix, time_slot_labels = build_ratings_matrix_SuggeritoreFasceOrarie(df_filter_week, users, num_time_slots)

    # get the ranking of the top time_slots
    top_n_indexes = np.argsort(matrix.sum(axis=0))[::-1][:SUGGFASCEORARIE_TOP_N].tolist()
    # normalize matrix
    matrix = decoupling_normalization(matrix)
    # build model
    (p, q) = pq_factor(matrix, SUGGFASCEORARIE_LATENT_VARS, SUGGFASCEORARIE_LEARNING_RATE, SUGGFASCEORARIE_EPOCHS)
    elem = {
        "p_matrix": p,
        "q_matrix": q,
        "users": users,
        "labels": time_slot_labels,
        "ranking": top_n_indexes,
    }
    
    if cache:
        TIMESLOT_WEEK_CACHE = elem.copy()
        format_print(f"{my_name}: cached data", "purple")

    record_metric(f"{my_name} end", "cpu", ps.cpu_percent(percpu=True))
    record_metric(f"{my_name} end", "ram", ps.virtual_memory())

    format_print(f"{my_name}: model built", "blue")

    return elem


def build_SuggeritoreFasceOrarie_recommendation_model(dataset_filepath):

    global timeslot_model

    format_print("Building time slot recommender model", "bold green")

    pool = mp.Pool(CPU_CORES)

    df = load_dataset(dataset_filepath)

    # retrieve list of users
    users = df["user"].unique()

    # get number of time slots
    num_time_slots = 24 * 60 // TIMESLOT_RES_MIN

    for i in range(7):
        pool.apply_async(task, args=(i, df, users, num_time_slots), callback=append_to_model)

    pool.close()
    pool.join()

    record_metric("time slot model end", "cpu", ps.cpu_percent(percpu=True))
    record_metric("time slot model end", "ram", ps.virtual_memory())
    return timeslot_model
