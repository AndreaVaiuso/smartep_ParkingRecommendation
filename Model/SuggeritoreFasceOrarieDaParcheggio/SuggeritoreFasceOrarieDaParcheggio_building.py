from utility.util import *
from parameters import *
from utility.MatrixFactorizationEngine import *
import multiprocessing as mp
import psutil as ps

model = []


SUGGFASCEORARIEPARCHEGGIO_WEEK_CACHE = None
SUGGFASCEORARIEPARCHEGGIO_PARKING_CACHE = [None] * 7 # CHECK

def append_to_model(elem):
    global model

    model.append(elem.copy())

def task(day_of_week, parking_lot, df, users, time_slots):

    global SUGGFASCEORARIEPARCHEGGIO_WEEK_CACHE
    global SUGGFASCEORARIEPARCHEGGIO_PARKING_CACHE

    my_name = f"{parking_lot}@{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}"

    format_print(f"{my_name}: building started", "green")

    df_filter_week = filter_by_day_of_week(df, day_of_week)

    cache_week = False

    # if there are no data in the dataset for the specified day of week
    if df_filter_week is None or df_filter_week.shape[0] == 0:
        format_print(f"{my_name}: no data in the dataset for the specified day of week", "yellow")

        # reload the model without filters
        if SUGGFASCEORARIEPARCHEGGIO_WEEK_CACHE is None:
            df_filter_week = df.copy()
            format_print(f"{my_name}: reloading model without filters", "yellow")
            cache_week = True
        # or use the cached data
        else:
            format_print(f"{my_name}: using cached data", "purple")
            record_metric(f"{my_name}: building finished", "cpu", ps.cpu_percent(percpu=True))
            record_metric(f"{my_name}: building finished", "ram", ps.virtual_memory())
            
            elem = SUGGFASCEORARIEPARCHEGGIO_WEEK_CACHE.copy()
            elem["day_of_week"] = day_of_week
            elem["parking_lot"] = parking_lot
            
            return elem
           
    
    # filter the dataset by parking lot
    df_filter_parking = df_filter_week[df_filter_week["parking_lot"] == parking_lot]


    cache_time = False

    # if there are no data in the dataset for the specified time slot
    if df_filter_parking.shape[0] == 0:

        format_print(f"{my_name}: no data in the dataset for the specified time slot", "yellow")
        # reload the model without filters
        if SUGGFASCEORARIEPARCHEGGIO_PARKING_CACHE[day_of_week] is None:
            df_filter_parking = df_filter_week.copy()
            format_print(f"{my_name}: reloading model without filters", "yellow")
            cache_time = True
        # or use the cached data
        else:
            format_print(f"{my_name}: using cached data", "purple")
            record_metric(f"{my_name}: building finished", "cpu", ps.cpu_percent(percpu=True))
            record_metric(f"{my_name}: building finished", "ram", ps.virtual_memory())
            elem = SUGGFASCEORARIEPARCHEGGIO_PARKING_CACHE[day_of_week].copy()
            elem["day_of_week"] = day_of_week
            elem["parking_lot"] = parking_lot

            return elem
            
    matrix = build_ratings_matrix_SuggeritoreFasceOrarieDaParcheggio(df_filter_parking, users, time_slots)
    # get the ranking of the top parkings
    top_timeslots = np.argsort(matrix.sum(axis=0))[::-1][:SUGGFASCEORARIEPARCHEGGIO_TOP_N].tolist()
    # normalize matrix
    matrix = decoupling_normalization(matrix)
    # build model
    (p, q) = pq_factor(matrix, SUGGFASCEORARIEPARCHEGGIO_LATENT_VARS, SUGGFASCEORARIEPARCHEGGIO_LEARNING_RATE, SUGGFASCEORARIEPARCHEGGIO_EPOCHS)
    elem = {
        "day_of_week": day_of_week,
        "parking_lot": parking_lot,
        "p_matrix": p,
        "q_matrix": q,
        "users_labels": users,
        "labels": time_slots,
        "ranking": top_timeslots,
    }
    
    if cache_week:
        SUGGFASCEORARIEPARCHEGGIO_WEEK_CACHE = elem.copy()
        format_print(f"{my_name}: cached week model", "purple")

    if cache_time:
        SUGGFASCEORARIEPARCHEGGIO_PARKING_CACHE[day_of_week] = elem.copy()
        format_print(f"{my_name}: cached time model", "purple")
    
    record_metric(f"{my_name}: building finished", "cpu", ps.cpu_percent(percpu=True))
    record_metric(f"{my_name}: building finished", "ram", ps.virtual_memory())
    
    format_print(f"{my_name}: building finished", "blue")

    return elem.copy()


def build_SuggeritoreFasceOrarieDaParcheggio_recommendation_model(time_slots, dataset_filepath):

    global model

    format_print("Building SuggeritoreFasceOrarieDaParcheggio recommender model", "bold green")

    pool = mp.Pool(CPU_CORES)

    df = load_dataset(dataset_filepath)

    # retrieve list of users
    users = df["user"].unique()

    # retrieve list of parkings
    parkings = df["parking_lot"].unique()

    # for each day of week
    for day_of_week in range(7):
        for parking in parkings:
            pool.apply_async(task, args=(day_of_week, parking, df, users, time_slots), callback=append_to_model)
        

    pool.close()
    pool.join()

    ret = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}}


    for elem in model:
        day_of_week = elem["day_of_week"]
        parking_lot = int(elem["parking_lot"])
        ret[day_of_week][parking_lot] = elem

    # convert ret to array
    final_ret = []
    for day_of_week in ret:
        final_ret.append(ret[day_of_week])

    record_metric("parking model end", "cpu", ps.cpu_percent(percpu=True))
    record_metric("parking model end", "ram", ps.virtual_memory())

    return final_ret
