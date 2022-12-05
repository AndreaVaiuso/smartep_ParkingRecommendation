from utility.util import *
from utility.MatrixFactorizationEngine import decoupling_normalization, sml_update
from parameters import *
import multiprocessing as mp
import psutil as ps

model_rebuilt = []


def append(elem):
    global model_rebuilt
    model_rebuilt.append(elem)


def task(time_slot_model, new_data, day_of_week, time_slot, combination_coefficient):

    try:

        my_name = f"{time_slot}@{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}"

        format_print(f"{my_name}: sml update started", "green")

        users = time_slot_model["users_labels"]
        parkings = time_slot_model["labels"]

        # find new users which are in the new data but not in the model
        new_users = list(set(new_data["user"]) - set(time_slot_model["users_labels"]))

        # add new users to the model
        time_slot_model["users_labels"] += new_users

        parking_matrix = build_ratings_matrix_SuggeritoreParcheggi(new_data, time_slot_model["users_labels"], parkings)
        parking_matrix = decoupling_normalization(parking_matrix)            
        
        p, q = sml_update(time_slot_model["p_matrix"], time_slot_model["q_matrix"], parking_matrix, new_users, combination_coefficient, SUGGPARCHEGGI_LEARNING_RATE, SUGGPARCHEGGI_EPOCHS)

        # update ranking
        ranking = np.argsort(parking_matrix.sum(axis=0))[::-1][:SUGGPARCHEGGI_TOP_N].tolist()

        record_metric(f"{my_name}: sml update finished", "cpu", ps.cpu_percent(percpu=True))
        record_metric(f"{my_name}: sml update finished", "ram", ps.virtual_memory())

        format_print(f"{my_name}: sml update finished", "green")

    except Exception as e:
        format_print(e, "red")
        exit()

    return (day_of_week, time_slot, p, q, ranking)


def sml_update_SuggeritoreParcheggi(model, new_data_filename, combination_coefficient):
    global model_rebuilt

    format_print("Updating SuggeritoreParcheggi model", "bold green")

    new_data = load_dataset(new_data_filename)

    pool = mp.Pool(CPU_CORES)

    # recompute time slot model
    for i, week_model in enumerate(model):
        for time_slot in week_model:
            time_slot_model = week_model[time_slot]
            pool.apply_async(task, args=(time_slot_model, new_data, i, time_slot, combination_coefficient), callback=append)
        
    pool.close()
    pool.join()

    format_print("SuggeritoreParcheggi model updated", "bold green")

    record_metric("SuggeritoreParcheggi model SML update end", "cpu", ps.cpu_percent(percpu=True))
    record_metric("SuggeritoreParcheggi model SML update end", "ram", ps.virtual_memory())

    return model_rebuilt
