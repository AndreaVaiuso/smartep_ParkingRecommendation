from utility.util import *
from utility.MatrixFactorizationEngine import decoupling_normalization, sml_update
from parameters import *
import multiprocessing as mp
import psutil as ps

model_rebuilt = []


def append(elem):
    global model_rebuilt
    model_rebuilt.append(elem)


def task(week_model, new_data, day_of_week, combination_coefficient):
    try:
        my_name = f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}"

        format_print(f"{my_name}: sml update started", "green")

        # get users, timeslots and parkings

        users = week_model["users"]
        num_time_slots = len(week_model["labels"])

        # find new users which are in the new data but not in the model
        new_users = list(set(new_data["user"]) - set(week_model["users"]))

        # add new users to the model
        week_model["users"] += new_users
        time_slot_matrix, time_slot_labels = build_ratings_matrix_SuggeritoreFasceOrarie(new_data, week_model["users"], num_time_slots)
        time_slot_matrix = decoupling_normalization(time_slot_matrix)
        p, q = sml_update(week_model["p_matrix"], week_model["q_matrix"], time_slot_matrix, new_users, combination_coefficient, SUGGFASCEORARIE_LEARNING_RATE, SUGGFASCEORARIE_EPOCHS)

        # update ranking
        ranking = np.argsort(time_slot_matrix.sum(axis=0))[::-1][:SUGGFASCEORARIE_TOP_N].tolist()

        record_metric(f"{my_name}: sml update finished", "cpu", ps.cpu_percent(percpu=True))
        record_metric(f"{my_name}: sml update finished", "ram", ps.virtual_memory())
        
        format_print(f"{my_name}: sml update finished", "green")
    
    except Exception as e:
        format_print(e, "red")
        exit()

    return (day_of_week, p, q, ranking)


def sml_update_SuggeritoreFasceOrarie(model, new_data_filename, combination_coefficient):
    global model_rebuilt

    format_print("Updating SuggeritoreFasceOrarie model", "bold green")

    new_data = load_dataset(new_data_filename)

    pool = mp.Pool(CPU_CORES)

    # recompute time slot model
    for i, week_model in enumerate(model):
        pool.apply_async(task, args=(week_model, new_data, i, combination_coefficient), callback=append)

    pool.close()
    pool.join()

    format_print("SuggeritoreFasceOrarie model updated", "bold green")

    record_metric("SuggeritoreFasceOrarie model SML update end", "cpu", ps.cpu_percent(percpu=True))
    record_metric("SuggeritoreFasceOrarie model SML update end", "ram", ps.virtual_memory())

    return model_rebuilt
