from matplotlib import rcParams
import tensorflow
from util.util import *
from parameters import *
from util.MatrixFactorizationEngine import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import json
from util.util import build_ratings_matrix_SuggeritoreFasceOrarie, build_ratings_matrix_SuggeritoreFasceOrarieDaParcheggio, build_ratings_matrix_SuggeritoreParcheggi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf = tensorflow.compat.v1
tf.disable_v2_behavior()


rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Helvetica'
rcParams['text.latex.preamble'] = r'\boldmath'
rcParams['font.size'] = 18.0

MAX_EPOCHS = 120000


def bold(s):
    if isinstance(s, str):
        return "\\bf{" + s + "}"
    else:
        list = []
        for item in s:
            list.append(bold(item))
        return list


def pq_factor(ratings_mat, latent_vars, learning_rate):

    print(f"latent_vars: {latent_vars}, learning_rate: {learning_rate}")

    loss_array = []

    # dimensions of the ratings matrix
    n = len(ratings_mat)
    m = len(ratings_mat[0])

    # copy of the matrix
    r = ratings_mat

    # random init of user and item latent matrices
    user_matrix = np.random.rand(n, latent_vars)
    items_matrix = np.random.rand(m, latent_vars)

    # placeholder for the ratings matrix
    ratings = tf.placeholder(tf.float32, name="ratings")
    p_matrix = tf.Variable(user_matrix, dtype=tf.float32)
    q_matrix = tf.Variable(items_matrix, dtype=tf.float32)

    # product user and item latent matrices
    p_times_q = tf.matmul(p_matrix, q_matrix, transpose_b=True)

    # compute the initial loss
    squared_error = tf.square(p_times_q - ratings)
    loss = tf.reduce_sum(squared_error) / (n * m)

    # start the optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(MAX_EPOCHS):
        sess.run(train, {ratings: r})

        loss_val = sess.run(loss, {ratings: r})

        print(f"Completed at {round(100*i/MAX_EPOCHS)}%", end="\r")
        loss_array.append(loss_val)

    # get the final user and item latent matrices
    return loss_array, sess.run(p_matrix), sess.run(q_matrix)


def run_optimization(model, latent_vars, learning_rates):
    df = load_dataset(DATASET_FILE_NAME)
    df = filter_by_day_of_week(df, 0)
    users = df["user"].unique()
    parkings = df["parking_lot"].unique()

    if "SuggeritoreFasceOrarie" == model:
        ratings, _ = build_ratings_matrix_SuggeritoreFasceOrarie(df, users, 48)
    elif "SuggeritoreParcheggi" == model:
        ratings = build_ratings_matrix_SuggeritoreParcheggi(df, users, parkings)
    elif "SuggeritoreFasceOrarieDaParcheggio" == model:
        _, timeslots = build_ratings_matrix_SuggeritoreFasceOrarie(df, users, 48)
        ratings = build_ratings_matrix_SuggeritoreFasceOrarieDaParcheggio(df, users, timeslots)

    map = {}
    for l in latent_vars:
        map[l] = {}

    for l in latent_vars:
        for lr in learning_rates:
            val, p, q = pq_factor(ratings, l, lr)

            # check if p or q are all made of Nan
            if np.isnan(p).any() or np.isnan(q).any():
                print(f"p or q are all made of Nan")
            else:
                ret = [x.astype(float) for x in val]
                map[l][lr] = ret

    filename = f"out/plots/loss_plot_{model}_data"

    # save to file
    with open(filename + ".json", "w") as f:
        f.write(json.dumps(map))


PARAMS_DICT = {
    ("SuggeritoreFasceOrarie",       "fixed_latent_size",    "25"):(     (0, 100),    (0,120000),  5,   None),
    ("SuggeritoreFasceOrarie",       "fixed_latent_size",    "75"):(     (0, 5),    (0,120000),  None,   None),
    ("SuggeritoreFasceOrarie",       "fixed_learning_rate",  "0.1"):(    (0, 5),    (0,120000),  None,   None),
    ("SuggeritoreFasceOrarie",       "fixed_learning_rate",  "0.01"):(   (0, 100),  (0,120000),  5,     None),

    ("SuggeritoreFasceOrarieDaParcheggio", "fixed_latent_size",    "25"):(     (0, 50),   (0,120000),  5,     None),
    ("SuggeritoreFasceOrarieDaParcheggio", "fixed_latent_size",    "75"):(     (0, 5),    (0,120000),  None,   None),
    ("SuggeritoreFasceOrarieDaParcheggio", "fixed_learning_rate",  "0.1"):(    (0, 5),    (0,120000),  None,   None),
    ("SuggeritoreFasceOrarieDaParcheggio", "fixed_learning_rate",  "0.01"):(   (0, 200),   (0,120000),  5,     None),

    ("SuggeritoreParcheggi", "fixed_latent_size",    "25"):(     (0, 10),    (0,10000),  None,   600),
    ("SuggeritoreParcheggi", "fixed_latent_size",    "75"):(     (0, 10),    (0,10000),  None,   600),
    ("SuggeritoreParcheggi", "fixed_learning_rate",  "0.1"):(    (0, 10),    (0,600),    None,   None),
    ("SuggeritoreParcheggi", "fixed_learning_rate",  "0.01"):(   (0, 10),    (0,10000),  None,   600),


}

def show_optimization_data(filename, model, type, value):

    data = {}

    # load from file
    with open(filename + ".json", "r") as f:
        data = json.loads(f.read())

    data_to_plot = {}
    legend= []
    title = ""

    if type == "fixed_learning_rate":
        title = f"Learning rate: {value}"
        for l in data:
            for lr in data[l]:
                if str(lr) == str(value):
                    data_to_plot[(l, lr)] = data[l][lr]
                    legend.append(f"spazio latente: {l}")

    elif type == "fixed_latent_size":
        title = f"Spazio latente: {value}"
        for l in data:
            for lr in data[l]:
                if str(l) == str(value):
                    data_to_plot[(l, lr)] = data[l][lr]
                    legend.append(f"learning rate: {lr}")

    # plot
    fig, ax = plt.subplots(1, 1)

    # plot target precision
    threshold = 0.000025 * 10**4
    ax.axhline(y=threshold, color='k', linestyle='--')
    legend = ["Precisione target"] + legend

    # plot data
    for m in data_to_plot:
        ax.plot([ 10 ** 4 * x for x in data_to_plot[m]])
    
    ax.set_xlabel(bold("Epoche"))
    ax.set_ylabel(bold("loss ($ \\times 10 ^{-4})$"))

    if ((model, type, str(value)) in PARAMS_DICT):
        ax.set_ylim(PARAMS_DICT[(model, type, str(value))][0])
        ax.set_xlim(PARAMS_DICT[(model, type, str(value))][1])
        
        if PARAMS_DICT[(model, type, str(value))][2] != None:
            ax.axhline(y=PARAMS_DICT[(model, type, str(value))][2], color='r', linestyle='--', markersize=1)
        
        if PARAMS_DICT[(model, type, str(value))][3] != None:
            ax.axvline(x=PARAMS_DICT[(model, type, str(value))][3], color='r', linestyle='--', markersize=1)
    
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_title(bold(title))
    fig.legend(bold(legend), loc="upper left",
               bbox_to_anchor=(0.91, 0.9), prop={'size': 11})
    fig.savefig(f"{filename}_{type}_{value}.ps", bbox_inches="tight")


if __name__ == "__main__":

    run_optimization("SuggeritoreFasceOrarie", [10,25,50,75], [0.1, 0.075, 0.05, 0.025, 0.01])
    run_optimization("SuggeritoreFasceOrarieDaParcheggio", [10,25,50,75], [0.1, 0.075, 0.05, 0.025, 0.01])
    run_optimization("SuggeritoreParcheggi", [10,25,50,75], [0.1, 0.075, 0.05, 0.025, 0.01])
     
    show_optimization_data("out/plots/loss_plot_SuggeritoreFasceOrarie_data", "SuggeritoreFasceOrarie", "fixed_latent_size", 25)
    show_optimization_data("out/plots/loss_plot_SuggeritoreFasceOrarie_data", "SuggeritoreFasceOrarie", "fixed_latent_size", 75)
    show_optimization_data("out/plots/loss_plot_SuggeritoreFasceOrarie_data", "SuggeritoreFasceOrarie", "fixed_learning_rate", 0.1)
    show_optimization_data("out/plots/loss_plot_SuggeritoreFasceOrarie_data", "SuggeritoreFasceOrarie", "fixed_learning_rate", 0.01)

    show_optimization_data("out/plots/loss_plot_SuggeritoreFasceOrarieDaParcheggio_data", "SuggeritoreFasceOrarieDaParcheggio", "fixed_latent_size", 25)
    show_optimization_data("out/plots/loss_plot_SuggeritoreFasceOrarieDaParcheggio_data", "SuggeritoreFasceOrarieDaParcheggio", "fixed_latent_size", 75)
    show_optimization_data("out/plots/loss_plot_SuggeritoreFasceOrarieDaParcheggio_data", "SuggeritoreFasceOrarieDaParcheggio", "fixed_learning_rate", 0.1)
    show_optimization_data("out/plots/loss_plot_SuggeritoreFasceOrarieDaParcheggio_data", "SuggeritoreFasceOrarieDaParcheggio", "fixed_learning_rate", 0.01)

    show_optimization_data("out/plots/loss_plot_SuggeritoreParcheggi_data", "SuggeritoreParcheggi", "fixed_latent_size", 25)
    show_optimization_data("out/plots/loss_plot_SuggeritoreParcheggi_data", "SuggeritoreParcheggi", "fixed_latent_size", 75)
    show_optimization_data("out/plots/loss_plot_SuggeritoreParcheggi_data", "SuggeritoreParcheggi", "fixed_learning_rate", 0.1)
    show_optimization_data("out/plots/loss_plot_SuggeritoreParcheggi_data", "SuggeritoreParcheggi", "fixed_learning_rate", 0.01)