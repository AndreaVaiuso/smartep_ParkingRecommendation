import matplotlib.pyplot as plt
import math
import datetime
import numpy as np

from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Helvetica'
rcParams['text.latex.preamble'] = r'\boldmath'
rcParams['font.size'] = 18.0

WINDOW_SIZE_S = 1

def bold(s):
    if isinstance(s, str):
        return "\\bf{" + s + "}"
    else:
        list = []
        for item in s:
            list.append(bold(item))
        return list

def load_metrics(filename):
    cpu_events_timestamps = []
    cpu_events_labels = []
    cpu_events_data = []

    ram_events_timestamps = []
    ram_events_labels = []
    ram_events_data = []

    with open(filename, 'r') as f:
        for line in f:
            timestamp = line.split(';')[0].strip()
            label = line.split(';')[1].strip()
            metric = line.split(';')[2].strip()
            data = line.split(';')[3].strip()[1:-1]

            if metric == 'cpu':
                cpus = [float(x) for x in data.split(',')]

                avg_val = sum(cpus) / len(cpus) 

                cpu_events_timestamps.append(timestamp)
                cpu_events_labels.append(label)
                cpu_events_data.append(avg_val)

            if metric == 'ram':
                value = float(data.split("used=")[1].split(",")[0])

                ram_events_timestamps.append(timestamp)
                ram_events_labels.append(label)
                ram_events_data.append(value)

    # convert timestamps to datetime and to seconds
    cpu_events_timestamps = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in cpu_events_timestamps]
    cpu_events_timestamps = [datetime.datetime.timestamp(x) for x in cpu_events_timestamps]

    ram_events_timestamps = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in ram_events_timestamps]
    ram_events_timestamps = [datetime.datetime.timestamp(x) for x in ram_events_timestamps]
    
    # normalize timestamps starting from 0
    min_timestamp = min(cpu_events_timestamps)
    cpu_events_timestamps = [x - min_timestamp for x in cpu_events_timestamps]

    #  normalize timestamps starting from 0
    min_ram = min(ram_events_timestamps)
    ram_events_timestamps = [x - min_ram for x in ram_events_timestamps]

    cpu_events = [(cpu_events_timestamps[i], cpu_events_data[i]) for i in range(len(cpu_events_data))]
    ram_events = [(ram_events_timestamps[i], ram_events_data[i]) for i in range(len(ram_events_data))]

    return cpu_events, ram_events

def stream_to_window_average(stream, window_size_s):
    max_timestamp = max([s[0] for s in stream])
    
    number_of_windows = math.ceil( max_timestamp / window_size_s)
    windows = [[] for _ in range(number_of_windows)]

    for elem in stream:
        timestamp = elem[0]
        value = elem[1]
        i = int(timestamp / window_size_s)
        windows[i].append(value)

    # compute average for each window
    windows_averages = []
    for window in windows:
        if len(window) == 0:
            windows_averages.append(0)
        else:
            windows_averages.append(sum(window) / len(window))

    # replace 0s with average of next window (psutil works like this)
    for i in list(range(len(windows_averages)))[::-1]:
        if windows_averages[i] == 0:
            windows_averages[i] = windows_averages[i+1]

    return windows_averages


def plot_cpus_rams(cpus_list, rams_list, label, legend):

    fig0, ax0 = plt.subplots(1, 1, figsize=(8, 5))
    
    # CPU plot
    for cpus in cpus_list:
        ax0.plot(range(len(cpus)), cpus)

    max_len = max([len(cpus) for cpus in cpus_list ] + [0])

    ax0.grid(True, axis='y')
    ax0.set_xticks(np.arange(0, max_len, 1))
    ax0.set_xticklabels([bold(f"{WINDOW_SIZE_S * x}") if x%150 == 0 else "" for x in np.arange(0, max_len, 1)  ], fontsize=11)
    ax0.set_xlabel(bold("Tempo trascorso (secondi)"))
    ax0.set_ylabel(bold("Utilizzo della CPU (\%)"))
    ax0.set_ylim(0,100)
    ax0.tick_params(axis='y', labelsize=15)
    fig0.legend(legend,prop={'size': 11}, loc="center right" , bbox_to_anchor=(0.95, 0.5))
    fig0.tight_layout()
    
    fig0.savefig(f"out/plots/{label}_cpu_usage.ps")
    
    # RAM plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    for rams in rams_list:
        rams = [(r - min(rams))/(1024 * 1024 * 1024) for r in rams]
        ax1.plot(range(len(rams)), rams)

    max_len = max([len(rams) for rams in rams_list] + [0])

    ax1.grid(True, axis='y')
    ax1.set_xticks(np.arange(0, max_len, 1))
    ax1.set_xticklabels([bold(f"{WINDOW_SIZE_S * x}") if x%150 == 0 else "" for x in np.arange(0, max_len, 1)  ], fontsize=11)
    ax1.set_xlabel(bold("Tempo trascorso (secondi)"))
    ax1.set_ylim(0,24*1024/1024)
    ax1.set_ylabel(bold("Utilizzo della RAM (GB)"))
    ax1.tick_params(axis='y', labelsize=15)

    ax2 = inset_axes(ax1,
                    width="87%", # width = 30% of parent_bbox
                    height=2., # height : 1 inch
                    loc=1)
    ax2.tick_params(axis='y', labelsize=15)
    for rams in rams_list:
        rams = [(r - min(rams))/(1024 * 1024 * 1024) for r in rams]
        ax2.plot(range(len(rams)), rams)
    ax2.set_xticks(np.arange(0, max_len, 1))
    ax2.set_xticklabels([bold(f"{WINDOW_SIZE_S * x}") if x%150 == 0 else "" for x in np.arange(0, max_len, 1)  ], fontsize=11)
    ax2.grid(True, axis='y')
    
    fig1.legend(legend, loc="upper left", bbox_to_anchor=(0.8, 0.9), prop={'size': 11})
    fig1.subplots_adjust(right=0.8)

    #fig1.tight_layout()
    fig1.savefig(f"out/plots/{label}_ram_usage.ps", bbox_inches="tight")

cpusfs, ramsfs = load_metrics("out/cloud/user_habits_reduced_train_metrics.csv")
cpusfs = stream_to_window_average(cpusfs, WINDOW_SIZE_S)
ramsfs = stream_to_window_average(ramsfs, WINDOW_SIZE_S)

cpus0, rams0 = load_metrics("out/cloud/user_habits_reduced_old_0_metrics.csv")
cpus0 = stream_to_window_average(cpus0, WINDOW_SIZE_S)
rams0 = stream_to_window_average(rams0, WINDOW_SIZE_S)

# cpus025, rams025 = load_metrics("out/user_habits_reduced_old_025_metrics.csv")
# cpus025 = stream_to_window_average(cpus025, WINDOW_SIZE_S)
# rams025 = stream_to_window_average(rams025, WINDOW_SIZE_S)

cpus05, rams05 = load_metrics("out/cloud/user_habits_reduced_old_05_metrics.csv")
cpus05 = stream_to_window_average(cpus05, WINDOW_SIZE_S)
rams05 = stream_to_window_average(rams05, WINDOW_SIZE_S)

# cpus075, rams075 = load_metrics("out/user_habits_reduced_old_075_metrics.csv")
# cpus075 = stream_to_window_average(cpus075, WINDOW_SIZE_S)
# rams075 = stream_to_window_average(rams075, WINDOW_SIZE_S)

cpus1, rams1 = load_metrics("out/cloud/user_habits_reduced_old_1_metrics.csv")
cpus1 = stream_to_window_average(cpus1, WINDOW_SIZE_S)
rams1 = stream_to_window_average(rams1, WINDOW_SIZE_S)

# plot_cpus_rams([cpusfs, cpus0, cpus025, cpus05, cpus075, cpus1], [ramsfs,  rams0, rams025, rams05, rams075, rams1], "cloud update", bold(["From scratch", "SML, $\\alpha=0$", "SML, $\\alpha=0.25$", "SML, $\\alpha=0.5$", "SML, $\\alpha=0.75$", "SML, $\\alpha=1$"]))
plot_cpus_rams([cpusfs, cpus0, cpus05, cpus1], [ramsfs,  rams0, rams05, rams1], "pc ufficio update", bold(["From scratch", "SML, $\\alpha=0$", "SML, $\\alpha=0.5$", "SML, $\\alpha=1$"]))
plt.cla()

cpusfs, ramsfs = load_metrics("out/cloud/user_habits_reduced_old_metrics.csv")
cpusfs = stream_to_window_average(cpusfs, WINDOW_SIZE_S)
ramsfs = stream_to_window_average(ramsfs, WINDOW_SIZE_S)

plot_cpus_rams([cpusfs], [ramsfs], "pc ufficio build", [])