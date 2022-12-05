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

def bold(s):
    if isinstance(s, str):
        return "\\bf{" + s + "}"
    else:
        list = []
        for item in s:
            list.append(bold(item))
        return list

alphas = [0,0.25,0.5,0.75,1]


ts_to_park = [100*x for x in [0.979381443298969, 0.9742268041237113, 0.9690721649484536, 0.9639175257731959, 0.9716494845360825]] #ts2parkings
park_to_ts = [100*x for x in [0.39690721649484534, 0.4536082474226804, 0.3943298969072165, 0.4097938144329897, 0.422680412371134]] #parkings2ts
none_tspark = [100*x for x in [0.4793814432989691, 0.47680412371134023, 0.4845360824742268, 0.5051546391752577, 0.47680412371134023]] #id2ts

ts_to_park_sat = []
park_to_ts_sat = [100*x for x in [0.6104163405288834, 0.627412510597436, 0.6300694522138376, 0.6718976190349168, 0.647238844634329]]
none_tspark_sat = [100*x for x in [0.7373143105018882, 0.7180654764739387, 0.7147518255198532, 0.7290613370950643, 0.7557932308320837]]

for (values, sat, filename) in zip ([ts_to_park, park_to_ts, none_tspark], [ts_to_park_sat, park_to_ts_sat, none_tspark_sat] ,["out/plots/alpha_plot_ts--parkings.ps","out/plots/alpha_plot_parkings--ts.ps","out/plots/alpha_plot_none--ts,parking.ps"]):

    plt.xlabel(bold("Valori di $\\alpha$ per l'aggiornamento SML"))
    plt.ylabel(bold("Punteggio (\\%)"))
    plt.ylim(0,100)
    plt.grid(True, axis='y')

    plt.plot(alphas, values, 'x-', label=bold('accuratezza'))
    
    if not "ts--parkings" in filename:
        plt.plot(alphas, sat, 'o-', label=bold('soddisfazione') + '\n' + bold('fascia oraria'))
    
    plt.legend(prop={'size': 11})
    plt.savefig(filename)
    plt.cla()
