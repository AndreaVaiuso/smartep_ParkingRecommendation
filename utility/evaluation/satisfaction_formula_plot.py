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

def s(d, dmin, m, n):
    return 1/(((abs(d)/dmin)/m)**n + 1)


xs = np.arange(0, 300, 0.01)
ys = [100* s(x, 30, 3, 5) for x in xs]

plt.xlabel(bold("Differenza dal valore misurato (minuti)"))
plt.ylabel(bold("Soddisfazione (\\%)"))
plt.grid(True)

plt.xticks(np.arange(0, 301, 30))
plt.gca().set_xticklabels([bold(str(x)) for x in np.arange(0, 301, 30)])
plt.gca().tick_params(axis='both', labelsize=15)

plt.plot(xs, ys)
plt.tight_layout()
plt.savefig('out/plots/satisfaction_formula_plot.ps')