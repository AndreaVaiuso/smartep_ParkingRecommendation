from matplotlib import pyplot as plt

fig, ax = plt.subplots()
time = [60,25,15,14,14]
res_fo_satisf = [0.7472750722265729,0.7372750722265729,0.7172750722265725,0.6566282989019987,0.5523649272041903]

ax.plot(time,res_fo_satisf)
plt.show()