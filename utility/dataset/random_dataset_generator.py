import uuid
from rich import print as rprint
import numpy as np
from datetime import datetime
import random

# input
USERS_NUM = 10
PARKING_LOTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
PARKINGS_NUM = 100

# generate USERS_NUM users with random string as id
USERS = [str(uuid.uuid4()) for _ in range(USERS_NUM)]

# generate PARKINGS_NUM parkings with (user, timestamp, parking_lot)
PARKINGS = []
for i in range(PARKINGS_NUM):
    if i % 100 == 0:
        print(f"{100*i/PARKINGS_NUM}%", end="\r")
    user = USERS[int(USERS_NUM * np.random.rand())]
    timestamp = datetime.fromtimestamp(random.uniform(1577833200, 1609455600))
    date = timestamp.strftime("%Y-%m-%d")
    time = timestamp.strftime("%H:%M:%S")
    parking_lot = PARKING_LOTS[int(len(PARKING_LOTS) * np.random.rand())]
    PARKINGS.append((user, date, time, parking_lot))

with open("parkings.csv", "w") as f:
    f.write("user,date,time,parking_lot\n")
    for parking in PARKINGS:
        f.write(f"{parking[0]},{parking[1]},{parking[2]},{parking[3]}\n")
