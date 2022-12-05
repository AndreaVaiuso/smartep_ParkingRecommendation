# general parameters
VERBOSE = True # if set to False ==> no output at all
RECORD_METRICS = True # if set to True ==> record CPU/RAM metrics
CPU_CORES = 24 # parallelization degree for model building and update

# model parameters
TIMESLOT_RES_MIN = 30  # resolution of the time slot model (in minutes)

SUGGFASCEORARIE_TOP_N = 4# 3  # top N parkings get by the time slot model
SUGGFASCEORARIE_TOP_N_THRESHOLD = 0# 0.7  # threshold of confidence to consider a time slot as good
SUGGFASCEORARIE_USER_INPUTS_TO_MODEL = 3  # number of considered used time slots for each user

SUGGPARCHEGGI_TOP_N = 4# 3  # top N parkings get by recommend__parkings model
SUGGPARCHEGGI_TOP_N_THRESHOLD = 0# 0.5  # threshold of confidence to consider a parking as good
SUGGPARCHEGGI_USER_INPUTS_TO_MODEL = 3  # number of considered used parkings for each user

SUGGFASCEORARIEPARCHEGGIO_TOP_N = 4# 3  # top N parkings get by recommend__parkings model
SUGGFASCEORARIEPARCHEGGIO_TOP_N_THRESHOLD = 0# 0.5  # threshold of confidence to consider a parking as good
SUGGFASCEORARIEPARCHEGGIO_USER_INPUTS_TO_MODEL = 3  # number of considered used parkings for each user

OCCUPANCY_PREDICTOR_TOP_N = 3  # top N parkings get by recommend__parkings model
OCCUPANCY_PREDICTOR_PORT = 5001
OCCUPANCY_PREDICTOR_URL = f"http://127.0.0.1:{OCCUPANCY_PREDICTOR_PORT}/api/smartep_ia/occupation/predict"
OCCUPANCY_PREDICTOR_TARGET_OCCUPANCY = 0.05 # policy for tradeoff user/parking lot owner
OCCUPANCY_PREDICTOR_ENABLED = False # enable or disable the occupancy predictor

# hyperparameters
SUGGFASCEORARIE_LATENT_VARS = 75
SUGGFASCEORARIE_LEARNING_RATE = 0.1
SUGGFASCEORARIE_EPOCHS = 5000

SUGGPARCHEGGI_LATENT_VARS = 50
SUGGPARCHEGGI_LEARNING_RATE = 0.1
SUGGPARCHEGGI_EPOCHS = 500

SUGGFASCEORARIEPARCHEGGIO_LATENT_VARS = 75
SUGGFASCEORARIEPARCHEGGIO_LEARNING_RATE = 0.1
SUGGFASCEORARIEPARCHEGGIO_EPOCHS = 6000