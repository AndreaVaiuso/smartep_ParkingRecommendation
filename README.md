# Notes
- Dataset should contain the column names: 'user', 'date', 'time', 'parking_lot'
- To run the server it is sufficient to run the script server.py
- The file 'parameters.py' contains the model building/update parameters

# API
Parameters should be passed as JSON in the request body.
## Model build
Build the model from a dataset and load it
- **endpoint**: /model/build
- **method**: POST
- **params**:
    - **dataset_filepath**: path to the dataset file for building the model
    - **output_model_filepath**: path for saving the model
- **return**:
    - **msg**: message indicating the reason of failure or success

## Model load
Load the model from an existing file
- **endpoint**: /model/load
- **method**: POST
- **params**:
    - **model_filepath**: path to the model file
- **return**:
    - **msg**: message indicating the reason of failure or success

## Model SML update
Update the model with the SML algorithm
- **endpoint**: /model/sml_update
- **method**: POST
- **params**:
    - **new_dataset_filepath**: path to the dataset file for building the model
    - **output_model_filepath**: path for saving the model
    - **combination_coefficient**: float value between 0 and 1 (alpha in SML algorithm)
- **return**:
    - **msg**: message indicating the reason of failure or success

## PFO recommendation
Get the PFO recommendations for a given user
- **endpoint**: /recommendation/pfo
- **method**: POST
- **params**:
    - **user_id**: user id
    - **user_db_filepath**: path to the database used to get user's observation (recommended: containing dataset used for training + newly collected data)
    - **day_of_week** [optional]: day of week for the recommendation (default: today)
- **return**:
    - **msg**: message indicating the reason of failure (if HTTP code != 200)
    - **recommendations**: list (timeslot, parking lot) of recommendations (if HTTP code == 200)

## FO recommendation
Get the FO recommendations for a given user
- **endpoint**: /recommendation/fo
- **method**: POST
- **params**:
    - **user_id**: user id
    - **parking_lot**: parking lot id for the recommendation
    - **user_db_filepath**: path to the database used to get user's observation (recommended: containing dataset used for training + newly collected data)
    - **day_of_week** [optional]: day of week for the recommendation (default: today)
- **return**:
    - **msg**: message indicating the reason of failure (if HTTP code != 200)
    - **recommendations**: list (timeslot, parking lot) of recommendations (if HTTP code == 200)

## P recommendation
Get the P recommendations for a given user
- **endpoint**: /recommendation/p
- **method**: POST
- **params**:
    - **user_id**: user id
    - **time_slot**: time slot for the recommendation (in the format HH:MM:SS, and aligned to time slot resolution (default: 30 minutes))
    - **user_db_filepath**: path to the database used to get user's observation (recommended: containing dataset used for training + newly collected data)
    - **day_of_week** [optional]: day of week for the recommendation (default: today)
- **return**:
    - **msg**: message indicating the reason of failure (if HTTP code != 200)
    - **recommendations**: list (timeslot, parking lot) of recommendations (if HTTP code == 200)


## Model evaluation
Get the metrics for the evaluation of the model
- **endpoint**: /model/evaluation
- **method**: POST
- **params**:
    - **user_db_filepath**: path to the database used to get user's observation (recommended: containing dataset used for training + newly collected data)
    - **out_plots_filepath**: path for the folder where the plots will be saved
    - **validation_set_filepath**: path to the set of data used for validation
- **return**:
    - **msg**: message indicating the reason of failure (if HTTP code != 200)
    - **results**: dictionary containing the values of the metrics Accuracy, Hit Rate, Mean Average Precision, Mean Average Recall, F1-Score, Coverage (if HTTP code == 200)