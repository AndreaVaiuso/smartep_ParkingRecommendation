import flask
from flask import request, jsonify

from utilities import bcolors
from Model.ModelHandler import *
from utility.recommendations import *
from utility.evaluation.validation import *
import re
import sys

NAME = "RecommendationPredictor"
VERSION = "1.0.0"

app = flask.Flask(__name__)
app.config["DEBUG"] = True

SERVER_MODEL = None

# preprocess request and check if malformed
def check_request(request, necessary_fields):
    if not request.data:
        return jsonify({'msg': 'Error: json content not found in request'}), 400

    data = None
    try:
        data = request.get_json()
    except:
        return jsonify({'msg': 'Err: malformed json content'}), 400

    if not all(k in data for k in necessary_fields):
        return jsonify({'msg': f'Error: json must contain the {necessary_fields} fields'}), 400

    return data, 200


# Build the model
@app.route('/model/build', methods=['POST'])
def model_build():

    global SERVER_MODEL

    data, code = check_request(request, ['dataset_filepath', 'output_model_filepath'])
    if code != 200:
        return data, code
    
    try:
        SERVER_MODEL = init_model(None, data['dataset_filepath'], data['output_model_filepath'])
    except Exception as e:
        return jsonify({'msg': 'Error: failed to initialize model', 'exception': str(e)}), 400

    return jsonify({"msg": "Model built correctly"}), 200


# Load the model
@app.route('/model/load', methods=['POST'])
def model_load():
    
    global SERVER_MODEL
    
    data, code = check_request(request, ['model_filepath'])
    if code != 200:
        return data, code

    try:
        SERVER_MODEL = init_model(data['model_filepath'], None, None)
    except:
        return jsonify({'msg': 'Error: failed to initialize model (model not found?)'}), 400

    return jsonify({"msg": "Model loaded correctly"}), 200


# SML update model
@app.route('/model/sml_update', methods=['POST'])
def model_sml_update():
    global SERVER_MODEL

    if SERVER_MODEL is None:
        return jsonify({'msg': 'Error: initialize model first'}), 400
    
    data, code = check_request(request, ['new_dataset_filepath', 'output_model_filepath', 'combination_coefficient'])
    
    if code != 200:
        return data, code

    try:
        SERVER_MODEL = model_update(data['new_dataset_filepath'], data['output_model_filepath'], float(data['combination_coefficient']))
    except:
        return jsonify({'msg': 'Error: failed to update model (check if new dataset and folder for updated model exist, or if alpha is a float number between 0 and 1)'}), 400

    return jsonify({"msg": "Model updated correctly"}), 200


# Perform a PFO recommendation
@app.route('/recommendation/pfo', methods=['POST'])
def recommend_pfo():
    global SERVER_MODEL

    if SERVER_MODEL is None:
        return jsonify({'msg': 'Error: initialize model first'}), 400
    
    data, code = check_request(request, ['user_id', 'user_db_filepath'])
    if code != 200:
        return data, code

    try:
        if 'day_of_week' in data:
            recommendations = get_recommended_parking(SERVER_MODEL, data['user_db_filepath'] ,int(data['user_id']), None, int(data['day_of_week']))
        else:
            recommendations = get_recommended_parking(SERVER_MODEL, data['user_db_filepath'], int(data['user_id']), None)
    except Exception as e:
        return jsonify({'msg': 'Error: failed to get recommendations', 'exception':str(e)}), 500
    
    return jsonify({"recommendations": recommendations}), 200



@app.route('/recommendation/p', methods=['POST'])
def recommend_p():
    global SERVER_MODEL

    if SERVER_MODEL is None:
        return jsonify({'msg': 'Error: initialize model first'}), 400
    
    data, code = check_request(request, ['user_id', 'time_slot', 'user_db_filepath'])
    if code != 200:
        return data, code

    if len(data['time_slot']) != 8:
        return jsonify({'msg': f'Error: time slot must be in the format hh:mm:ss, and aligned to the time slot resolution: {TIMESLOT_RES_MIN}'}), 400

    try:
        if 'day_of_week' in data:
            recommendations = get_recommended_parking(SERVER_MODEL, data['user_db_filepath'], int(data['user_id']), [data['time_slot']], int(data['day_of_week']))
        else:
            recommendations = get_recommended_parking(SERVER_MODEL, data['user_db_filepath'], int(data['user_id']), [data['time_slot']])
    except Exception as e:
        return jsonify({'msg': 'Error: failed to get recommendations', 'exception': str(e)}), 500

    return jsonify({"recommendations": recommendations}), 200



@app.route('/recommendation/fo', methods=['POST'])
def recommend_fo():
    global SERVER_MODEL

    if SERVER_MODEL is None:
        return jsonify({'msg': 'Error: initialize model first'}), 400
    
    data, code = check_request(request, ['user_id', 'parking_lot', 'user_db_filepath'])
    if code != 200:
        return data, code

    try:
        if 'day_of_week' in data:
            recommendations = get_recommended_timeslot(SERVER_MODEL, data['user_db_filepath'], int(data['user_id']), [data['parking_lot']], int(data['day_of_week']))
        else:
            recommendations = get_recommended_timeslot(SERVER_MODEL, data['user_db_filepath'], int(data['user_id']), [data['parking_lot']])
    except Exception as e:
        return jsonify({'msg': 'Error: failed to get recommendations', 'exception': str(e)}), 500

    return jsonify({"recommendations": recommendations}), 200


@app.route('/model/evaluation', methods=['POST'])
def model_evaluation():
    global SERVER_MODEL

    if SERVER_MODEL is None:
        return jsonify({'msg': 'Error: initialize model first'}), 400
    
    data, code = check_request(request, ['user_db_filepath', 'validation_set_filepath', 'out_plots_dirpath'])
    if code != 200:
        return data, code

    try:
        results_pfo = validation(SERVER_MODEL, "data/user_habits_reduced.csv", "data/user_habits_reduced_valid.csv", "pfo")
        results_p = validation(SERVER_MODEL, "data/user_habits_reduced.csv", "data/user_habits_reduced_valid.csv", "p")
        results_fo = validation(SERVER_MODEL, "data/user_habits_reduced.csv", "data/user_habits_reduced_valid.csv", "fo")

        results = {
            "pfo": results_pfo,
            "p": results_p,
            "fo": results_fo
        }
        validation_plots(results, data['out_plots_dirpath'])

        ret = {"pfo" : {}, "p": {}, "fo": {}}
        for usecase in results:
            elem = results[usecase]
            ret[usecase] = {
                "Accuracy": elem["accuracy"][-1],
                "Hit Rate": elem["hit_rate"][-1],
                "Mean Average Precision": elem["precision"][-1],
                "Mean Average Recall": elem["recall"][-1],
                "F1-Score": elem["f1-score"],
                "Coverage": elem["coverage"][-1],
                "Satisfaction": elem["satisfaction"][-1]
            }

    except Exception as e:
        print(e.with_traceback(None))
        return jsonify({'msg': 'Error: failed to evaluate model', 'exception': str(e)}), 500

    return jsonify({"results": ret}), 200


if __name__ == '__main__':
    print("Starting up service: " + NAME + " - ver: " + VERSION)
    prt = 5000
    hst = "127.0.0.1"
    p = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    try:
        if sys.argv[1] is None:
            print("Usage: " + sys.argv[0] + " [address] [port]")
            print("Starting server using default address:",hst)
        else:
            temp = str(sys.argv[1])
            if p.match(temp):
                hst = temp
                print("Starting server using address:",hst)
            else:
                print(bcolors.FAIL+"ERROR: You must specify a valid address (like 127.0.0.1)"+bcolors.ENDC)
                print(bcolors.WARNING+"Starting server using default address: "+str(hst)+bcolors.ENDC)
        if sys.argv[2] is None:
            print(bcolors.WARNING+"Starting server using default server port: "+prt+bcolors.ENDC)
        else:
            try:
                prt = int(sys.argv[2])
                print("Starting server using port:",prt)
            except:
                print(bcolors.FAIL+"ERROR: You must specify an integer value for server port."+bcolors.ENDC)
                print(bcolors.WARNING+"Starting server using default server port: "+str(prt)+bcolors.ENDC)
    except IndexError:
        print(bcolors.WARNING+"Usage: " + sys.argv[0] + " [address] [port]"+bcolors.ENDC)
        print("Starting server using default server port:",prt)
    app.run(host=hst, port=prt)
