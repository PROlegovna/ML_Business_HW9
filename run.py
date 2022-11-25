import logging
from logging.handlers import RotatingFileHandler
import os
import pickle
import time

import pandas as pd

import flask

import models

from models import NumberTaker, ExperienceTransformer, NumpyToDataFrame


def load_model(model_path):
    # load the pre-trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model


# Logging
logfile = 'model_api.log'
handler = RotatingFileHandler(filename=logfile, maxBytes=1048576, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Initialize Flask app
app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def general():
    return """Welcome to employee job change prediction API!"""


@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the view
    response = {'success': False}
    curr_time = time.strftime('[%Y-%b-%d %H:%M:%S]')

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == 'POST':
        request_json = flask.request.get_json()

        input_data = pd.DataFrame({
            'Id': request_json.get('Id', ''),
            'EmployeeName': request_json.get('EmployeeName', ''),
            'JobTitle': float(request_json.get('JobTitle', '')),
            'BasePay': request_json.get('BasePay', ''),
            'OvertimePay': request_json.get('OvertimePay', ''),
            'OtherPay': request_json.get('OtherPay', ''),
            'Benefits': request_json.get('Benefits', ''),
            'TotalPay': request_json.get('TotalPay', ''),
            'Year': request_json.get('Year', ''),
            'Agency': request_json.get('Agency', ''),
            'Status': request_json.get('Status', ''),
            'Notes': request_json.get('Notes', '')),
        }, index=[0])
        logger.info(f'{curr_time} Data: {input_data.to_dict()}')

        try:
            # Predict the result
            preds = model.predict_proba(input_data)
            response['predictions'] = round(preds[:, 1][0], 5)
            # Request successful
            response['success'] = True
        except AttributeError as e:
            logger.warning(f'{curr_time} Exception: {str(e)}')
            response['predictions'] = str(e)
            # Request unsuccessful
            response['success'] = False

    # return the data dictionary as a JSON response
    return flask.jsonify(response)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    port = int(os.environ.get('FLASK_SERVER_PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)