# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import *
from bigdl.nn.layer import *
from zoo.common.nncontext import *
from zoo.pipeline.nnframes.nn_classifier import *


prefix = '/opt/ml/'
model_path = os.path.join(os.path.join(prefix, 'model'), "logstic_regression")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    sparkConf = SparkConf().setAppName("Sagemaker Zoo example")
    sc = init_nncontext(sparkConf)
    sql_context = SQLContext(sc)

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        model_def = os.path.join(model_path, 'module')
        model_weights = os.path.join(model_path, 'weight')
        if cls.model == None:
            cls.model = Model.loadModel(model_def, model_weights)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        model = cls.get_model()
        classifier_model = NNClassifierModel(model, SeqToTensor([4])) \
            .setBatchSize(4)
        return classifier_model.transform(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    df = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        lines = s.readlines()
        features = [map(lambda x: float(x), line.split(",")) for line in lines]

        schema = StructType([
            StructField("a", FloatType()),
            StructField("b", FloatType()),
            StructField("c", FloatType()),
            StructField("d", FloatType()),

        ])
        raw = ScoringService.sql_context.createDataFrame(features, schema=schema)
        df = raw.withColumn("features", array("a", "b", "c", "d"))
        print('Invoked with {} records'.format(len(lines)))
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    # Do the prediction
    predictions = ScoringService.predict(df)
    output = predictions.collect()

    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    pd.DataFrame({'results':output}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    print(result)

    return flask.Response(response=result, status=200, mimetype='text/csv')
