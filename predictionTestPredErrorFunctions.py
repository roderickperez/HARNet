import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def testPrediction2(model, train, df):
    testPred = model.predict(
        start=len(train), end=len(df)-1)
    return testPred


def errorCalculation(test, testPred):
    mse = round(mean_squared_error(test, testPred), 10)
    mae = round(mean_absolute_error(test, testPred), 10)
    r2 = round(r2_score(test, testPred), 10)
    rmse = round(np.sqrt(mean_squared_error(test, testPred)), 10)

    return mse, mae, r2, rmse

    # modelResultsExpander.metric("MSE", mse)
    # modelResultsExpander.metric("MAE", mae)
    # modelResultsExpander.metric("R2", r2)
    # modelResultsExpander.metric("RMSE", rmse)
