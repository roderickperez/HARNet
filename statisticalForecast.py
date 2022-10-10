from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import pandas as pd


def autoRegressioModel(train, lagsSelection):
    st.subheader("Auto Regression Model")
    model = AutoReg(train.values, lags=lagsSelection).fit()
    return model


def ARtestPrediction1(model, train, df):
    testPred = model.predict(
        start=len(train), end=len(df)-1)
    return testPred


def ARFuturePrediction1(model, df, futurePrediction):
    indexFutureDates = pd.DataFrame(pd.date_range(
        start=pd.to_datetime(df.index[-1]), periods=futurePrediction+1, freq='D'))
    futurePred = pd.DataFrame(model.predict(
        start=len(df), end=len(df)+futurePrediction))
    futurePred = pd.concat([indexFutureDates, futurePred],
                           ignore_index=True, axis=1)
    futurePred = futurePred.set_index(futurePred[0], drop=True, inplace=False)
    futurePred.rename(columns={1: 'FuturePrediction'}, inplace=True)
    return futurePred


def futurePrediction(model, df, daysForecast):
    indexFutureDates = pd.DataFrame(pd.date_range(
        start=pd.to_datetime(df.index[-1]), periods=futurePrediction+1, freq='D'))
    futurePred = pd.DataFrame(model.predict(
        start=len(df), end=len(df)+daysForecast))
    futurePred = futurePred.reset_index(drop=True)
    futurePred = pd.concat([indexFutureDates, futurePred],
                           ignore_index=True, axis=1)
    futurePred = futurePred.set_index(futurePred[0], drop=True, inplace=False)
    futurePred.rename(columns={1: 'FuturePrediction'}, inplace=True)
    return futurePred
