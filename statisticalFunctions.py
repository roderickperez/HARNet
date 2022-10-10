import streamlit as st
from statsmodels.tsa.stattools import adfuller


def adTest(df):
    dftest = adfuller(df, autolag='AIC')
    st.write("1. ADF: ", dftest[0])
    st.write("2. P-Value: ", dftest[1])
    st.write("3. Num of Lags: ", dftest[2])
    st.write(
        "4. Num of Observations Used for ADF Regression and Critical Values Calculation: ", dftest[3])
    st.write("5. Critical Values: ")
    for key, val in dftest[4].items():
        st.write("\t", key, ": ", val)


def variableMean(df, variable):
    mean = df[variable].mean()
    return mean


def variableVariance(df, variable):
    variance = df[variable].var()
    return variance
