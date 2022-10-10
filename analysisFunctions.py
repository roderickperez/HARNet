import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statisticalFunctions import *
from plottingFunctions import *


def seasonalDecomposition(df, variable):

    seasonalParametersExpander = st.sidebar.expander(
        "Seasonal Decomposition Parameters")

    seasonalModel = seasonalParametersExpander.radio("Select Seasonal Model", [
        "Additive", "Multiplicative"], horizontal=True)
    periodSelection = seasonalParametersExpander.slider(
        'Select Period', 1, 25, 10)

    if seasonalModel == "Additive":
        model = "additive"
    else:
        model = "multiplicative"

    decomposition = seasonal_decompose(
        x=df[variable], model=model, extrapolate_trend='freq', period=periodSelection)
    plt.rcParams.update({'figure.figsize': (14, 12)})
    plt.rcParams.update({'font.size': 14})
    plt.style.context('dark_background')
    fig = decomposition.plot()
    st.plotly_chart(fig)


def analysisOptions(df, variable, analysisSelection):
    if analysisSelection == 'ADF Test':
        adTest(df[variable])

    elif analysisSelection == 'PACF':
        lagsSelection = st.sidebar.slider('Select Lags', 1, 25, 10)
        pacfValues = st.sidebar.radio("Show Partial Autocorrelation Coefficient", [
            "Yes", "No"], index=1, horizontal=True)
        pacf = plot_pacf(df[variable], lags=lagsSelection)

        if pacfValues == 'Yes':
            st.sidebar.write(
                "Partial Autocorrelation coefficient:", df[variable])

        st.pyplot(pacf)

    elif analysisSelection == 'ACF':
        lagsSelection = st.sidebar.slider('Select Lags', 1, 25, 10)
        acfValues = st.sidebar.radio("Show Autocorrelation Coefficients", [
            "Yes", "No"], index=1, horizontal=True)
        acf = plot_acf(df[variable], lags=lagsSelection)

        if acfValues == 'Yes':
            st.sidebar.write("Autocorrelation Coefficients:", df[variable])
        st.pyplot(acf)

    elif analysisSelection == 'Seasonal Decomposition':
        seasonalDecomposition(df, variable)

    elif analysisSelection == 'Histogram':
        histogramSelection = st.sidebar.radio(
            "Select Histogram Type", ['Normal', 'Normalized'], index=0, horizontal=True)

        st.sidebar.info(
            'If the histograms plot it is a Gaussian Distribution, thus the data is stationary.')
        if histogramSelection == 'Normal':
            histogramPlot(df, variable)
        elif histogramSelection == 'Normalized':
            histogramNormalizedPlot(df, variable)

    elif analysisSelection == 'Mean & Variance Test':
        mean = variableMean(df, variable)
        variance = variableVariance(df, variable)

        st.metric('Mean:', mean)
        st.metric(
            'Variance:', variance)

        st.info(
            'If there is no difference between the mean and variance, which means the data is stationary and invariant.')
