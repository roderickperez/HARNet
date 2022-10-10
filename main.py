from msilib import sequence
from re import T
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date

import numpy as np
###############################
# Internal Libraries
from loadFunctions import *
from statisticalFunctions import *
from plottingFunctions import *
from dataRangesConvertions import *
from analysisFunctions import *
from statisticalForecast import *
from deepLearningModelFunctions import *
from volatilityFunctions import *
from utilFunctions import *
from predictionTestPredErrorFunctions import *
from util import *
# from model import *
# from main import *
##############################
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
# from keras.preprocessing.sequence import TimeseriesGenerator
############################
############################

plt.rcParams["figure.figsize"] = (6, 2)
plt.rcParams.update({'font.size': 6})
##############################
st.set_page_config(page_title="HARNet | UniWien Research Project",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")

#########################
# ---- Header ----
with st.container():
    st.title(':chart_with_upwards_trend: HarNet App')

# Sidebar

st.sidebar.image("images/uniWienLogo.png", use_column_width=True)

########################################
# Data Loading
dataSelection = st.sidebar.selectbox(
    "Dataset Selection", ["Yahoo Finance", "MAN", "USEPUINDXD", "VIXCLS", "Own Dataset"])

generalParametersExpander = st.sidebar.expander('General Session Parameters')


if dataSelection == "Yahoo Finance":
    stock = ("AAPL", "AMZN", "GOOG", "MSFT", "INTC", "CSCO", "NVDA", "AMD", "TSLA", "MU", "NFLX", "PYPL", "ADBE", "ADP", "ATVI", "BIDU", "CMCSA", "COST", "CRM", "CSC", "CSX", "CTSH", "CTXS", "CTX", "DISH", "DLTR", "EA", "EBAY", "EXPE", "FAST", "FISV", "GILD", "GOOGL", "GPRO", "HAS", "HOLX",
             "ILMN", "INTU", "ISRG", "KLAC", "LRCX", "MAR", "MAT", "MCHP", "MDLZ", "MNST", "MSI", "NTAP", "NTES", "NVDA", "ORLY", "PAYX", "PCAR", "PYPL", "QCOM", "REGN", "ROST", "SBUX", "SIRI", "SNPS", "STX", "SWKS", "TMUS", "TRIP", "TSLA", "TXN", "VRSK", "VRTX", "WBA", "WDC", "ZM", "ZTS")
    stockSelection = st.sidebar.selectbox("Select a stock", stock)

    dateStart = "2015-01-01"
    dateEnd = date.today().strftime("%Y-%m-%d")

    filename = None
    df = loadData(filename, dataSelection, stockSelection, dateStart, dateEnd)

elif dataSelection == 'MAN':
    filename = "data/MAN_data.csv"
    df = loadData(filename, dataSelection, stock=None,
                  dateStart=None, dateEnd=None)

    stocksList = df['Symbol'].unique()
    # Deafult value is DJI
    stockSelection = st.sidebar.selectbox(
        "Symbol", index=5, options=stocksList)

    # Group the data by stock symbol
    df_symbol = df.groupby(['Symbol'])
    df = df_symbol.get_group(stockSelection)
    df = df.drop('Symbol', axis=1)

elif dataSelection == 'USEPUINDXD':
    filename = "data/USEPUINDXD_data.csv"
    df = loadData(filename, dataSelection, stock=None,
                  dateStart=None, dateEnd=None)

elif dataSelection == 'VIXCLS':
    filename = "data/VIXCLS_data.csv"
    df = loadData(filename, dataSelection, stock=None,
                  dateStart=None, dateEnd=None)
    df = df['VIXCLS'].astype(float)
    df = pd.DataFrame(df, columns=['VIXCLS'])
################################
# Horizontal Menu
# ---- Header ----
with st.container():
    selected = option_menu(
        menu_title=None,
        options=["Data", "Stats", "Analysis",
                 "Volatility", "Plot", "Forecast"],
        icons=['table', 'clipboard-data', 'sliders',
               'graph-up', 'activity', 'share'],
        orientation="horizontal",
        default_index=0,
    )
########################################
if selected == 'Data':
    if dataSelection == "Yahoo Finance":
        st.subheader(f"Data: {dataSelection} | Stock: {stockSelection}")
        st.dataframe(df)
    elif dataSelection == "MAN":
        st.subheader(f"Data: {dataSelection} | Stock: {stockSelection}")
        st.dataframe(df)
    elif dataSelection == "Own Dataset":
        pass
        # if uploaded_file is None:
        #     st.subheader("Please Upload a Dataset")
        #     df = None
        # elif uploaded_file is not None:
        #     st.subheader(f"Data: Own Dataset | File: {uploaded_file.name}")
        #     st.dataframe(df)
    else:
        st.subheader(f"Data: {dataSelection}")
        st.dataframe(df)

elif selected == 'Stats':
    if dataSelection == "Yahoo Finance":
        st.subheader(f"Data: {dataSelection} | Stock: {stockSelection}")
        st.dataframe(df.describe())
    elif dataSelection == "MAN":
        st.subheader(f"Data: {dataSelection} | Stock: {stockSelection}")
        st.dataframe(df.describe())
    elif dataSelection == "Own Dataset":
        pass
    else:
        st.subheader(f"Data: {dataSelection}")
        st.dataframe(df.describe())

elif selected == 'Analysis':

    if dataSelection == "Yahoo Finance":
        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        analysisSelection = st.sidebar.radio(
            'Select Analysis Mode', ('ADF Test', 'PACF', 'ACF', 'Seasonal Decomposition', 'Histogram', 'Mean & Variance Test'))
        st.subheader(
            f"Data: {dataSelection} | Stock: {stockSelection} | Variable: {variable}")

        analysisOptions(df, variable, analysisSelection)

    elif dataSelection == "MAN":
        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        analysisSelection = st.sidebar.radio(
            'Select Analysis Mode', ('ADF Test', 'PACF', 'ACF', 'Seasonal Decomposition', 'Histogram', 'Mean & Variance Test'))
        st.subheader(
            f"Data: {dataSelection} | Stock: {stockSelection} | Variable: {variable}")

        analysisOptions(df, variable, analysisSelection)

    elif dataSelection == "Own Dataset":
        pass
    elif dataSelection == "VIXCLS":
        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        analysisSelection = st.sidebar.radio(
            'Select Analysis Mode', ('ADF Test', 'PACF', 'ACF', 'Seasonal Decomposition', 'Histogram', 'Mean & Variance Test'))
        st.subheader(
            f"Data: {dataSelection}")
        analysisOptions(df, variable, analysisSelection)
    else:
        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        analysisSelection = st.sidebar.radio(
            'Select Analysis Mode', ('ADF Test', 'PACF', 'ACF', 'Seasonal Decomposition', 'Histogram', 'Mean & Variance Test'))
        st.subheader(
            f"Data: {dataSelection}")
        analysisOptions(df, variable, analysisSelection)

elif selected == 'Volatility':
    if dataSelection == "Yahoo Finance" or dataSelection == 'MAN':

        # Horizontal Menu
        volatilityHorizontalMenu = option_menu(
            menu_title=None,
            options=["Data", "Plots"],
            icons=['table', 'graph-up'],
            orientation="horizontal",
            default_index=1,
        )

        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        st.subheader(
            f"Data: {dataSelection} | Stock: {stockSelection} | Variable: {variable}")

        volatilityParametersExpander = st.sidebar.expander(
            "Volatility Parameters")

        volatilitySelection = volatilityParametersExpander.radio(
            'Select Volatility Mode', ('Volatility', 'Historical Volatility', 'Realized Volatility'))

        volatilityRollingWindow = volatilityParametersExpander.slider(
            'Select RollingWindow', 1, 25, 5)

        variableName = volatilityParametersExpander.text_input(
            'Variable Name', value=str(f"CalcVolatility_{volatilitySelection}_{variable}_{volatilityRollingWindow}"))

        fileExportName = volatilityParametersExpander.text_input(
            'File Export Name', value='New Volatility File')
        fileExportName = fileExportName + '.csv'

        st.sidebar.info(
            'The recommended variability to calculate the volatility is the Adj Close variable.')

        if volatilitySelection == 'Volatility':
            df = volatility(df, variable, variableName)
        elif volatilitySelection == 'Historical Volatility':
            df = historicalVolatility(
                df, variable, volatilityRollingWindow, variableName)
        elif volatilitySelection == 'Realized Volatility':
            df = realizedVolatility(
                df, variable, volatilityRollingWindow, variableName)

        df_csv = to_excel(df)
        st.sidebar.download_button(label='📥 Download Current Result',
                                   data=df_csv,
                                   file_name=str(fileExportName))

        if volatilityHorizontalMenu == "Data":
            st.dataframe(df)
        elif volatilityHorizontalMenu == "Plots":
            plotDataFrame(df[variableName])

    else:
        st.subheader("This feature is not available for this dataset.")

elif selected == 'Plot':

    if dataSelection == "Yahoo Finance":
        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        st.subheader(
            f"Data: {dataSelection} | Stock: {stockSelection} | Variable: {variable}")
        plotDataFrame(df[variable])
    elif dataSelection == "MAN":
        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        st.subheader(
            f"Data: {dataSelection} | Stock: {stockSelection} | Variable: {variable}")
        plotDataFrame(df[variable])
    elif dataSelection == "Own Dataset":
        pass

    elif dataSelection == "VIXCLS":
        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        st.subheader(
            f"Data: {dataSelection}")
        plotDataFrame(df[variable])
    else:
        variable = st.sidebar.selectbox(
            "Select Variable to Analyze", df.columns)
        st.subheader(
            f"Data: {dataSelection}")
        plotDataFrame(df[variable])

elif selected == 'Forecast':
    variable = st.sidebar.selectbox(
        "Select Variable to Analyze", df.columns)

    modelTypeSelectionExpander = st.sidebar.expander("Model Type")

    modelType = modelTypeSelectionExpander.radio(
        'Select Mode', ('Statistical', 'Deep Learning'))

    if modelType == 'Statistical':

        statisticalModelMode = modelTypeSelectionExpander.selectbox(
            'Mode', ['Select Best Model (Auto)', 'Select Model'])
        modelResultsExpander = st.sidebar.expander("Model Results")

        if statisticalModelMode == 'Select Best Model (Auto)':
            if modelTypeSelectionExpander.button('Select Best Model (Auto)'):
                stepwise_fit = auto_arima(
                    df[variable], trace=True, suppress_warnings=True)
                st.write(stepwise_fit.summary())
        else:
            modelParametersExpander = st.sidebar.expander("Model Parameters")

            daysForecast = modelParametersExpander.slider(
                'Select Days to Forecast', 1, 360, 10)

            dataSelectionExpander = st.sidebar.expander(
                "Data Train | Test Split")
            train_size = dataSelectionExpander.slider(
                'Train Size', 0.0, 1.0, 0.8, 0.05)

            train, test = splitData(df, variable, train_size)

            dataSelectionExpander.write(f"Train Size: {train.shape}")
            dataSelectionExpander.write(f"Test Size: {test.shape}")

            modelSelection = modelTypeSelectionExpander.selectbox(
                "Select Statistical Model", ["AR", "MA", "ARIMA", "SARIMAX"])
            # # Horizontal Menu
            # forecastHorizontalMenu = option_menu(
            #     menu_title=None,
            #     options=["Model Summary", "Data", "Plots"],
            #     icons=['clipboard-data', 'table', 'graph-up'],
            #     orientation="horizontal",
            #     default_index=0,
            # )

            if modelSelection == 'AR':
                p = modelParametersExpander.slider(
                    'p', 0, 10, 1)
                if modelTypeSelectionExpander.button('Compute Selected Model'):

                    model = ARIMA(df[variable], order=(p, 0, 0))
                    model = model.fit()

                    testPred = model.predict(
                        start=len(train), end=len(df)-1)

                # if forecastHorizontalMenu == "Model Summary":
                    st.write("Model Summary")
                    st.write(model.summary())
                # elif forecastHorizontalMenu == "Data":
                    st.write('Test Predictions')
                    st.dataframe(testPred)

                    st.write('Metrics')
                    mse, mae, r2, rmse = errorCalculation(test, testPred)
                    st.write('MSE: ' + str(mse))
                    st.write('MAE: ' + str(mae))
                    st.write('R2: ' + str(r2))
                    st.write('RMSE: ' + str(rmse))

                    st.write('Forecast')
                    indexFutureDates = pd.DataFrame(pd.date_range(start=pd.to_datetime(
                        df.index[-1]), periods=daysForecast+1, freq='D'))
                    futurePred = model.predict(
                        start=len(df), end=len(df)+daysForecast)
                    futurePred = futurePred.reset_index(drop=True)
                    futurePred = pd.concat([indexFutureDates, futurePred],
                                           ignore_index=True, axis=1)
                    futurePred = futurePred.set_index(
                        futurePred[0], drop=True, inplace=False)
                    futurePred.rename(
                        columns={1: 'FuturePrediction'}, inplace=True)
                    futurePred.index = futurePred.index.date
                    futurePred.drop(columns=[0], inplace=True)

                    plotForecast(train, test, testPred, futurePred)

            elif modelSelection == 'MA':
                q = modelParametersExpander.slider(
                    'q', 0, 10, 1)
                if modelTypeSelectionExpander.button('Compute Selected Model'):

                    model = ARIMA(df[variable], order=(0, 0, q))
                    model = model.fit()

                    testPred = model.predict(
                        start=len(train), end=len(df)-1)

                # if forecastHorizontalMenu == "Model Summary":
                    st.write("Model Summary")
                    st.write(model.summary())
                # elif forecastHorizontalMenu == "Data":
                    st.write('Test Predictions')
                    st.dataframe(testPred)

                    st.write('Metrics')
                    mse, mae, r2, rmse = errorCalculation(test, testPred)
                    st.write('MSE: ' + str(mse))
                    st.write('MAE: ' + str(mae))
                    st.write('R2: ' + str(r2))
                    st.write('RMSE: ' + str(rmse))

                    st.write('Forecast')
                    indexFutureDates = pd.DataFrame(pd.date_range(start=pd.to_datetime(
                        df.index[-1]), periods=daysForecast+1, freq='D'))
                    futurePred = model.predict(
                        start=len(df), end=len(df)+daysForecast)
                    futurePred = futurePred.reset_index(drop=True)
                    futurePred = pd.concat([indexFutureDates, futurePred],
                                           ignore_index=True, axis=1)
                    futurePred = futurePred.set_index(
                        futurePred[0], drop=True, inplace=False)
                    futurePred.rename(
                        columns={1: 'FuturePrediction'}, inplace=True)
                    futurePred.index = futurePred.index.date
                    futurePred.drop(columns=[0], inplace=True)

                    plotForecast(train, test, testPred, futurePred)

            elif modelSelection == 'ARIMA':
                p = modelParametersExpander.slider(
                    'p', 0, 10, 1)
                d = modelParametersExpander.slider(
                    'd', 0, 10, 1)
                q = modelParametersExpander.slider(
                    'q', 0, 10, 1)
                if modelTypeSelectionExpander.button('Compute Selected Model'):

                    model = ARIMA(df[variable], order=(p, d, q))
                    model = model.fit()

                    testPred = model.predict(
                        start=len(train), end=len(df)-1)

                # if forecastHorizontalMenu == "Model Summary":
                    st.write("Model Summary")
                    st.write(model.summary())
                # elif forecastHorizontalMenu == "Data":
                    st.write('Test Predictions')
                    st.dataframe(testPred)

                    st.write('Metrics')
                    mse, mae, r2, rmse = errorCalculation(test, testPred)
                    st.write('MSE: ' + str(mse))
                    st.write('MAE: ' + str(mae))
                    st.write('R2: ' + str(r2))
                    st.write('RMSE: ' + str(rmse))

                    st.write('Forecast')
                    indexFutureDates = pd.DataFrame(pd.date_range(start=pd.to_datetime(
                        df.index[-1]), periods=daysForecast+1, freq='D'))
                    futurePred = model.predict(
                        start=len(df), end=len(df)+daysForecast)
                    futurePred = futurePred.reset_index(drop=True)
                    futurePred = pd.concat([indexFutureDates, futurePred],
                                           ignore_index=True, axis=1)
                    futurePred = futurePred.set_index(
                        futurePred[0], drop=True, inplace=False)
                    futurePred.rename(
                        columns={1: 'FuturePrediction'}, inplace=True)
                    futurePred.index = futurePred.index.date
                    futurePred.drop(columns=[0], inplace=True)

                    plotForecast(train, test, testPred, futurePred)

            elif modelSelection == 'SARIMAX':
                p = modelParametersExpander.slider(
                    'p', 0, 10, 1)
                d = modelParametersExpander.slider(
                    'd', 0, 10, 1)
                q = modelParametersExpander.slider(
                    'q', 0, 10, 1)
                modelParametersExpander.write("Seasonal Parameters")
                P = modelParametersExpander.slider(
                    'P', 1, 10, 1)
                D = modelParametersExpander.slider(
                    'D', 1, 10, 1)
                Q = modelParametersExpander.slider(
                    'Q', 1, 10, 1)
                M = modelParametersExpander.slider(
                    'M', 1, 12, 7)
                if modelTypeSelectionExpander.button('Compute Selected Model'):

                    model = SARIMAX(df[variable], order=(
                        p, d, q), seasonal_order=(P, D, Q, M))
                    model = model.fit()

                    testPred = model.predict(
                        start=len(train), end=len(df)-1)

                # if forecastHorizontalMenu == "Model Summary":
                    st.write("Model Summary")
                    st.write(model.summary())
                # elif forecastHorizontalMenu == "Data":
                    st.write('Test Predictions')
                    st.dataframe(testPred)

                    st.write('Metrics')
                    mse, mae, r2, rmse = errorCalculation(test, testPred)
                    st.write('MSE: ' + str(mse))
                    st.write('MAE: ' + str(mae))
                    st.write('R2: ' + str(r2))
                    st.write('RMSE: ' + str(rmse))

                    st.write('Forecast')
                    indexFutureDates = pd.DataFrame(pd.date_range(start=pd.to_datetime(
                        df.index[-1]), periods=daysForecast+1, freq='D'))
                    futurePred = model.predict(
                        start=len(df), end=len(df)+daysForecast)
                    futurePred = futurePred.reset_index(drop=True)
                    futurePred = pd.concat([indexFutureDates, futurePred],
                                           ignore_index=True, axis=1)
                    futurePred = futurePred.set_index(
                        futurePred[0], drop=True, inplace=False)
                    futurePred.rename(
                        columns={1: 'FuturePrediction'}, inplace=True)
                    futurePred.index = futurePred.index.date
                    futurePred.drop(columns=[0], inplace=True)

                    plotForecast(train, test, testPred, futurePred)

    elif modelType == 'Deep Learning':
        modelParametersExpander = st.sidebar.expander("Model Parameters")

        daysForecast = modelParametersExpander.slider(
            'Select Days to Forecast', 1, 360, 10)

        dataSelectionExpander = st.sidebar.expander(
            "Data Train | Test Split")
        train_size = dataSelectionExpander.slider(
            'Train Size', 0.0, 1.0, 0.8, 0.05)

        train, test = splitData(df, variable, train_size)

        dataSelectionExpander.write(f"Train Size: {train.shape}")
        dataSelectionExpander.write(f"Test Size: {test.shape}")

        modelSelection = modelTypeSelectionExpander.selectbox(
            "Select Statistical Model", ["LSTM", "HAR Net"])

        scaleData = dataSelectionExpander.radio("Scale Data", [
            "Yes", "No"], horizontal=True, index=0)

        if scaleData == "Yes":
            scaler = MinMaxScaler().fit(train)
        else:
            pass

        if modelSelection == "LSTM":
            n_input = modelParametersExpander.number_input(
                'Number of inputs', 1, 20, 7)
            n_features = modelParametersExpander.number_input(
                'Number of features', 1, 20, 1)
            batchSize = modelParametersExpander.number_input(
                'Batch Size', 1, 20, 1)
            epochs = modelParametersExpander.number_input('Epochs', 1, 100, 10)
            neurons = modelParametersExpander.number_input(
                'Neurons', 1, 100, 50)
            activationFunction = modelParametersExpander.radio(
                'Activation Function', ['relu', 'tanh', 'sigmoid'], index=0)
            optimizer = modelParametersExpander.radio(
                'Optimizer', ['adam', 'sgd', 'rmsprop'], index=0)
            lossFunction = modelParametersExpander.radio(
                'Loss Function', ['mse', 'mae', 'mape'], index=0)

            if modelTypeSelectionExpander.button('Calculate Selected Model'):
                pass
                # generator = TimeseriesGenerator(
                #     train, train, length=n_input, batch_size=batchSize)

                # LSTMmodel = LSTMModel(n_input, n_features, neurons,
                #                       epochs, activationFunction, optimizer, lossFunction)
                # st.write(LSTMmodel.summary(print_fn=lambda x: st.text(x)))

                # st.write(generator)
                # LSTMmodel.fit(generator, epochs=epochs)

        elif modelSelection == "HAR Net":
            pass
######################################
with st.sidebar.container():
    st.sidebar.subheader("University of Vienna | Research Project")
    st.sidebar.write(
        "###### App Authors: Roderick Perez & Le Thi (Janie) Thuy Trang")
    st.sidebar.write("###### Faculty Advisor: Xandro Bayer")
    st.sidebar.write("###### Updated: 30/9/2022")
