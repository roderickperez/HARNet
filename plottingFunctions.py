from plotly import graph_objects as go
import streamlit as st


def plotDataFrame(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df))
    fig.layout.update(
        xaxis_rangeslider_visible=True)
    fig.update_layout(
        autosize=False,
        width=1400,
        height=400,
        plot_bgcolor="black",
        margin=dict(
            l=50,
            r=50,
            b=0,
            t=0,
            pad=2
        ))

    st.plotly_chart(fig)


def plotForecast(train, test, testPred, futurePred):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train.index, y=train['Open'], name='Train'))
    fig.add_trace(
        go.Scatter(x=test.index, y=test['Open'], name='Test'))
    # fig.add_trace(
    #     go.Scatter(x=testPred.index, y=testPred[0], name='Test Prediction'))
    fig.add_trace(
        go.Scatter(x=futurePred.index, y=futurePred['FuturePrediction'], name='Forecast'))
    fig.layout.update(
        xaxis_rangeslider_visible=True)
    fig.update_layout(
        autosize=False,
        width=1400,
        height=450,
        plot_bgcolor="black",
        margin=dict(
            l=50,
            r=50,
            b=0,
            t=0,
            pad=2
        ))
    st.plotly_chart(fig)


def histogramPlot(df, variable):
    fig = go.Figure(data=[go.Histogram(x=df[variable])])
    fig.update_layout(
        autosize=False,
        width=1400,
        height=400,
        plot_bgcolor="black",
        margin=dict(
            l=50,
            r=50,
            b=0,
            t=0,
            pad=2
        ))
    st.plotly_chart(fig)


def histogramNormalizedPlot(df, variable):
    fig = go.Figure(
        data=[go.Histogram(x=df[variable], histnorm='probability')])
    fig.update_layout(
        autosize=False,
        width=1400,
        height=400,
        plot_bgcolor="black",
        margin=dict(
            l=50,
            r=50,
            b=0,
            t=0,
            pad=2
        ))
    st.plotly_chart(fig)
