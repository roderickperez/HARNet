import numpy as np

def volatility(df, variable, variableName):
    df[variableName] = df[variable].pct_change().apply(
        lambda x: np.log(1+x)).dropna()
    return df


def historicalVolatility(df, variable, volatilityRollingWindow, variableName):
    df[variableName] = df[variable].pct_change().rolling(
        volatilityRollingWindow).std().dropna()
    return df


def realizedVolatility(df, variable, volatilityRollingWindow, variableName):
    realizedVolatility = np.log(
        df[variable]/df[variable].shift(1))
    # window/time tells us how many days out vol you want. ~21 = 1 month out vol (~21 trading days in a month)
    # we do this so we can match up with the vix which is the 30 day out (~21 trading day) calculated vol
    df[variableName] = realizedVolatility.rolling(
        window=volatilityRollingWindow).std()*np.sqrt(252)
    return df