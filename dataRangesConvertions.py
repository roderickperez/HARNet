import pandas as pd
from datetime import date, datetime
import pytz
utc = pytz.UTC


def convertDatesFormat(dateStart, dateEnd):
    dateStart = pd.to_datetime(dateStart)
    dateEnd = pd.to_datetime(dateEnd)

    my_time = datetime.min.time()

    dateStart = datetime.combine(dateStart, my_time)
    dateEnd = datetime.combine(dateEnd, my_time)

    dateStart = dateStart.replace(tzinfo=utc)
    dateEnd = dateEnd.replace(tzinfo=utc)

    return dateStart, dateEnd


def splitData(df, variable, train_size):
    df_train = df[[variable]].iloc[:int(len(df[variable])*train_size)]
    df_test = df[[variable]].iloc[int(len(df[variable])*train_size):]
    return df_train, df_test
