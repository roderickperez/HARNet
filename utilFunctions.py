def to_excel(df):
    return df.to_csv().encode('utf-8')