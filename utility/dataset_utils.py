import pandas as pd


# it takes only the slice of the dataframe you are interested in
def cut_dataset_by_range(PATH, crypto_symbol, start_date, end_date, features_to_use=None):
    if features_to_use != None:
        df = pd.read_csv(PATH + crypto_symbol + ".csv", delimiter=',', header=0, usecols=features_to_use)
    else:
        df = pd.read_csv(PATH + crypto_symbol + ".csv", delimiter=',', header=0)
    df = df.set_index("Date")
    df1 = df.loc[start_date:end_date, :]
    df1 = df1.reset_index()
    return df1


""" for row in df.itertuples():
            #print(row.Open)
            if (math.isnan(row.Open)):
                fin_date=row.Index
                #df=df.drop(df.index[init_date:row.Index])
                df=df.query('index < @init_date or index > @fin_date')
                init_date=row.Index
    df.to_csv('../dataset/reviewed/'+file,",")   """
