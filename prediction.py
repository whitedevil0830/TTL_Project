import joblib as jb, category_encoders as ce, pandas as pd
df = pd.read_csv("C:/Users/KIIT/TTL_PrOjEcT/Traffic_Accident_new.csv")
def pred(data):
    model = jb.load("LR.sav")
    model1 = jb.load("KNN.sav")
    model2 = jb.load("NB.sav")
    enc = ce.OrdinalEncoder(cols=df.columns)
    enc.fit_transform(df)
    new_data_transformed = enc.transform(data)
    return model.predict(new_data_transformed), model1.predict(new_data_transformed), model2.predict(new_data_transformed)