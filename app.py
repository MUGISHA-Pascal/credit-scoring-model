import pandas as pd

data = pd.read_excel("./creditdataset2/credit-data-training.xlsx")
cleaned_data = data.dropna()
print(data.head())