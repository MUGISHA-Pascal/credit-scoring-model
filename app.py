import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_excel("./creditdataset2/credit-data-training.xlsx")
cleaned_data = data.dropna()
x=cleaned_data.drop(columns="Credit-Application-Result")
y=cleaned_data["Credit-Application-Result"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test);

accuracy = accuracy_score(y_test,y_pred)
print(f"accuracy is {accuracy}")

plt.figure(figsize=(12,8))
tree.plot_tree(model,filled=True,feature_names=x.columns,class_names=["Creditworthy","Non-Creditworthy"])