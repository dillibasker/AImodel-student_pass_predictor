import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt 

df=pd.read_csv('student_data.csv')
x=df.drop('Pass',axis=1)
y=df['Pass']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)

accuracy=accuracy_score(y_test,predictions)
cm = confusion_matrix(y_test, predictions)

print("✅ Model Accuracy:", round(accuracy * 100, 2), "%")
# 7. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Pass/Fail")
plt.show()

