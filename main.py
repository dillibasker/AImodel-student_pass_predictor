import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt 

df=pd.read_csv('student_data.csv')
x=df.drop('Pass',axis=1)