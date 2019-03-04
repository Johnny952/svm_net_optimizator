from sklearn.preprocessing import StandardScaler
import SVMOpt
import SVMNet
from SVMNet import extract_classes
from sklearn.model_selection import train_test_split
import csv
import numpy as np

iris=[]
with open('iris.csv') as csvarchivo:
    entrada = csv.reader(csvarchivo)
    for reg in entrada:
        iris.append(reg)
iris = np.array(iris)
iris_data = iris[:,:-1].astype(np.float)
iris_labels = iris[:,-1]

X_train, X_val, y_train, y_val = train_test_split(iris_data, iris_labels, test_size=0.25, stratify = iris_labels, random_state=42)


scaler2 = StandardScaler()
scaler2.fit(X_train)
X_train = scaler2.transform(X_train)
X_val = scaler2.transform(X_val)

svmop = SVMOpt.SVMOpt()

svmop.set_encode(iris_labels)
y2_train = svmop.encode_labels(y_train)
y2_val = svmop.encode_labels(y_val)

svmop.set_selection("rank")
best, a7, a8 = svmop.run(X_train, y2_train, X_val, y2_val, 40, 0.8, 0.4, 40, exponential = True)
