#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:53:58 2018

@author: quark
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Data Preprocessing
"""
Created on Sun Apr  8 13:25:54 2018

@author: quark
"""
# importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importer le dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# diviser le dataset entre training et test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/3, random_state=0)

# construction du modèle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# faire de nouvelles prédictions
y_pred = regressor.predict(X_test)
regressor.predict(15)

# visualiser les résultats
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title( 'Salaire VS Expérience')
plt.xlabel('Expérience')
plt.ylabel('Salaire')
plt.show()