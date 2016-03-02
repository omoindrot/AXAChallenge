import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Read data
#X = pd.read_csv('data/train_2011_2012.csv', sep=';')


X_cleaned = pd.read_pickle('tmp/X_cleaned')

assignment = 'Tech. Axa'
X_ass = X_cleaned[X_cleaned['ASS_ASSIGNMENT'] == assignment]
X_ass = pd.DataFrame(X_ass.iloc[:, 2:].values, index=X_ass.iloc[:, 1])
X_sum = X_ass.sum(axis=1)
days = pd.date_range('2011-01-01', end='2012-12-31', freq='D')

lundi = pd.date_range('2011-01-03', periods=52*2, freq='7D')
mardi = pd.date_range('2011-01-04', periods=52*2, freq='7D')
mercredi = pd.date_range('2011-01-05', periods=52*2, freq='7D')
jeudi = pd.date_range('2011-01-06', periods=52*2, freq='7D')
vendredi = pd.date_range('2011-01-07', periods=52*2, freq='7D')
samedi = pd.date_range('2011-01-01', periods=52*2, freq='7D')
dimanche = pd.date_range('2011-01-02', periods=52*2, freq='7D')

X_ass.loc[mardi, 23].plot()
X_sum.loc[days].plot()
plt.show()

meteo_2011 = pd.read_csv('data/meteo/meteo_2011.csv', header=None, index_col=[0], names=['DPT', 'CITY', 'TEMP', 'WIND', 'WIND_DIR', 'PRECIP', 'PRESSURE'])
meteo_2012 = pd.read_csv('data/meteo/meteo_2012.csv', header=None, index_col=[0], names=['DPT', 'CITY', 'TEMP', 'WIND', 'WIND_DIR', 'PRECIP', 'PRESSURE'])
meteo = pd.concat([meteo_2011, meteo_2012])
meteo = meteo.drop(['CITY', 'DPT', 'WIND', 'WIND_DIR', 'PRESSURE'], axis=1)

meteo = meteo.groupby(meteo.index).mean()


# Recuperer les donnees pour un jour pour une entreprise

# Plotter le nombre d'appels sur une journee / une semaine...

