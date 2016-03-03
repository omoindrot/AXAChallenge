import pandas as pd
import numpy as np
from datetime import date, datetime


# Get the data
meteo_2011 = pd.read_csv('data/meteo/meteo_2011.csv', header=None, index_col=[0], names=['DPT', 'CITY', 'TEMP', 'WIND', 'WIND_DIR', 'PRECIP', 'PRESSURE'])
meteo_2012 = pd.read_csv('data/meteo/meteo_2012.csv', header=None, index_col=[0], names=['DPT', 'CITY', 'TEMP', 'WIND', 'WIND_DIR', 'PRECIP', 'PRESSURE'])
meteo = pd.concat([meteo_2011, meteo_2012])
meteo = meteo.drop(['CITY', 'DPT', 'WIND', 'WIND_DIR'], axis=1)

meteo = meteo.groupby(meteo.index).mean()

index_days = pd.date_range('2011-01-01', '2012-12-31', freq='D')
meteo2 = pd.DataFrame(np.zeros((731, 3)), index=index_days, columns=['TEMP', 'PRESSURE', 'PRECIP'])

index_hours = pd.date_range('2011-01-01 00:00', '2012-12-31 23:00', freq='H')
meteo_cleaned = pd.DataFrame(index=index_hours, columns=['TEMP', 'PRESSURE', 'PRECIP'])
for hour in index_hours:
    if hour.isoformat(' ')[:-3] in meteo.index:
        meteo_cleaned.loc[hour] = meteo.loc[hour.isoformat(' ')[:-3]]
meteo_cleaned['HOUR'] = index_hours
meteo_cleaned['DAY'] = meteo_cleaned['HOUR'].apply(lambda h: h.date())
meteo_cleaned = meteo_cleaned.drop('HOUR', axis=1)

# We create a better dataframe meteo2 indexed by day
meteo_grouped = meteo_cleaned.groupby(by='DAY')
for day, x in meteo_grouped:
    x.drop('DAY', inplace=True, axis=1)
    meteo2.loc[day] = x.mean().copy()

# We fill in the gap values
meteo2['TEMP'].loc['2011-08-11':'2011-08-16'] = 17.
meteo2['PRESSURE'].loc['2011-08-11':'2011-08-16'] = 1013.
meteo2['PRECIP'].loc['2011-08-11':'2011-08-16'] = 0.
meteo2['TEMP'].loc['2012-02-29'] = 8.
meteo2['PRESSURE'].loc['2012-02-29'] = 1024.
meteo2['PRECIP'].loc['2012-02-29'] = 0.

# Finally we reshape the values to have 0 mean and variance 1
meteo2 = (meteo2 - meteo2.mean())/meteo2.std()

pd.to_pickle(meteo2, 'tmp/meteo')
