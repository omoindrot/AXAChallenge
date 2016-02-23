import pandas as pd
import numpy as np

# Sauvestre pour toi

# Read CSV file

X = pd.read_csv('data/train_2011_2012.csv', sep=';')

res = pd.read_csv('submission.txt', sep='\t')


# Cleanup of data
# Ici retire les colonnes qui servent a rien (cf. cleanup.py)
# Ici remplace les colonnes avec des string par plusieurs colonnes de 0/1
X.drop('ASS_COMENT', 1, inplace=True)


# Separation training set / validation set



# Creation d'un modele
# ex: kNN, logistic regression...

model = None


# Evaluation des resultats sur le validation set

MSE_val = 0


