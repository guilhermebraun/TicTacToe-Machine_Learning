import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split

dados = pd.read_csv("Dados-T1-IA.csv", sep=";")

print(dados.shape)
print(dados.columns)
print(dados.head())
X = dados.drop(columns=("Resultado"))
print(X.head())
Y = dados["Resultado"].values
print(Y)
xTreino, xTemporario, yTreino, yTemporario = train_test_split()

