import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
dados = pd.read_csv("Painted data.csv", delimiter=',')
dados.head()
#Visualizando documentos
dados.plot.scatter(x='x',y='y')
# é necessário adicionar uma constante a matriz X
xMaisConstantes = sm.add_constant(dados['x'])

# OLS vem de Ordinary Least Squares e o método fit irá treinar o modelo
results = sm.OLS(dados['y'], xMaisConstantes).fit()

# mostrando as estatísticas do modelo
results.summary()
#Mostrando graficamente
sns.regplot(x = 'x' , y = 'y' , data = dados, color ="b",  marker="p" ,  ci=80, x_bins=10)
plt.show()
