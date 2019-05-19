# importando as bibiliotecas

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#Lendo documentos
dados = pd.read_csv("Dados retirados do orange.csv", delimiter=',')
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
fig = sns.regplot(x = 'x' , y = 'y' , data = dados, color ="r",  marker="o" ,  ci=100)
plt.show(fig)
#, x_bins=10
