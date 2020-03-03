import numpy as np

class Metricas:

	acuracia        = 0
	precisoes  = [] 
	revocacoes = []

	def __init__(self, matrizConfusao):
		self.acuracia        = self.calcularAcuracia(matrizConfusao)
		self.precisoes  = self.calcularPrecisoes(matrizConfusao)
		self.revocacoes = self.calcularRevocacoes(matrizConfusao)

	def calcularAcuracia(self, matrizConfusao):
		# DEFINIDA COMO (TP + TN)/(TP + FP + TN + FN) -> VALORES DA DIAGONAL PRINCIPAL SOMADOS DIVIDIDO PELO VALOR DE TODAS AS CELULAS SOMADAS
		somaDiagonalPrincipal = 0
		somaTudo              = 0

		for i in range(len(matrizConfusao)):
			for j in range(len(matrizConfusao[i])):
				
				somaTudo += matrizConfusao[i][j]
				
				if i == j:
					somaDiagonalPrincipal += matrizConfusao[i][j]

		return somaDiagonalPrincipal / somaTudo

	def calcularPrecisoes(self, matrizConfusao):
		
		arrayPrecisoes = []

		for i, linha in enumerate(matrizConfusao):
			precisaoAtual = linha[i]/sum(linha)
			arrayPrecisoes.append(precisaoAtual)

		return arrayPrecisoes

	def calcularRevocacoes(self, matrizConfusao):
		
		transposta = np.array(matrizConfusao).T

		arrayRevocacoes = []

		for i, linha in enumerate(transposta):
			revocacaoAtual = linha[i]/sum(linha)
			arrayRevocacoes.append(revocacaoAtual)

		return arrayRevocacoes
