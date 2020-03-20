import numpy as np
import math

class Metricas:

	acuracia          = 0
	precisoes         = [] 
	revocacoes        = []
	mediaPrecisoes    = 0
	mediaRevocacoes   = 0
	desvPadPrecisoes  = 0
	desvPadRevocacoes = 0

	def __init__(self, matrizConfusao):
		self.acuracia          = self.calcularAcuracia(matrizConfusao)
		self.precisoes         = self.calcularPrecisoes(matrizConfusao)
		self.revocacoes        = self.calcularRevocacoes(matrizConfusao)
		self.mediaPrecisoes    = np.mean(self.precisoes)
		self.mediaRevocacoes   = np.mean(self.revocacoes)
		self.desvPadPrecisoes  = np.std(self.precisoes)
		self.desvPadRevocacoes = np.std(self.revocacoes)

	def calcularAcuracia(self, matrizConfusao):
		# DEFINIDA COMO (TP + TN)/(TP + FP + TN + FN) -> VALORES DA DIAGONAL PRINCIPAL SOMADOS DIVIDIDO PELO VALOR DE TODAS AS CELULAS SOMADAS
		somaDiagonalPrincipal = np.sum(np.diag(matrizConfusao))
		somaTudo              = np.sum(matrizConfusao)

		return somaDiagonalPrincipal / somaTudo

	def calcularRevocacoes(self, matrizConfusao):
		
		arrayRevocacoes = []

		for i, linha in enumerate(matrizConfusao):
			
			somaLinha = np.sum(linha)

			if somaLinha != 0: 
				revocacaoAtual = linha[i]/somaLinha
			else:
				revocacaoAtual = 0

			arrayRevocacoes.append(revocacaoAtual)

		return arrayRevocacoes

	def calcularPrecisoes(self, matrizConfusao):
		
		transposta = np.array(matrizConfusao).T

		arrayPrecisoes = []

		for i, linha in enumerate(transposta):
			
			somaLinha = np.sum(linha)

			if somaLinha != 0: 
				precisaoAtual = linha[i]/somaLinha
			else:
				precisaoAtual = 0

			arrayPrecisoes.append(precisaoAtual)

		return arrayPrecisoes
