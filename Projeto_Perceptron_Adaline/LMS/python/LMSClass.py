import numpy as np

class LMS:

	pesos = []
	MSE   = 0
	RMSE  = 0

	def __init__(self, dados, target):
		dadosComBias = self.__adicionarDimensaoBias(dados)
		self.__calcularPesos(dadosComBias, target)
		self.__calcularMSE(dadosComBias, target)
		self.RMSE = self.MSE**(1/2)

	def __adicionarDimensaoBias(self, vetorDados):    
		novoX = []
		for dadoAtual in vetorDados:
			novoDadoAtual = np.append(1, dadoAtual)
			novoX.append(novoDadoAtual)
		return np.array(novoX)

	def __calcularPesos(self, dadosComBias, target):
		self.pesos = np.matmul(np.matmul(np.linalg.inv(np.matmul(dadosComBias.T, dadosComBias)), dadosComBias.T), target.T)

	def __calcularMSE(self, dadosComBias, target):
		vetorErros = []	    
		for dadoAtual, targetAtual in zip(dadosComBias, target):
			erroAtual = targetAtual - np.matmul(self.pesos.T, dadoAtual)
			vetorErros.append(erroAtual)	        
		vetorErros = np.array(vetorErros)	    
		self.MSE = np.matmul(vetorErros.T, vetorErros)/len(target)

	def predizer(self, xTeste):
		# ADICIONANDO UMA DIMENSAO PARA O BIAS
		xTeste = self.__adicionarDimensaoBias(xTeste)
		
		# PREDIZENDO CADA DADO
		yPred = []        
		for dadoAtual in xTeste:
			yPred.append(np.matmul(self.pesos.T, dadoAtual))
			
		return yPred

	def obterEixosParaPlotarRetaLMS(self, dadosOriginais):
		# OBVIAMENTE SO FUNCIONA PARA CASOS DE UMA UNICA FEATURE
		eixoXRetaLMS = [dadosOriginais[0], dadosOriginais[-1]]
		eixoYRetaLMS = [self.pesos[0]+dadosOriginais[0]*self.pesos[1], self.pesos[0]+dadosOriginais[-1]*self.pesos[1]]
		return eixoXRetaLMS, eixoYRetaLMS