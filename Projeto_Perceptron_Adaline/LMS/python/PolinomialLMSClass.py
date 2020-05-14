import numpy as np

class PolinomialLMS:

	pesos = []
	MSE   = 0
	RMSE  = 0
	grau  = 0

	def __init__(self, dados, target, grau=3):
		self.grau = grau
		dadosComDimensoes = self.__adicionarDimensoes(dados)
		self.__calcularPesos(dadosComDimensoes, target)
		self.__calcularMSE(dadosComDimensoes, target)
		self.RMSE = self.MSE**(1/2)

	def __adicionarDimensoes(self, dados):  
		novosDados = []
		for dadoAtual in dados:
			novoDadoAtual = []
			for i in range(0, self.grau+1):
				novoDadoAtual.append(dadoAtual**i)
			novosDados.append(novoDadoAtual)
		return np.array(novosDados)

	def __calcularPesos(self, dadosComDimensoes, target):
		self.pesos = np.matmul(np.matmul(np.linalg.inv(np.matmul(dadosComDimensoes.T, dadosComDimensoes)), dadosComDimensoes.T), target.T)

	def __calcularMSE(self, dadosComDimensoes, target):
		vetorErros = []	    
		for dadoAtual, targetAtual in zip(dadosComDimensoes, target):
			erroAtual = targetAtual - np.matmul(self.pesos.T, dadoAtual)
			vetorErros.append(erroAtual)	        
		vetorErros = np.array(vetorErros)	    
		self.MSE = np.matmul(vetorErros.T, vetorErros)/len(target)

	def predizer(self, xTeste):
		# ADICIONANDO UMA DIMENSAO PARA O BIAS
		xTeste = self.__adicionarDimensoes(xTeste)
		
		# PREDIZENDO CADA DADO
		yPred = []        
		for dadoAtual in xTeste:
			yPred.append(np.matmul(self.pesos.T, dadoAtual))
			
		return np.array(yPred)

	def obterEixosParaPlotarRetaLMS(self, dadosOriginais, qtdPontos=1000):
		# OBVIAMENTE SO FUNCIONA PARA CASOS DE UMA UNICA FEATURE
		eixoXRetaLMS = np.linspace(dadosOriginais[0], dadosOriginais[-1], qtdPontos)
		eixoYRetaLMS = self.predizer(eixoXRetaLMS)
		return eixoXRetaLMS, eixoYRetaLMS