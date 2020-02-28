from scipy.stats import mode
import math
import numpy as np

class KNN:

	k      = 0
	xTrain = []
	yTrain = []

	def __init__(self, k):
		self.k = k		

	def treinar(self, xTrain, yTrain):
		# Função para realizar o treinamento
		# No KNN o treinamento e basicamente um armazenamento dos dados de treinamento, os calculos ficam pra cada dado de teste individualmente

		self.xTrain = xTrain
		self.yTrain = yTrain

	def predizer(self, xTest):
		# Função para predizer todo um conjunto de dados de teste
		# Essa função fará todo o trabalho, recebendo um dataset de teste para retornar um array com a classificação de cada dado de teste
	
		predicoes = []
		
		# PARA CADA DADO
		for dadoTesteAtual in xTest:
			
			# EU CALCULO AS DISTANCIAS ATE CADA DADO DE TREINAMENTO
			matrizIndexesEDistancias = self.obterIndexesEDistancias(dadoTesteAtual)
			
			# ORDENO PELAS DISTANCIAS MAIS PROXIMAS
			matrizIndexesEDistancias = self.ordenarPelaDistancia(matrizIndexesEDistancias)
			
			# OBTENHO OS INDICES DOS K PRIMEIROS MAIS PROXIMOS
			indexesKMaisProximos = self.obterKPrimeirosIdexes(matrizIndexesEDistancias)
			
			# FACO A CLASSIFICACAO DO DADO ATUAL
			classificacaoDadoAtual = self.classificarUnicoDado(indexesKMaisProximos)
			
			# COLOCO A CLASSIFICACAO NO ARRAY DE PREDICOES
			predicoes.append(classificacaoDadoAtual)
			
		return predicoes

	def calcularDistancia(self, pontoA, pontoB):
	
		# Função para calcular a distância entre dois pontos do dataset

		distancia = 0
		
		# PASSO POR CADA DIMENSAO SUBTRAINDO AS COORDENADAS E ELEVANDO AO QUADRADO
		for i in range(len(pontoA)):
			
			# SOMO NA DISTANCIA TOTAL
			distancia += (pontoA[i] - pontoB[i]) ** 2
			
		# TIRO A RAIZ
		return distancia ** (1/2)

	def obterIndexesEDistancias(self, novoDado):

		# Função para gerar uma matriz de distâncias e indexes
		# Dado um ponto que se queira classificar, essa função deverá retornar uma matriz onde cada linha representa um dado de treinamento. A primeira coluna mostrará o index desse elemento e a segunda, a distância em relação ao ponto em que se queira classificar.
	
		# INICIANDO A MATRIZ
		matrizIndexesEDistancias = []
		
		# PARA CADA ELEMENTO
		for i, dadoTreinoAtual in enumerate(self.xTrain):
			matrizIndexesEDistancias.append([i, self.calcularDistancia(novoDado, dadoTreinoAtual)])
			
		return matrizIndexesEDistancias

	def ordenarPelaDistancia(self, matrizIndexesEDistancias):

		# Função para ordenar a matriz de indexes e distâncias de acordo com as menores distâncias
		# Essa função deverá receber a matriz de indexes e distâncias e ordenar as linhas de acordo com as menores distâncias.
	
		linhaAuxiliar = []
		
		for i in range(0, len(matrizIndexesEDistancias)):
			for j in  range(i+1, len(matrizIndexesEDistancias)):
				
				if matrizIndexesEDistancias[j][1] < matrizIndexesEDistancias[i][1]:
					linhaAuxiliar = matrizIndexesEDistancias[i]
					matrizIndexesEDistancias[i] = matrizIndexesEDistancias[j]
					matrizIndexesEDistancias[j] = linhaAuxiliar
					
		return matrizIndexesEDistancias

	def obterKPrimeirosIdexes(self, matrizIndexesEDistancias):
	
		# Função para retornar os k primeiros indexes da matriz de distâncias depois que ela já estiver ordenada

		indexesKMaisProximos = np.array(matrizIndexesEDistancias)[0:self.k, 0]
		
		return list(indexesKMaisProximos)

	def classificarUnicoDado(self, indexesKMaisProximos):

		# Função para classificar um único dado
		# Essa função deverá receber os indexes dos dados de treino mais próximos e o y de treinamento. Vou pegar o valor do y de cada k dado mais próximo de treinamento e usar a moda para definir a classe do novo dado.
		# A função que calcula a moda já vai garantir que apenas um valor seja retornado, não preciso me preocupar com empates.
	
		kClassesMaisProximas = []
		
		for i in indexesKMaisProximos:
			kClassesMaisProximas.append(self.yTrain[int(i)])
		
		moda = mode(kClassesMaisProximas)[0][0]
		
		return moda
