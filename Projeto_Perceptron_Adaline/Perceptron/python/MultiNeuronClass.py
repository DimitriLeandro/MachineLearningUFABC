import numpy as np
import sys
from joblib import Parallel, delayed
sys.path.append("/home/dimi/Programming/MachineLearningUFABC/Projeto_Perceptron_Adaline/Perceptron/python/")
from NeuronClass import Neuron

# Usando a implementação do neurônio binário, a ideia é implementar o algoritmo do Perceptron para datasets com mais de uma classe.
class MultiNeuron:

	arrayNeuronios             = []
	taxaAprendizagem           = 0
	qtdMaxEpocas               = 0
	armazenarEvolucaoNeuronios = True
	percentualSemMelhora       = 0 # o neuronio para de treinar caso a acurácia de treinamento não tiver melhorado nas últimas J iterações, sendo J equivalente a um percentual do limite de épocas determinado

	def __init__(self, taxaAprendizagem=1e-3, qtdMaxEpocas=1000, percentualSemMelhora=0.5, armazenarEvolucaoNeuronios=True):
		self.taxaAprendizagem           = taxaAprendizagem
		self.qtdMaxEpocas               = qtdMaxEpocas
		self.armazenarEvolucaoNeuronios = armazenarEvolucaoNeuronios
		self.percentualSemMelhora       = percentualSemMelhora

	def __transformarTarget(self, target, classeDesejada):
    	
    	# Função para criar um target binário
		# Um neurônio vai ser criado para cada classe do dataset. Cada neurônio vai ser treinado como "Verifique se é da classe atual ou se é de qualquer outra classe". Dado uma classe, essa função vai transformar o vetor target em 0s e 1s, sendo 1 pertencendo à classe desejada e 0 qualquer outra.

	    novoTarget = []
	    
	    for rotuloAtual in target:
	        if rotuloAtual == classeDesejada:
	            novoTarget.append(1)
	        else:
	            novoTarget.append(0)
	            
	    return np.array(novoTarget)

	def __criarETreinarUnicoNeuronio(self, xTreinamento, yTreinamento, classeDesejada):
    	# Função para criar e treinar um único neurônio dada uma classe

	    # CRIANDO UM DATASET DO TIPO "CLASSE ATUAL (1) X QQR OUTRA CLASSE (0)"
	    yBinarioClasseAtual = self.__transformarTarget(yTreinamento, classeDesejada)
	    
	    # INSTANCIANDO E TREINANDO O NEURONIO ATUAL
	    objNeuron = Neuron()
	    objNeuron.treinar(xTreinamento, yBinarioClasseAtual, self.taxaAprendizagem, self.qtdMaxEpocas, self.percentualSemMelhora, self.armazenarEvolucaoNeuronios)
	    
	    return objNeuron

	def treinar(self, xTreinamento, yTreinamento):
    	# Função para treinar o algoritmo do perceptron multiclasse
	    # VERIFICANDO QUAIS SAO AS CLASSES EXISTENTES NO DATASET
	    arrayClasses = np.unique(yTreinamento)

	    # CRIANDO E TREINANDO UM NEURONIO PARA CADA CLASSE DO DATASET
	    self.arrayNeuronios = Parallel(n_jobs=-1)(delayed(self.__criarETreinarUnicoNeuronio)(xTreinamento, yTreinamento, classeAtual) for classeAtual in arrayClasses)

	def __maiorEntradaLiquida(self, dado):
    	# Função para devolver o index do neuronio com a maior entrada líquida para um único dado de entrada
	    # ADICIONANDO BIAS AO FINAL DO DADO
	    dadoComBias = np.append(dado, 1)
	    
	    # VOU VERIFICAR A ENTRADA LIQUIDA DE CADA NEURONIO, QUEM TIVER A MAIOR GANHA
	    arrayEntradasLiquidas = []
	    for i, neuronioAtual in enumerate(self.arrayNeuronios):
	        arrayEntradasLiquidas.append(neuronioAtual.calcularEntradaLiquida(dadoComBias))
	        
	    # RETORNANDO O INDEX DO NEURONIO COM A MAIOR ENTRADA LIQUIDA COMO A CLASSE PREDITA
	    return arrayEntradasLiquidas.index(max(arrayEntradasLiquidas))

	def predizer(self, xTeste):    
	    arrayPredicoes = Parallel(n_jobs=-1)(delayed(self.__maiorEntradaLiquida)(dadoAtual) for dadoAtual in xTeste)
	    return arrayPredicoes