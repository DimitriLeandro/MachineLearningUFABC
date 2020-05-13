import numpy as np
from random import randint

class Adaline:

    pesos         = []
    qtdIteracoes  = 0 
    evolucaoMSE   = []
    evolucaoPesos = []
    motivoParada  = ""

    def __adicionarDimensaoBias(self, dados):    
        novosDados = []
        for dadoAtual in dados:
            novoDadoAtual = np.append(1, dadoAtual)
            novosDados.append(novoDadoAtual)
        return np.array(novosDados)

    def __iniciarPesos(self, tamanho):
        self.pesos = np.zeros(tamanho)
        #print("Iniciando pesos:", self.pesos)

    def __calcularSaida(self, dado):
        return np.matmul(self.pesos.T, dado)

    def __calcularMSE(self, dadosComBias, target):
        vetorErros = []
        for dadoAtual, targetAtual in zip(dadosComBias, target):
            erroAtual = targetAtual - self.__calcularSaida(dadoAtual)
            vetorErros.append(erroAtual)
        vetorErros = np.array(vetorErros)
        return np.matmul(vetorErros.T, vetorErros)/len(target)

    def __atualizarPesosUnicaEpoca(self, xTreinamento, yTreinamento, taxaAprendizagem):
        #print("Taxa:", taxaAprendizagem)
        #print("Iniciando uma época. Pesos antes:", self.pesos)
        # PASSANDO POR CADA DADO
        for dadoAtual, targetAtual in zip(xTreinamento, yTreinamento):

            # FAZENDO A PREDICAO DO DADO ATUAL
            saidaCalculadaAtual = self.__calcularSaida(dadoAtual)
            
            # CALCULANDO O ERRO
            erroAtual = targetAtual - saidaCalculadaAtual

            # ATUALIZANDO O VETOR DE PESOS
            self.pesos = self.pesos + taxaAprendizagem * erroAtual * dadoAtual
        #print("Finalizando uma época. Pesos depois:", self.pesos)

    def __iterarEpocasTreinamento(self, xTreinamento, yTreinamento, taxaAprendizagem=1e-6, qtdMaxEpocas=50000, percentualSemMelhora=0.35, armazenarEvolucao=True):
        # PRA GUARDAR A EVOLUCAO DAS PARADAS
        arrayMSEs           = [self.__calcularMSE(xTreinamento, yTreinamento)]
        matrizEvolucaoPesos = [self.pesos]

        # VOU RETORNAR SO A MELHOR ITERACAO
        melhorMSE     = self.__calcularMSE(xTreinamento, yTreinamento)
        melhoresPesos = self.pesos
        
        # COMECANDO
        epocas = 0
        qtdEpocasSemMelhoria = 0
        while epocas < qtdMaxEpocas:
            epocas += 1

            # COMECO VERIFICANDO O MSE DESSES PESOS
            MSEAtual = self.__calcularMSE(xTreinamento, yTreinamento)

            # ARMAZENO A EVOLUCAO DA EPOCA
            if armazenarEvolucao == True:
                arrayMSEs.append(MSEAtual)
                matrizEvolucaoPesos.append(self.pesos)

            # VERIFICO SE FOI A MELHOR ITERACAO ATE AGORA
            if MSEAtual < melhorMSE:
                melhorMSE = MSEAtual
                melhoresPesos = self.pesos
                qtdEpocasSemMelhoria = 0
            else:
                qtdEpocasSemMelhoria += 1
                if qtdEpocasSemMelhoria/qtdMaxEpocas >= percentualSemMelhora:
                    motivoParada = "Sem melhora no MSE há " + str(percentualSemMelhora*100) + "% do total de épocas."
                    break

            # SE TIVER CONVERGIDO EU PARO
            if MSEAtual == 0:
                motivoParada = "O MSE de treinamento chegou a 0."
                break

            # SE NAO TIVER CONVERGIDO EU CONTINUO O ALGORITMO E AJUSTO OS PESOS DESSA EPOCA
            self.__atualizarPesosUnicaEpoca(xTreinamento, yTreinamento, taxaAprendizagem)

        if epocas >= qtdMaxEpocas:
            motivoParada = "A quantidade máxima de épocas foi atingida."

        # TRANSFORMANDO OS ARRAYS EM NUMPY
        # VOU TRANSPOR A MATRIZ DE EVOLUCAO DOS PESOS, ASSIM, CADA LINHA E UM PESO E CADA COLUNA UMA EPOCA
        epocas              = np.array(epocas)
        arrayMSEs           = np.array(arrayMSEs)
        matrizEvolucaoPesos = np.array(matrizEvolucaoPesos).T

        # AGORA QUE TERMINOU, OS PESOS DA CLASSE SERAO OS MELHORES PESOS DESSA FUNCAO
        self.pesos = melhoresPesos

        return epocas, arrayMSEs, matrizEvolucaoPesos, motivoParada

    def treinar(self, xTreinamento, yTreinamento, taxaAprendizagem=1e-6, qtdMaxEpocas=30000, percentualSemMelhora=0.35, armazenarEvolucao=True):
        # COLOCANDO A DIMENSAO DO BIAS NO FINAL DE CADA DADO
        xTreinamento = self.__adicionarDimensaoBias(xTreinamento)

        # INICIANDO O VETOR DE PESOS
        self.__iniciarPesos(len(xTreinamento[0]))

        # TREINANDO DE FATO
        epocas, arrayMSEs, matrizEvolucaoPesos, motivoParada = self.__iterarEpocasTreinamento(xTreinamento, yTreinamento, taxaAprendizagem, qtdMaxEpocas, percentualSemMelhora, armazenarEvolucao)
        
        # COLOCANDO NA CLASSE
        self.qtdIteracoes  = epocas 
        self.evolucaoMSE   = arrayMSEs
        self.evolucaoPesos = matrizEvolucaoPesos
        self.motivoParada  = motivoParada

    def predizer(self, xTeste):
        # ADICIONANDO UMA DIMENSAO PARA O BIAS
        xTeste = self.__adicionarDimensaoBias(xTeste)
        
        # PREDIZENDO CADA DADO
        yPred = []        
        for dadoAtual in xTeste:
            yPred.append(self.__calcularSaida(dadoAtual))
            
        return yPred

    def obterEixosParaPlotarRetaAdaline(self, dadosOriginais):
        # OBVIAMENTE SO FUNCIONA PARA CASOS DE UMA UNICA FEATURE
        eixoXRetaAdaline = [dadosOriginais[0], dadosOriginais[-1]]
        eixoYRetaAdaline = [self.pesos[0]+dadosOriginais[0]*self.pesos[1], self.pesos[0]+dadosOriginais[-1]*self.pesos[1]]
        return eixoXRetaAdaline, eixoYRetaAdaline

