import numpy as np
from random import randint

# A classe abaixo serve para um único neurônio, isto é, um classificador binário, uma classe versus todas as outras. Depois de implementar um único neurônio, eu vou criar vários para fazer predições com datasets com mais de uma classe.
class Neuron:

    pesos                = []
    qtdIteracoes         = 0 # qtd de iteracoes realizadas ate que o treinamento fosse finalizado
    evolucaoAcuracias    = [] # array com as acuracias a cada epoca de treinamento
    evolucaoPesos        = [] # matriz em que cada linha e um peso e cada coluna e uma epoca
    motivoParada         = "" # string que vai informar qual foi o criterio de parada atingido

    def __adicionarDimensaoBias(self, x):
        x = np.append(x, np.ones((len(x), 1)), axis=1)
        return x

    def __iniciarPesos(self, tamanho):
        # Função para iniciar os pesos aleatórios de um neurônio
        # Será necessário enviar o tamanho do vetor JÁ CONTANDO COM O BIAS!
        self.pesos = []
        
        for i in range(tamanho):
            self.pesos.append(randint(-1000, 1000) / 1000)
    
    def calcularEntradaLiquida(self, dado):
        #print(self.pesos)
        return np.matmul(np.array(self.pesos).T, dado)

    def __calcularFuncaoAtivacao(self, entradaLiquida):
        if entradaLiquida >= 0:
            return 1
        else:
            return 0

    def __predizerUnicoDado(self, dado):
        # Função para predizer um único dado
        # Como o neurônio é binário, essa função só vai dizer se o dado pertence à classe 1 ou à classe 0. Ela só vai unir as funções de entrada líquida e de ativação.
        return self.__calcularFuncaoAtivacao(self.calcularEntradaLiquida(dado))

    def __calcularAcuraciaTreinamento(self, x, yReal):
        
        qtdAcertos = 0
        
        for dadoAtual, classeReal in zip(x, yReal):
            classePredita = self.__predizerUnicoDado(dadoAtual)
            if classePredita == classeReal:
                qtdAcertos += 1

        return qtdAcertos/len(x)

    def __atualizarPesosUnicaEpoca(self, xTreinamento, yTreinamento, taxaAprendizagem):
        # Função para realizar uma época e atualizando os pesos do neurônio
        # PASSANDO POR CADA DADO
        for dadoAtual, classeReal in zip(xTreinamento, yTreinamento):

            # FAZENDO A PREDICAO DO DADO ATUAL
            classePredita = self.__predizerUnicoDado(dadoAtual)
            
            # CALCULANDO O ERRO
            erro = classeReal - classePredita

            # ATUALIZANDO O VETOR DE PESOS
            self.pesos = self.pesos + taxaAprendizagem * erro * dadoAtual

    def __iterarEpocasTreinamentoNeuronio(self, xTreinamento, yTreinamento, taxaAprendizagem=1e-3, qtdMaxEpocas=1000, percentualSemMelhora=0.5, armazenarEvolucao=True):
        # Função para ajustar os pesos do neurônio época após época
        # É a função que vai ficar responsável por iterar cada época. Ela também dará o critério de parada.
        
        # ARRAYS PARA ARMAZENAR A EVOLUCAO DE CADA EPOCA (VOU PRECISAR PLOTAR ISSO DEPOIS).
        arrayAcuracias = [self.__calcularAcuraciaTreinamento(xTreinamento, yTreinamento)] # ARRAY DE ACURACIAS AO LONGO DAS EPOCAS
        matrizEvolucaoPesos = [self.pesos] # CADA LINHA UMA EPOCA, CADA COLUNA UM PESO
        
        # NO FINAL, EU VOU RETORNAR OS PESOS QUE TIVERAM A MELHOR ACURACIA
        melhorAcuracia = 0
        melhoresPesos  = self.pesos
        
        # TREINANDO POR VARIAS EPOCAS. A EPOCA 0 SAO OS PESOS ALEATORIOS
        # A VARIAVEL qtdEpocasSemMelhoria VAI GUARDAR QUANTAS EPOCAS DE TREINAMENTO SE PASSARAM
        # SEM QUE HOUVESSE MELHORIA NA ACURACIA. SE ESSE VALOR CHEGAR A 25% DA QTD MAXIMA DE ITERACOES
        # O ALGORITMO PARA. ESSE VAI SER UM DOS CRITERIOS DE PARADA TB
        epocas = 0
        qtdEpocasSemMelhoria = 0
        while epocas < qtdMaxEpocas:
            epocas += 1
            
            # COMECO VERIFICANDO A ACURACIA DESSE NEURONIO
            acuraciaAtual = self.__calcularAcuraciaTreinamento(xTreinamento, yTreinamento)
            
            # ARMAZENO A EVOLUCAO DA EPOCA
            if armazenarEvolucao == True:
                arrayAcuracias.append(acuraciaAtual)
                matrizEvolucaoPesos.append(self.pesos)
                
            # VERIFICO SE FOI A MELHOR ITERACAO ATE AGORA
            if acuraciaAtual > melhorAcuracia:
                melhorAcuracia = acuraciaAtual
                melhoresPesos  = self.pesos
                qtdEpocasSemMelhoria = 0
            else:
                qtdEpocasSemMelhoria += 1
                if qtdEpocasSemMelhoria/qtdMaxEpocas >= percentualSemMelhora:
                    self.motivoParada = "Sem melhora na acurácia há " + str(percentualSemMelhora*100) + "% do total de épocas."
                    break
        
            # SE TIVER CONVERGIDO EU PARO
            if acuraciaAtual == 1:
                self.motivoParada = "A acurácia de treinamento chegou a 100%."
                break
                
            # SE NAO TIVER CONVERGIDO EU CONTINUO O ALGORITMO E AJUSTO OS PESOS DESSA EPOCA
            self.__atualizarPesosUnicaEpoca(xTreinamento, yTreinamento, taxaAprendizagem)

        if epocas >= qtdMaxEpocas:
            self.motivoParada = "A quantidade máxima de épocas foi atingida."
            
        # TRANSFORMANDO OS ARRAYS EM NUMPY
        # VOU TRANSPOR A MATRIZ DE EVOLUCAO DOS PESOS, ASSIM, CADA LINHA E UM PESO E CADA COLUNA UMA EPOCA
        epocas              = np.array(epocas)
        arrayAcuracias      = np.array(arrayAcuracias)
        matrizEvolucaoPesos = np.array(matrizEvolucaoPesos).T

        # AGORA QUE TERMINOU, OS PESOS DA CLASSE SERAO OS MELHORES PESOS DESSA FUNCAO
        self.pesos = melhoresPesos
            
        return epocas, arrayAcuracias, matrizEvolucaoPesos

    def __empurraozinho(self, xTreinamento, yTreinamento):
        print("b")
        # Como o neuronio e um classificador binario, se os pesos aleatorios definidos no comeco
        # chegarem em uma acuracia de treinamento de menos de 50%, entao eu posso dar um empurraozinho
        # multiplicando todos os pesos por -1. Isso tudo antes de comecar o treinamento em si.
        # Dessa forma, supondo que o neuronio tinha uma acuracia de 30% antes de comecar o treinamento, 
        # agora ele vai comecar o treinamento já com 70%! Olha que maravilha!
        acuraciaInicial = self.__calcularAcuraciaTreinamento(xTreinamento, yTreinamento)
        if acuraciaInicial < 0.5:
            for i, pesoAtual in enumerate(self.pesos):
                self.pesos[i] = -1 * pesoAtual

    def treinar(self, xTreinamento, yTreinamento, taxaAprendizagem=1e-3, qtdMaxEpocas=1000, percentualSemMelhora=0.5, armazenarEvolucao=True):
        # Função de treinamento do neurônio
        # Primeiro, como tem toda a bagunça de adicionar bias no final dos arrays, vou importar o dataset de novo.
        
        # COLOCANDO A DIMENSAO DO BIAS NO FINAL DE CADA DADO
        xTreinamento = self.__adicionarDimensaoBias(xTreinamento)

        # INICIANDO O VETOR DE PESOS
        self.__iniciarPesos(len(xTreinamento[0]))

        # DANDO UM EMPURRAOZINHO
        self.__empurraozinho(xTreinamento, yTreinamento)

        # TREINANDO DE FATO
        epocas, arrayAcuracias, matrizEvolucaoPesos = self.__iterarEpocasTreinamentoNeuronio(xTreinamento, yTreinamento, taxaAprendizagem, qtdMaxEpocas, percentualSemMelhora, armazenarEvolucao)
        
        # COLOCANDO NA CLASSE
        self.qtdIteracoes      = epocas 
        self.evolucaoAcuracias = arrayAcuracias
        self.evolucaoPesos     = matrizEvolucaoPesos

    def predizer(self, xTeste):
        # Função para predizer todos os dados do conjunto de teste
        # ADICIONANDO UMA DIMENSAO PARA O BIAS
        xTeste = self.__adicionarDimensaoBias(xTeste)
        
        # PREDIZENDO CADA DADO
        yPred = []
        for dadoAtual in xTeste:
            yPred.append(self.__predizerUnicoDado(dadoAtual))
            
        return yPred



