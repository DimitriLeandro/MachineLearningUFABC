# TODA A IMPLEMETACAO DESSA CLASSE ESTA MUITO BEM DOCUMENTADA NO JUPYTER "
# Implementação de classe para extração de features - SESA Dataset"
# VOU COLOCAR O MAXIMO DE COMENTARIOS POSSIVEIS AQUI TAMBEM, MAS PARA MAIS INFORMACOES
# E MELHOR OLHAR POR LA

# COMO USAR A CLASSE:
# A CLASSE ENTRA EM UMA DETERMINADA PASTA CONTENDO APENAS ARQUIVOS WAV E CRIA UM CSV DE FEATURES
# A CLASSE TAMBEM E RESPONSAVEL POR ESCALONAR AS FEATURES E FAZER UMA REDUCAO DE DIMENSIONALIDADE

# PRIMEIRO, OS PARAMETROS ABAIXOS DEVEM SER SETADOS:
# diretorio      -> PASTA ONDE A CLASSE VAI PROCURAR PELOS WAVs PARA GERAR O CSV
# freqAmostragem -> A FREQUENCIA DE AMOSTRAGEM DOS AUDIOS DESSA PASTA
# frameLength    -> O TAMANHO DAS JANELAS DE CADA AUDIO EM QTD DE FRAMES
# overlapLength  -> TAMANHO DA SOBREPOSICAO EM QTD DE FRAMES

# O COMANDO ABAIXO INSTANCIA A CLASSE. O CONSTRUTOR DEVOLVE UM DATAFRAME PANDAS AO MESMO TEMPO QUE CRIA O CSV
# dataframeGeral = extrairFeatures(diretorio, freqAmostragem, frameLength, overlapLength)

import os
import librosa
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

class ExtrairFeatures:

	arrayTempoProcessamentoCadaAudioCompleto = [] # ARRAY QUE VAI QUARDAR O TEMPO PARA EXTRAIR AS FEATURES DE TODAS AS JANELAS DE UM UNICO AUDIO

	# CONSTRUTOR---------------------------------------------------------------------------------------------
	def __init__(self, diretorio, freqAmostragem, frameLength, overlapLength, escalonamento=True):    
		# PRIMEIRO EU CRIO O DATAFRAME DE TODOS OS AUDIOS DA PASTA
		dataframeGeral = self.montarDataframeTodosOsAudios(diretorio, freqAmostragem, frameLength, overlapLength)
		
		# DEPOIS EU ESCALONO AS FEATURES
		if escalonamento == True:
			dataframeGeral = self.escalonarFeatures(dataframeGeral)
			
		# PARA FINALIZAR, VOU ESCREVER O DATAFRAME NUM CSV
		nomeCSV = diretorio + str(time.time()) + ".csv"
		print("Escrevendo CSV:", nomeCSV)
		dataframeGeral.to_csv(nomeCSV, index=False)
		
		# TAMBEM VOU RETORNAR O DATAFRAME, PQ VAI QUE EU PRECISO NE
		print("Operação finalizada")
		
		# return dataframeGeral

	# DEFINICAO DE FUNCOES INTERMEDIARIAS--------------------------------------------------------------------	
	def fazerJanelamento(self, sinal, frameLength, overlapLength):
		# Função que faz o janelamento
		# Existe um problema em deixar que as funções de extração de features criadas acima façam o janelamento: 
		# ao invés de retornarem valores unitários para as features, elas vão retornar um array em que cada posição 
		# representa um janelamento. Portanto, a solução é fazer o janelamento antes de extrair as features e deixar 
		# para mandar para essas funções apenas as janelas, fazendo com que frameLength seja igual ao tamanho da 
		# janela que está sendo enviada e que overlapLength seja 0.
		# A função abaixo usa a função frame do librosa que retorna as janelas como COLUNAS. 
		# Como eu quero que cada janela seja uma LINHA, eu retorno a transposta dessa função.
		return librosa.util.frame(sinal, frame_length=frameLength, hop_length=overlapLength).T

	def extrairFeaturesUnicoFrame(self, sinal, freqAmostragem, frameLength):
		# Função que extrai features de um único frame
		# Em algum momento, a classe deverá passar em cada um dos áudios da pasta para ir extraindo as features. 
		# A próxima função extrai as features de um único frame de áudio e retorna um array com essas features. 
		# Posteriormente esse array deverá ser integrado à matriz de features de todos os áudios.
		# Haverá uma outra função para extrair as features de um único áudio. Ela deverá pegar um áudio, usar a 
		# função de janelamento, e usar a função abaixo para extrair as features de cada uma das janelas. Depois, 
		# ela vai retornar uma matriz com as features de cada frame de um único áudio.
		# 
		# PARA IMPEDIR QUE O LIBROSA CONTINUE FAZENDO O JANELAMENTO DO ÁUDIO MESMO QUE frameLength 
		# SEJA DO TAMANHO DO ÁUDIO, O PARÂMETRO DE OVERLAP DEVE SER MAIOR QUE frameLength
		overlapLength = frameLength + 1
		
		# CRIANDO O ARRAY DE FEATURES DO FRAME EM QUESTAO
		arrayFeaturesFrame = []
		
		# PRIMEIRO, E PRECISO CRIAR A MATRIZ DOS MFCCS
		matrizMFCC          = self.extrairMatrizMFCC(sinal, freqAmostragem)
		
		# AGORA SIM EU SAIO EXTRAINDO AS FEATURES 
		arrayFeaturesFrame += self.extrairMFCCs(matrizMFCC)
		arrayFeaturesFrame += self.extrairDeltas(matrizMFCC)
		arrayFeaturesFrame += self.extrairDeltaDeltas(matrizMFCC)

		# POR FIM, RETORNO O ARRAY DE FEATURES DO AUDIO QUE FOI ENVIADO PARA ESSA FUNCAO
		return arrayFeaturesFrame

	def extrairFeaturesUnicoAudio(self, sinal, freqAmostragem, frameLength, overlapLength):
		# Função que extrai as features de um único áudio
		# Essa função vai pegar um áudio, usar a função de janelamento, e para cada janela do áudio em questão, 
		# ela vai usar a função de extrair as features de uma única janela (implementada acima). 
		# Depois, ela vai retornar uma matriz com as features de cada frame de um único áudio, onde cada linha 
		# é uma frame e cada coluna é uma feature.

		# PRIMEIRO, VOU CRIAR A MATRIZ QUE VAI CONTER AS FEATURES DE CADA JANELA
		# CADA LINHA E UMA JANELA E CADA COLUNA E UMA FEATURE
		matrizFeaturesAudio = []

		# COMECO A MEDIR O TEMPO DE PROCESSAMENTO PARA JANELAR O AUDIO E EXTRAIR AS FEATURES DE TODAS AS JANELAS DESSE AUDIO
		tempoInicio = time.time()
		
		# DEPOIS, VOU FAZER O JANELAMENTO
		matrizFramesAudio = self.fazerJanelamento(sinal, frameLength, overlapLength)
		
		# AGORA, PARA CADA JANELA, VOU EXTRAIR AS FEATURES E COLOCAR COMO UMA LINHA NOVA NA MATRIZ
		for frameAtual in matrizFramesAudio:
			matrizFeaturesAudio.append(self.extrairFeaturesUnicoFrame(frameAtual, freqAmostragem, frameLength))

		# TERMINO DE MEDIR O TEMPO
		tempoFim = time.time()
		self.arrayTempoProcessamentoCadaAudioCompleto.append(tempoFim - tempoInicio)
		
		# RETORNO A MATRIZ DE FEATURES DESSE AUDIO
		return matrizFeaturesAudio

	def adicionarNomeArquivoEClasse(self, matrizFeaturesAudioAtual, nomeArquivo, classificacaoCorreta):
		#Função que coloca o nome do arquivo e a classificação correta na matriz de features de um áudio
		#Haverá uma matriz de dados que terá as seguintes colunas: nomeArquivo, ...features... e classificacaoCorreta. 
		#Mas, a função implementada para gerar a matriz de features de um único áudio extrairFeaturesUnicoAudio 
		#devolve uma matriz de features sem o nome do áudio e a classificação correta. Portanto, a função abaixo 
		# apenas pega essa última matriz e coloca o nome do arquivo no começo e a classificação correta ao final. 
		# Posteriormente, o resultado será agregado à matriz de dados citada em primeiro lugar.

		#ESSA FUNÇÃO RETORNA UM DATAFRAME PANDAS, NÃO UMA MATRIZ QUALQUER.

		#https://www.geeksforgeeks.org/python-pandas-dataframe-insert/

		# PRIMEIRO TRANSFORMO A MATRIZ NUM PANDAS DATAFRAME
		dataframeAudioAtual = pd.DataFrame(matrizFeaturesAudioAtual)

		# ALERTA DE MUDANCA DE CODIGO PARA A DISCIPLINA DE MACHINE LEARNING DA POS
			# seguinte: essa classe toda ja estava implementada por causa da minha IC
			# mas agr a gnt vai precisar dela pro projeto final de machine learning da pos
			# a maior diferenca de todas vai ser feita aqui
			# as unicas coisas que eu mudei que nao estao nessa funcao aqui foram:
			# exclui todas as funcoes de extracao que nao sao de mfcc, delta e delta delta
			# ou seja, essa classe aqui so extrai essas 3 features, sendo 20 pra cada, totalizando 60 features
			# fora isso eu so tirei a funcao que reduzia a dimensionalidade com PCA pq aquela funcao ja tava uma merda mesmo
			# BELEZA, QUAL E A GRANDE MUDANCA QUE VAI ACONTECER AGORA?
			# bem simples
			# preciso adicionar mais 4 colunas ao CSV final: velocidade, ruido, tempo e pitch
			# essas colunas so vao ser preenchidas com 0 e 1, indicando se o audio atual sofreu esses efeitos
			# e so pra isso mesmo
		# FIM DO ALERTA DE MUDANCA DE CODIGO PARA A DISCIPLINA DE MACHINE LEARNING DA POS

		# VERIFICANDO QUAIS FORAM OS EFEITOS APLICADOS NESSE AUDIO
		arrayEfeitos = nomeArquivo[:-4].split("_")[2:]

		velocidade = 0
		tempo      = 0
		pitch      = 0
		ruido      = 0

		if "velocidade" in arrayEfeitos:
		    velocidade = 1
		if "tempo" in arrayEfeitos:
		    tempo = 1
		if "pitch" in arrayEfeitos:
		    pitch = 1
		if "ruido" in arrayEfeitos:
		    ruido = 1

		# AGORA EU COLOCO ESSAS COLUNAS NO DATAFRAME ATUAL .insert(posicaoNovaColuna, nomeNovaColuna, valorParaTodasAsLinhas)
		dataframeAudioAtual.insert(0, "velocidade", velocidade, True)
		dataframeAudioAtual.insert(0, "tempo", tempo, True)
		dataframeAudioAtual.insert(0, "pitch", pitch, True)
		dataframeAudioAtual.insert(0, "ruido", ruido, True)
		
		# AGORA COLOCO A COLUNA DO NOME NO COMECO (posicaoNovaColuna, nomeNovaColuna, valorParaTodasAsLinhas)
		dataframeAudioAtual.insert(0, "nomeArquivo", nomeArquivo, True)
		
		# AGORA COLOCO A COLUNA DA CLASSIFICACAO CORRETA NA ULTIMA POSICAO
		dataframeAudioAtual.insert(len(dataframeAudioAtual.columns), "classificacaoCorreta", classificacaoCorreta, True)
		
		return dataframeAudioAtual

	def verificarClassificacaoCorreta(self, nomeArquivo):
		#Função para verificar qual é a classificação correta de acordo com o nome do arquivo
		#Essa é bem básica. Em algum determinado momento eu vou precisar saber a classficação 
		# correta de um determinado áudio. Eu só consigo saber isso pelo nome do arquivo. LEMBRANDO QUE 
		# TODO O CÓDIGO ESCRITO AQUI SERVE PARA O BANCO DE DADOS SESA. Os arquivos são nomeados 
		# como classe_contador.wav. Por exemplo: casual_000.wav, explosion_032.wav e gunshot_032.wav.
		arrayNome = nomeArquivo.split("_")
		return arrayNome[0]

	def montarDataframeTodosOsAudios(self, diretorio, freqAmostragem, frameLength, overlapLength):
		# Função que passa por todos os áudios da pasta e vai montando o dataframe
		# Essa é função que une todas as outras. Ela vai passar por todos os áudios da pasta, 
		# verificar qual é a classificação correta de cada áudio, extrair as features, gerar o 
		# dataframe de features do áudio atual e agregar ao dataframe de todos os áudios.
	
		# CRIANDO O ARRAY COM O NOME DOS ARQUIVOS
		arrayNomeArquivos = os.listdir(diretorio)
		
		# PEGANDO O TOTAL DE ARQUIVOS NA PASTA APENAS PARA FINS DE PRINT
		totalArquivosNaPasta = len(arrayNomeArquivos)
		
		# ABAIXO, VOU CRIAR O DATAFRAME DE TODOS OS AUDIOS
		dataframeGeral = pd.DataFrame()

		# VOU PASSAR POR TODOS OS AUDIOS DO DIRETORIO
		arrayDataframesUnicoAudio = Parallel(n_jobs=-1, verbose=100)(delayed(self.montarDataframeUnicoAudio)(nomeArquivo, diretorio, freqAmostragem, frameLength, overlapLength, i, totalArquivosNaPasta) for i, nomeArquivo in enumerate(arrayNomeArquivos))

		for dataframeAudioAtual in arrayDataframesUnicoAudio:
			# CONCATENANDO O DATAFRAME DO AUDIO ATUAL AO DATAFRAME GERAL
			dataframeGeral = pd.concat([dataframeGeral, dataframeAudioAtual])
			
		# POR ULTIMO, RETORNO O DATAFRAME GERAL
		return dataframeGeral

	def montarDataframeUnicoAudio(self, nomeArquivo, diretorio, freqAmostragem, frameLength, overlapLength, i, totalArquivosNaPasta):

		# PRINTANDO O PROGRESSO
		#print("Extraindo features do arquivo", i+1, "de", totalArquivosNaPasta, "-> " + str(100*((i+1)/totalArquivosNaPasta)) + "%")

		# ABRO O AUDIO ATUAL COM O LIBROSA
		audioAtual, freqAmostragem = librosa.load(diretorio+nomeArquivo, sr=freqAmostragem, mono=True) 
		
		# VERIFICAO QUAL E A CLASSIFICACAO CORRETA
		classeAudioAtual = self.verificarClassificacaoCorreta(nomeArquivo)
		
		# MONTO A MATRIZ DE FEATURES DE CADA FRAME DO AUDIO ATUAL (a funcao abaixo
		# devolve uma matriz normal e faz o janelamento em outra funcao tb)
		matrizFeaturesAudioAtual = self.extrairFeaturesUnicoAudio(audioAtual, freqAmostragem, frameLength, overlapLength)
		
		# AGORA E HORA DE COLOCAR O NOME E A CLASSIFICACAO CORRETA NA MATRIZ
		# MAAAS, NA FUNCAO ABAIXO, O BICHO VIRA UM DATAFRAME PANDAS
		dataframeAudioAtual = self.adicionarNomeArquivoEClasse(matrizFeaturesAudioAtual, nomeArquivo, classeAudioAtual)
		return dataframeAudioAtual

	def escalonarFeatures(self, dataframeGeral):
		
		# Função para escalonar as features
		# Essa função recebe o dataframe geral, remove as cinco primeiras e última colunas (nome, efeitos(são 4) e classificação), 
		# faz o escalonamento e depois coloca as colunas de nome e classificação de volta.
	
		#print("Escalonando features")
		
		# COPIANDO AS COLUNAS DE NOME, EFEITOS E CLASSIFICACAO
		colunaArquivo       = dataframeGeral["nomeArquivo"]
		colunaVelocidade    = dataframeGeral["velocidade"]
		colunaTempo         = dataframeGeral["tempo"]
		colunaPitch         = dataframeGeral["pitch"]
		colunaRuido         = dataframeGeral["ruido"]
		colunaClassificacao = dataframeGeral["classificacaoCorreta"]
		
		# DELETANDO AS COLUNAS ARQUIVO, EFEITOS E CLASSIFICACAO
		dataframeGeral = dataframeGeral.drop(['nomeArquivo', 'velocidade', 'tempo', 'pitch', 'ruido', 'classificacaoCorreta'], axis=1)
		
		# ESCALONANDO
		dataframeGeral = pd.DataFrame(StandardScaler().fit_transform(dataframeGeral))
		
		# COM ESSA COISA DE TIRA E POE COLUNA, O PANDAS NAO SABE LINDAR COM OS INDEXES,
		# ABAIXO EU ESTOU RESETANDO TUDO
		colunaArquivo.reset_index(inplace=True, drop=True)
		colunaVelocidade.reset_index(inplace=True, drop=True)
		colunaTempo.reset_index(inplace=True, drop=True)
		colunaPitch.reset_index(inplace=True, drop=True)
		colunaRuido.reset_index(inplace=True, drop=True)
		colunaClassificacao.reset_index(inplace=True, drop=True)
		dataframeGeral.reset_index(inplace=True, drop=True)
		
		# ADICIONANDO AS COLUNAS QUE FORAM EXCLUIDAS (posicaoNovaColuna, nomeNovaColuna, valorParaTodasAsLinhas)
		dataframeGeral.insert(0, "velocidade", colunaVelocidade, True)
		dataframeGeral.insert(0, "tempo", colunaTempo, True)
		dataframeGeral.insert(0, "pitch", colunaPitch, True)
		dataframeGeral.insert(0, "ruido", colunaRuido, True)
		dataframeGeral.insert(0, "nomeArquivo", colunaArquivo, True)
		dataframeGeral.insert(len(dataframeGeral.columns), "classificacaoCorreta", colunaClassificacao, True)
		
		return dataframeGeral

	def extrairMatrizMFCC(self, sinal, freqAmostragem):
		return librosa.feature.mfcc(y=sinal, sr=freqAmostragem)

	def extrairMFCCs(self, matrizMFCC):
		
		arrayMFCCs = []
		
		for linha in matrizMFCC:
			arrayMFCCs.append(np.mean(linha))
			
		return arrayMFCCs

	def extrairDeltas(self, matrizMFCC):
		matrizDelta = librosa.feature.delta(matrizMFCC, order=1)

		arrayDelta = []

		for linha in matrizDelta:
			arrayDelta.append(np.mean(linha))

		return arrayDelta

	def extrairDeltaDeltas(self, matrizMFCC):
		matrizDeltaDelta = librosa.feature.delta(matrizMFCC, order=2)

		arrayDeltaDelta = []

		for linha in matrizDeltaDelta:
			arrayDeltaDelta.append(np.mean(linha))

		return arrayDeltaDelta
