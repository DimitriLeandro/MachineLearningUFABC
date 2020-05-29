import librosa
import numpy as np

def normalizarEnergia(sinalOriginal):
	return sinalOriginal/np.std(sinalOriginal)

def adicionarRuidoBrancoAleatoriamente(sinalOriginal, snrMinima=10, snrMaxima=20):
    # Função para adição de ruído branco com SNR aleatória
	# A distribuição da SNR será uniforme entre 10 dB e 20 dB.
    snr          = np.random.uniform(snrMinima, snrMaxima)
    energiaRuido = np.var(sinalOriginal)/(10**(snr/10))
    ruido        = np.random.normal(0, 1, size=len(sinalOriginal)) * energiaRuido**(1/2)
    return sinalOriginal + ruido, snr

def deslocarTempoAleatoriamente(sinalOriginal, percentualMinimo=-0.4, percentualMaximo=0.4):
	# A distribuição do percentual do tamanho do sinal original que será deslocado também será uniforme. O percentual pode ser negativo ou positivo, indicando se o sinal resultante será adiantado ou atrasado em relação ao original. A distribuição ficará entre -0.4 e +0.4.
    percentualDeslocamento = np.random.uniform(percentualMinimo, percentualMaximo)
    while percentualDeslocamento > -0.1 and percentualDeslocamento < 0.1:
    	percentualDeslocamento = np.random.uniform(percentualMinimo, percentualMaximo)
    qtdAmostrasDeslocadas  = int(np.absolute(percentualDeslocamento) * len(sinalOriginal))
    if percentualDeslocamento > 0:
        return np.append(sinalOriginal[qtdAmostrasDeslocadas:], np.zeros(qtdAmostrasDeslocadas)), percentualDeslocamento
    else:
        return np.append(np.zeros(qtdAmostrasDeslocadas), sinalOriginal[0:-qtdAmostrasDeslocadas]), percentualDeslocamento

def alterarVelocidadeAleatoriamente(sinalOriginal, velocidadeMinima=0.5, velocidadeMaxima=2):
    # A velocidade será uniforme entre 0.5 e 2.
    velocidade = np.random.uniform(velocidadeMinima, velocidadeMaxima)
    while velocidade > 0.9 and velocidade < 1.1:
    	velocidade = np.random.uniform(velocidadeMinima, velocidadeMaxima)
    return librosa.effects.time_stretch(sinalOriginal, velocidade), velocidade

def mudarPitchAleatoriamente(sinalOriginal, freqAmostragem, pitchMinimo=-7, pitchMaximo=7):
    # O pitch terá distribuição uniforme entre -7 e 7.
    pitch = np.random.uniform(pitchMinimo, pitchMaximo)
    while pitch > -1 and pitch < 1:
    	pitch = np.random.uniform(pitchMinimo, pitchMaximo)
    return librosa.effects.pitch_shift(sinalOriginal, freqAmostragem, pitch), pitch
