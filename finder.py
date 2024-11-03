import numpy as np
import pygame
from pygame.locals import *
import sys
from classes import *
import keras.api as keras
from keras.api.layers import *
import copy
import time
import tensorflow as tf


def distanceAngle(finalPosition,positionIndividual):
    """
    Retorna um vetor de 2 posições sendo a distância d e o angulo O, em radianos.
    """
    dx = positionIndividual[0] - finalPosition[0]
    dy = positionIndividual[1] - finalPosition[1]
    d = np.sqrt(np.power(dx,2)+np.power(dy,2)) / np.sqrt(np.power(collums,2)+ np.power(rows,2))
    sen = np.sin(np.atan2(dy,dx))
    cos = np.cos(np.atan2(dy,dx))
    return np.array([d,dx,dy])

def misturador_redes(model1,model2):
    w1 = model1.get_weights()
    w2 = model2.get_weights()
    model3 = copy.copy(model1)
    model4 = copy.copy(model1)
    pesos3 = []
    pesos4 = []
    for i in range(len(w1)):
        if i%2:
            pesos3.append(w1[i])
            pesos4.append(w2[i])
        else:
            pesos3.append(w2[i])
            pesos4.append(w1[i])

    model3.set_weights(pesos3)
    model4.set_weights(pesos4)
    return model3,model4


def entradaRede(finalPosition, positionIndividual):
    dx = (finalPosition[0] - positionIndividual[0])/collums
    dy = (finalPosition[1] - positionIndividual[1])/rows
    out = np.array([dx,dy])#/np.linalg.norm([collums,rows])*10
    return out


def create_model():

    input = Input(shape=(2,), name='Entrada')
    x = Dense(5, name='Densa_1')(input)
    #x = Dense(5, name='Densa_2')(x)
    #x = Dense(10, name='Densa_2')(x)
    x = Dense(2, activation='tanh', name='Saida')(x)
    model = keras.models.Model(inputs = input, outputs = x)
    #model.summary()
    return model

def train_model(model):
    weights = model.get_weights()
    new_weights = []
    for weight in weights:
        new_weights.append(weight+np.random.normal(0,mutation_rate,size=weight.shape))
    model.set_weights(new_weights)
    return model

class Individual:

    def __init__(self,x=0,y=0) -> None:
        self.position = np.array([x,y])
        self.velocity = np.array([0,0])

collums = 1280
rows = 720
FPS = 60
n_models = 50
tempo = 10
mutation_rate = 1
speed = 300

posicao_inicial = np.array([[np.random.randint(0,collums),np.random.randint(0,rows)]])
posicao_final = np.array([[np.random.randint(0,collums),np.random.randint(0,rows)]])

pygame.init()
screen = pygame.display.set_mode((collums,rows))
clock = pygame.time.Clock()
clock.tick(FPS)

model = [train_model(create_model()) for _ in range(n_models)]




for geracao in range(100):

    posicao_final = np.array([np.random.randint(0,collums),np.random.randint(0,rows)])
    posicao_inicial = np.array([np.random.randint(0,collums),np.random.randint(0,rows)])
    individuos = [Individual(posicao_inicial[0],posicao_inicial[1]) for _ in range(n_models)]
    t = 0
    mutation_rate /= 2

    while t < tempo:

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0,0,0)) #Limpar a tela
        result = []
        for i in range(n_models):
            result.append(distanceAngle(posicao_final,individuos[i].position))
            pygame.draw.circle(screen,(0,0,255),(individuos[i].position[0],individuos[i].position[1]),5)
        pygame.draw.circle(screen,(255,0,0),(posicao_final[0],posicao_final[1]),10) #Ponto de interesse/final

        result = np.array(result)
        menor1 = np.argmin(result[:,0])
        result[menor1,0] = np.inf
        menor2 = np.argmin(result[:,0])
        pygame.draw.line(screen,(0,255,0),individuos[menor1].position,posicao_final)
        pygame.draw.line(screen,(0,255,0),individuos[menor2].position,posicao_final)
        pygame.display.update()
        

        #input = input_calc(posicao_final,individuos)
        for i in range(n_models):
            input = entradaRede(posicao_final,individuos[i].position)
            input = np.expand_dims(input, axis=0)
            #input = np.expand_dims(input, axis=0)
            #out = np.array(model[i](input, training=False))
            out = model[i].predict_on_batch(input)
            # individuos[i].velocity[0] += out[0,0]*speed*clock.get_time()/1000
            # individuos[i].velocity[1] += out[0,1]*speed*clock.get_time()/1000
            # individuos[i].position[0] += individuos[i].velocity[0]*clock.get_time()/1000
            # individuos[i].position[1] += individuos[i].velocity[1]*clock.get_time()/1000
            delta_T = clock.get_time()
            individuos[i].position[0] += out[0,0]*speed*delta_T/1000.
            individuos[i].position[1] += out[0,1]*speed*delta_T/1000.

        
        t += clock.get_time()/1000.

    score = np.array([distanceAngle(posicao_final,individuos[i].position)[0] for i in range(n_models)])
    score /= score.sum()
    best_model1 = model[np.argmin(score)]
    score[np.argmin(score)] = np.inf
    best_model2 = model[np.argmin(score)]
    model = []
    model.append(best_model1)
    model.append(best_model2)
    best_model1,best_model2 = misturador_redes(best_model1,best_model2)
    model.append(best_model1)
    model.append(best_model2)    
    for _ in range(n_models-4):
        model.append(train_model(copy.copy(best_model1)))
    pygame.display.update()