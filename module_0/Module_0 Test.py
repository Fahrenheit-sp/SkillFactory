#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def score_game(algorithm):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    range_start = 1
    range_end = 101
    attempts = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(range_start,range_end, size=(1000))
    for number in random_array:
        attempts.append(algorithm(number, range_start, range_end))
    score = int(np.mean(attempts))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return(score)

def binary_search(number, range_start, range_end):
    attempts = 1
    predict = int(abs(range_end - range_start) / 2) #устанавливаем предположение в середину радиуса
    #переменные для сохранения радиуса поиска. изначально число больше начала радиуса и меньше конца радиуса
    greater_than = range_start-1
    less_than = range_end
    while number != predict:
        attempts += 1
        if number > predict:
            greater_than = predict #устанавливаем новое нижнее число радиуса поиска
            predict += int((less_than - greater_than) / 2)  #устанавливаем предположение в середину нового радиуса
        elif number < predict:
            less_than = predict #устанавливаем новое верхнее число радиуса поиска
            predict -= int((less_than - greater_than) / 2)
    return(attempts)
    
# Проверяем
score_game(binary_search)

