#!/usr/bin/env python
# coding: utf-8

# # Загрузка Pandas и очистка данных

# In[721]:


import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler


# In[722]:


df = pd.read_csv('main_task.csv')
df.columns = ['restaurant_id', 'city', 'cuisine', 'ranking', 'rating',
              'price_range', 'reviews_count', 'reviews', 'url_ta', 'id_ta']


# <h3>Вспомогательные функции</h3>

# In[723]:


def convert_string_to_list(string):
    if type(string) is str:
        return string.strip('][').replace('\'', '').split(', ')
    if type(string) is list:
        return string
    else:
        return []


def get_unique_values_from(series):
    result = set()
    for value in series.values:
        if value == value:
            values_list = convert_string_to_list(value)
            for subvalue in values_list:
                if subvalue != '':
                    result.add(subvalue)
    return result


df_length = df.shape[0]

def create_zeros_series():
    return pd.Series(np.zeros(df_length), dtype='int')


# In[724]:


# Ваш код по очистке данных и генерации новых признаков
# При необходимости добавьте ячейки


# In[725]:


# удалим столбцы, не представляющие ценности
df.drop(['restaurant_id','url_ta', 'id_ta'], 1, inplace=True)

# заполним пропуски в reviews_count
df.reviews_count = df.reviews_count.fillna(0)


# <h3>Города</h3>
# 
# сгенерируем dummy переменные для городов (в последствии я увидел хороший способ на kaggle, через get_dummies, но я решил оставить как есть, потому что так честно)
# к тому же, я возможно еще буду использовать этот столбец для группировки далее (хотя ничто не мешало мне точно так же сгенерировать dummy после использования столбца :D )

# In[726]:


cities = get_unique_values_from(df.city)
for city in cities:
    df[city] = df.city.apply(lambda x: 1 if city == x else 0)

# в курсе был вопрос, какие еще можно параметры извлечь из городов, а так же 
# на звонке с куратором по проекту поднимался вопрос. собственно там я и предложил возможно имеет смысл
# взять численность населения в городах. попробуем
populations = {'Amsterdam': 1148972,
              'Athens': 3153355,
              'Barcelona': 5585556,
              'Berlin': 3562038,
              'Bratislava': 424428,
              'Brussels': 2080788,
              'Budapest': 1768073,
              'Copenhagen': 1346485,
              'Dublin': 1228179,
              'Edinburgh': 536775,
              'Geneva': 613373,
              'Hamburg': 1789954,
              'Helsinki': 1304851,
              'Krakow': 768731,
              'Lisbon': 2956879,
              'Ljubljana': 279631,
              'London': 9304016,
              'Luxembourg': 613894,
              'Lyon': 1719268,
              'Madrid': 6617513,
              'Milan': 3140181,
              'Munich': 1538302,
              'Oporto': 1312947,
              'Oslo': 1041377,
              'Paris': 11017230,
              'Prague': 1305737,
              'Rome': 4257056,
              'Stockholm': 1632798,
              'Vienna': 1929944,
              'Warsaw': 1783251,
              'Zurich': 1395356}

# добавим значения в датафрейм
df['populations'] = df.city.apply(lambda x: populations[x])

# нормализуем значения
cities_scaler = MinMaxScaler()
result = cities_scaler.fit_transform(np.array(df.populations).reshape(-1, 1)).reshape(1, -1)[0]
df.populations = pd.Series(result)


# <h3>Разберем типы кухонь</h3>

# In[727]:


# заполним пропуски
df.cuisine = df.cuisine.fillna('')

# получим все уникальные кухни из датасета
cuisines = get_unique_values_from(df.cuisine)

# добавим новые столбцы в датасет для каждого типа кухонь, 1 говорит что кухня присутствует в данном ресторане
for cuisine in cuisines:
    df[cuisine] = df.cuisine.apply(lambda x: 1 if cuisine in x else 0)
    
# в столбец cuisiune запишем количество кухонь представленных в ресторане
def update_cuisines_count(row):
    cuisines_list = row[list(cuisines)] #получаем часть строки, в которой хранятся все столбцы с кухнями
    row['cuisine'] = sum(cuisines_list) # количество кухонь равно сумме всех единиц в списке возможных кухонь
    return row

df = df.apply(update_cuisines_count, axis=1)

# переименуем колонку
if 'cuisine_count' not in df.columns:
    df.rename(columns={'cuisine': 'cuisine_count'}, inplace=True)


# <h3>Разбор дат отзывов</h3>

# In[728]:


# получим даты из строки с отзывами и обработаем их
def get_review_dates_from(row):
    reviews = row['reviews']
    dates = re.findall(r'\d{2}/\d{2}/\d{4}', reviews)
    datetimes = list(map(pd.to_datetime, dates))
    row['last_review_date'] = pd.NaT if not datetimes else max(datetimes)
    row['previous_review_date'] = pd.NaT if not datetimes else min(datetimes)
    row['days_between_reviews'] = 0 if not datetimes else (row['last_review_date'] - row['previous_review_date']).days
    return row

# добавим новые столбцы для дат последнего/предпоследнего отзывов,
# а так же для количества дней между отзывами
if 'last_review_date' not in df.columns:
    df['last_review_date'] = create_zeros_series()
    df['previous_review_date'] = create_zeros_series()
    df['days_between_reviews'] = create_zeros_series()
    df = df.apply(get_review_dates_from, axis=1)


# <h3>Разбор тональности отзывов</h3>

# изначально, я планировал просто удалить оставшийся столбец с отзывами, но затем я вспомнил о материале об определении тональности текста из курса по ML и решил сделать свой, только на минималках :) я добавлю два столбца 'has_good_vibes', и 'has_bad_vibes', в который буду записывать 1/0 если в тексте отзывов есть "хорошие" слова типа "good, perfect, excellent", или плохие "bad, poor, unsatisfied" и т.д. при этом я понимаю что могут быть случаи типа "not bad", и всех их я не могу обработать, поэтому постараюсь покрыть максимальное количество слов/словосочетаний :)

# In[729]:


good_words = ['good', 'perfect', 'not bad', 'excellent',
              'great', ' fine ', 'exceptional', 'super',
              'nice', 'positive', 'satisfying', 'acceptable']
bad_words = [' bad ', 'awful', 'poor', ' sad ', 'unacceptable',
             'garbage', 'junky', 'not good']

def determine_reviews_vibe_for(row):
    reviews = row['reviews']
    # возможно, следовало бы обрабатывать одновременно наличие хороших и отстутсвие плохих
    # оттенков в отзывах (и наоборот), но это сильно повлияет на производительность, поэтому
    # будем считать что все отзывы могут быть только одной тональности
    for word in good_words:
        if word in reviews:
            row['has_good_vibes'] = 1
    for word in bad_words:
        if word in reviews:
            row['has_bad_vibes'] = 1
    return row
            
if 'has_good_vibes' not in df.columns:
    df['has_good_vibes'] = create_zeros_series()
    df['has_bad_vibes'] = create_zeros_series()
    df = df.apply(determine_reviews_vibe_for, axis=1)


# на этом этапе, думаю, столбец reviews можно удалить за ненадобностью
if 'reviews' in df.columns:
    df.drop(['reviews'], 1, inplace=True)


# <h3>Разбор рейтингов по городам</h3>
# предлагаю нормализовать данные значения

# In[730]:


scaler = MinMaxScaler()
result = scaler.fit_transform(np.array(df.ranking).reshape(-1, 1)).reshape(1, -1)[0]
df.ranking = pd.Series(result)


# <h3>Разбор диапазона цен</h3>
# заменим условные обозначения на упорядоченные признаки, т.е. 1 - низкие цены ($), 3 - высокие и подумаем как заполнить пропуски

# In[731]:


df.price_range.value_counts(dropna=False)
price_dict = {'$': 1,
              '$$ - $$$': 2,
              '$$$$': 3}
if df.price_range.dtype == 'object':
    df.price_range = df.price_range.apply(lambda x: 0 if x != x else price_dict[x])

# для заполнения пропусков возьмем медианное значение цены в данном городе
median_group = df.groupby(by=['city'])['price_range'].median()
# видим что в двух городах медианное значение == 0, заменим их на 1
median_group['Bratislava'] = 1
median_group['Hamburg'] = 1

#заполним пропуски
df.price_range = np.where(df['price_range'] == 0, median_group[df['city']], df['price_range'])


# <h2>Работа с моделью</h2>

# In[732]:


# Разбиваем датафрейм на части, необходимые для обучения и тестирования модели
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
X = df.drop(['city', 'rating', 'last_review_date', 'previous_review_date'], axis = 1)
y = df['rating']


# In[733]:


# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split


# In[734]:


# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# # Создаём, обучаем и тестируем модель

# In[735]:


# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели


# In[736]:


# Создаём модель
regr = RandomForestRegressor(n_estimators=100)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)


# In[737]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

feat_importances = pd.Series(regr.feature_importances_, index=X.columns)
feat_importances.nlargest(15)

