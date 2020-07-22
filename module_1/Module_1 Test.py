#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# In[126]:


data = pd.read_csv('module_1.csv')


# In[127]:


data.describe()


# # Предобработка

# In[128]:


answers = {} # создадим словарь для ответов

# Разделение значений с вертикальными линиями
def split_vertical_line_values(values):
    return values.split('|')

data['genres_split'] = data.genres.apply(split_vertical_line_values)
data['directors'] = data.director.apply(split_vertical_line_values)
data['actors'] = data.cast.apply(split_vertical_line_values)
data['companies'] = data.production_companies.apply(split_vertical_line_values)

# Расчет прибыльности
def calculate_profit(row):
    return row['revenue'] - row['budget']

data['profit'] = data.apply(calculate_profit, axis=1)

# Получение дат
data['date'] = pd.to_datetime(data.release_date)
data['month'] = data.date.dt.month

# Вспомогательные функции
def get_counter_for(series):
    counter = Counter()
    splitted_values = series.values
    for values in splitted_values:
        for value in values:
            counter[value] += 1
    return counter

def calculate_average_length_for_companies_in(df, column='original_title', isCountingSymbols=True):
    length_counter = Counter()
    movies_counter = Counter()
    for companies, value in zip(df['companies'], df[column]):
        for company in companies:
            length_counter[company] += len(value) if isCountingSymbols else len(value.split())
            movies_counter[company] += 1
    average_counter = Counter()
    for company in movies_counter:
        movies_count = movies_counter[company]
        total_length = length_counter[company]
        average_counter[company] = total_length / movies_count
    return average_counter


# # 1. У какого фильма из списка самый большой бюджет?

# Использовать варианты ответов в коде решения запрещено.    
# Вы думаете и в жизни у вас будут варианты ответов?)

# In[129]:


# в словарь вставляем номер вопроса и ваш ответ на него
answers['1'] = '5. Pirates of the Caribbean: On Stranger Tides (tt1298650)' 
# +


# In[130]:


# тут пишем ваш код для решения данного вопроса:
max_budget = data.budget.max()
max_budget_movie = data[data.budget == max_budget]
max_budget_movie


# ВАРИАНТ 2

# In[131]:


# можно добавлять разные варианты решения


# # 2. Какой из фильмов самый длительный (в минутах)?

# In[132]:


# думаю логику работы с этим словарем вы уже поняли, 
# по этому не буду больше его дублировать
answers['2'] = '2. Gods and Generals (tt0279111)'
# +


# In[133]:


max_runtime = data.runtime.max()
longest_movie = data[data.runtime == max_runtime]
longest_movie


# # 3. Какой из фильмов самый короткий (в минутах)?
# 
# 
# 
# 

# In[134]:


answers['3'] = '3. Winnie the Pooh (tt1449283)'
# +
min_runtime = data.runtime.min()
shortest_movie = data[data.runtime == min_runtime]
shortest_movie


# # 4. Какова средняя длительность фильмов?
# 

# In[135]:


answers['4'] = '2. 110'
# +
average_runtime = data.runtime.mean()
average_runtime


# # 5. Каково медианное значение длительности фильмов? 

# In[136]:


answers['5'] = '1. 107'
# +
median_runtime = data.runtime.median()
median_runtime


# # 6. Какой самый прибыльный фильм?
# #### Внимание! Здесь и далее под «прибылью» или «убытками» понимается разность между сборами и бюджетом фильма. (прибыль = сборы - бюджет) в нашем датасете это будет (profit = revenue - budget) 

# In[137]:


answers['6'] = '5. Avatar (tt0499549)'
# +
max_profit = data.profit.max()
most_profitable_movie = data[data.profit == max_profit]
most_profitable_movie


# # 7. Какой фильм самый убыточный? 

# In[138]:


answers['7'] = '5. The Lone Ranger (tt1210819)'
# +
min_profit = data.profit.min()
least_profitable_movie = data[data.profit == min_profit]
least_profitable_movie


# # 8. У скольких фильмов из датасета объем сборов оказался выше бюджета?

# In[139]:


answers['8'] = '1. 1478'
# +
profitable_movies = data[data.profit > 0]
len(profitable_movies)


# # 9. Какой фильм оказался самым кассовым в 2008 году?

# In[140]:


answers['9'] = '4. The Dark Knight (tt0468569)'
# +

movies_in_2008 = data.query('release_year == 2008')
max_revenue_in_2008 = movies_in_2008.revenue.max()
most_revenue_movie_in_2008 = movies_in_2008[movies_in_2008.revenue == max_revenue_in_2008]
most_revenue_movie_in_2008


# # 10. Самый убыточный фильм за период с 2012 по 2014 г. (включительно)?
# 

# In[141]:


answers['10'] = '5. The Lone Ranger (tt1210819)'
# +
movies_from_2012_to_2014 = data[data.release_year.isin([2012, 2013, 2014])]
least_profit_in_period = movies_from_2012_to_2014.profit.min()
least_profitable_movie_in_period = movies_from_2012_to_2014[movies_from_2012_to_2014.profit == least_profit_in_period]
least_profitable_movie_in_period


# # 11. Какого жанра фильмов больше всего?

# In[142]:


# эту задачу тоже можно решать разными подходами, попробуй реализовать разные варианты
# если будешь добавлять функцию - выноси ее в предобработку что в начале
answers['11'] = '3. Drama'
# +
genres_counter = get_counter_for(data.genres_split)
genres_counter.most_common()


# ВАРИАНТ 2

# In[ ]:





# # 12. Фильмы какого жанра чаще всего становятся прибыльными? 

# In[143]:


answers['12'] = '1. Drama'
# +
# Фильмы какого жанра чаще становятся прибыльными
profitable_movies_genres_counter = get_counter_for(profitable_movies.genres_split)
profitable_movies_genres_counter.most_common()


# # 13. У какого режиссера самые большие суммарные кассовые сбооры?

# In[144]:


answers['13'] = '5. Peter Jackson'
# +

directors_grouped = data.groupby(['director']).sum()
most_profitable_director = directors_grouped[directors_grouped.profit == directors_grouped.profit.max()]
most_profitable_director


# # 14. Какой режисер снял больше всего фильмов в стиле Action?

# In[145]:


answers['14'] = '3. Robert Rodriguez'
# +

action_movies = data[data.genres.str.contains('Action')]
action_movies_directors_counter = get_counter_for(action_movies.directors)
action_movies_directors_counter.most_common()


# # 15. Фильмы с каким актером принесли самые высокие кассовые сборы в 2012 году? 

# In[146]:


answers['15'] = '3. Chris Hemsworth'
# +

def calculate_revenue_sum_for_actors_in(df):
    counter = Counter()
    for actors, revenue in zip(df['actors'], df['revenue']):
        for actor in actors:
            counter[actor] += revenue
    return counter

profitable_movies_in_2012 = profitable_movies.query('release_year == 2012')
actors_revenue_counter = calculate_revenue_sum_for_actors_in(profitable_movies_in_2012)
actors_revenue_counter.most_common()


# # 16. Какой актер снялся в большем количестве высокобюджетных фильмов?

# In[147]:


answers['16'] = '3. Matt Damon'
# + в данном вопросе сначала ответил неверно, поскольку посчитал что высокобюджетный фильм это фильм с прибылью > 0
# а не с бюджетом выше среднего

mean_budget = data.budget.mean()
high_budget_movies = data[data.budget > mean_budget]
actors_movies_counter = get_counter_for(high_budget_movies.actors)
actors_movies_counter.most_common()


# # 17. В фильмах какого жанра больше всего снимался Nicolas Cage? 

# In[148]:


answers['17'] = '2. Action'
# +
nicolas_cage_movies = data[data.cast.str.contains('Nicolas Cage')]
nicolas_cage_counter = get_counter_for(nicolas_cage_movies.genres_split)
nicolas_cage_counter.most_common()


# # 18. Самый убыточный фильм от Paramount Pictures

# In[149]:


answers['18'] = '1. K-19: The Widowmaker (tt0267626)'
# +
paramount_movies = data[data.production_companies.str.contains('Paramount Pictures')]
least_paramount_profit = paramount_movies.profit.min()
worst_paramount_movie = paramount_movies[paramount_movies.profit == least_paramount_profit]
worst_paramount_movie


# # 19. Какой год стал самым успешным по суммарным кассовым сборам?

# In[150]:


answers['19'] = '5. 2015'
# +
year_grouped = data.groupby(['release_year']).sum()
max_year_revenue = year_grouped.revenue.max()
best_year = year_grouped[year_grouped.revenue == max_year_revenue]
best_year


# # 20. Какой самый прибыльный год для студии Warner Bros?

# In[151]:


answers['20'] = '1. 2014'
# +
warner_movies = data[data.production_companies.str.contains('Warner Bros')]
warner_year_grouped = warner_movies.groupby(['release_year']).sum()
max_warner_revenue = warner_year_grouped.revenue.max()
best_warner_year = warner_year_grouped[warner_year_grouped.revenue == max_warner_revenue]
best_warner_year


# # 21. В каком месяце за все годы суммарно вышло больше всего фильмов?

# In[152]:


answers['21'] = '4. Сентябрь'
# +
month_grouped = data.groupby(['month']).count()
month_grouped.sort_values(by='imdb_id', ascending=False)


# # 22. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)

# In[153]:


answers['22'] = '2. 450'
# +
summer_movies = month_grouped.head(8).tail(3)
summer_movies.imdb_id.sum()


# # 23. Для какого режиссера зима – самое продуктивное время года? 

# In[154]:


answers['23'] = '5. Peter Jackson'
# +
winter_movies = data[data.month.isin([1, 2, 12])]
directors_counter = get_counter_for(winter_movies.directors)
directors_counter.most_common()


# # 24. Какая студия дает самые длинные названия своим фильмам по количеству символов?

# In[155]:


answers['24'] = '5. Four By Two Productions'
# +

avg_length_counter = calculate_average_length_for_companies_in(data)
avg_length_counter.most_common()


# # 25. Описание фильмов какой студии в среднем самые длинные по количеству слов?

# In[156]:


answers['25'] = '3. Midnight Picture Show'
# +
avg_words_counter = calculate_average_length_for_companies_in(data, column='overview', isCountingSymbols=False)
avg_words_counter.most_common()


# # 26. Какие фильмы входят в 1 процент лучших по рейтингу? 
# по vote_average

# In[157]:


answers['26'] = '1. Inside Out, The Dark Knight, 12 Years a Slave'
# +

sorted_by_ratings = data.sort_values(by='vote_average', ascending=False)
one_percent_count = int(len(data) / 100)
top_1_percent = sorted_by_ratings.head(one_percent_count)
top_1_percent


# # 27. Какие актеры чаще всего снимаются в одном фильме вместе?
# 

# In[158]:


answers['27'] = '5. Daniel Radcliffe & Rupert Grint'
# +

def get_actor_pairs_in(series):
    pairs = []
    # я после прочитал в слаке, что есть функция combinations, но я и подумать о таком не мог
    # следовательно, раз сам не стал искать, то оставил свою реализацию :D
    for actors in series.values:
        for actor in actors:
            index = actors.index(actor)
            # проходим до конца списка актеров, и создаем все возможные пары из списка
            while index + 1 < len(actors):
                pairs.append(actor + ' + ' + actors[index+1])
                index += 1
    return pairs

# Мне кажется это и не предполагается в задании, но вообще есть ситуация,
# когда в одном фильме актеры идут в порядке A, B, C
# следовательно имеем 3 пары актеров: [A, B], [A, C], [B, C].
# Но если в другом фильме актеры будут идти в порядке B, A, C, то пары будут [B, A], [B, C], [A, C]. 
# и поскольку мы имеем строки в качестве элементов массива, то строка АВ != BA
# получается что актеры одни и те же, но пара уже считается разной
# если есть какое-то красивое решение этой ситуации, то буду рад комментариям :)

pairs = get_actor_pairs_in(data.actors)
pairs_counter = Counter(pairs)
pairs_counter.most_common()


# ВАРИАНТ 2

# # Submission

# In[159]:


# в конце можно посмотреть свои ответы к каждому вопросу
answers


# In[160]:


# и убедиться что ни чего не пропустил)
len(answers)


# In[ ]:





# In[ ]:




