#!/usr/bin/env python
# coding: utf-8

# In[565]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind

pd.set_option('display.max_rows', 50) # показывать больше строк
pd.set_option('display.max_columns', 50) # показывать больше колонок

data = pd.read_csv('module_2.csv')

# просмотрим базовую информацию о датасете
data.info()

## Вывод:
# В датасете 17 строковых столбцов, 16 столбцов типа float, 1 столбец типа int
# В датасете содержится 395 строк (записей)
# Часть данных отсутсвует

## Вспомогательные функции

def values_outside_iqr_for(series):
    median = series.median()
    print('Медиана ' + series.name + ' = ' + str(median))
    iqr = series.quantile(0.75) - series.quantile(0.25)
    print('IQR ' + series.name + ' = ' + str(iqr))
    lower_bound = median - (1.5 * iqr)
    upper_bound = median + (1.5 * iqr)
    print('Нижняя граница выброса ' + series.name + ' = ' + str(lower_bound))
    print('Верхняя граница выброса ' + series.name + ' = ' + str(upper_bound))
    return series.between(lower_bound, upper_bound)
    
def display_normalized_values_for(series):
    print(data[series.name].value_counts(dropna=False, normalize=True).head(10))
    
def display_absent_values_for(series):
    display(data.loc[data[series.name].isna()])
    
def replace_yes_no_with_integers_in(series):
    data[series.name] = data[series.name].apply(lambda x: None if x == None else 1 if x == 'yes' else 0)
    
def replace_empty_string_values_in(series):
    series.apply(lambda x: None if x.strip() == '' else x.strip())

## переименуем некоторые столбцы для большего удобства использования
data.rename(columns={'famsize': 'family_size',
                     'Pstatus': 'parents_status',
                     'Medu': 'm_education',
                     'Fedu': 'f_education',
                     'Mjob': 'm_job',
                     'Fjob': 'f_job',
                     'schoolsup': 'school_support',
                     'famsup': 'family_support',
                     'famrel': 'family_relations',
                     'studytime, granular': 'studytime_granular',
                     'goout': 'go_out'}, inplace=True)


# In[566]:


## Разберем основные числовые столбцы


# In[567]:


# age
data.age.hist()
print(data.age.describe())

# видим что есть ученики старше 20 лет, сложно представить школьника старше 20 лет, к тому же
# таковые значения являются выбросами, следовательно отбросим их
data = data.loc[values_outside_iqr_for(data.age)]


# In[568]:


# m_education
data.m_education.hist(bins=4)
print(data.m_education.describe())
#видим что большинство матерей имеют высшее либо среднее-специальное образование, часть данных отсутсвует
#рассмотрим отсутствующие данные
display_absent_values_for(data.m_education)

#строка с индексом 239 содержит большое количество пропусков (m/f_education, internet, romantic)
#к тому же значение score = 0. Отбросим ее
data = data.drop(index=239)

#в остальных случаях заменим пропуски на образование, соответствующее образованию отца
data.m_education = np.where(data.m_education.isna(), data['f_education'], data['m_education'])


# In[569]:


# f_education
data.f_education.hist(bins=4)
print(data.f_education.describe())
# Видим в этом столбце выброс в виде значения 40. 
# предположим что это ошибочное значение, правильное значение должно быть 4 (высшее образование)
data.loc[(data['f_education'] == 40), 'f_education'] = 4

#рассмотрим отсутсвующие данные
display_absent_values_for(data.f_education)

#для отсутствующих значений используем тот же прием, что и с образованием матери
#т.e. предположим что образование отца соответствует образованию матери
data.f_education = np.where(data.f_education.isna(), data['m_education'], data['f_education'])


# In[570]:


# traveltime
print(data.traveltime.describe())
display_normalized_values_for(data.traveltime)

# видим что часть данных отсутсвует, рассмотрим отстутвующие данные
display_absent_values_for(data.traveltime)

# заменим отсутсвующие данные в зависимости от типа места жительства
# очевидно, что живущие в городе тратят меньше времени на путь в школу, чем живущие в деревне
# найдем медианные значения для каждого из типов места жительства, и заменим отсутсвующие значения
data.groupby(by=['address'])['traveltime'].median()

# видим что для деревни traveltime составляет 2, для города 1
# если тип адреса так же неизвестен, то опустим такую строку
data = data.dropna(subset=['address', 'traveltime'], how='all')
data.loc[(data['address'] == 'U') & (data['traveltime'].isna()), 'traveltime'] = 1
data.loc[(data['address'] == 'R') & (data['traveltime'].isna()), 'traveltime'] = 2


# In[571]:


# studytime

print(data.studytime.describe())
data.studytime.hist()
# видим что подавляющее большинство учеников занимаются дополнительно 2-5 часов (2)
# часть данных отсутсвует. рассмотрим отсутсвующие данные

display_absent_values_for(data.studytime)

# считаю что данные о дополнитльном обучении достаточно важная информация, 
# которая оказывает существенное влияние на итоговый результат, поэтому замена
# отсутсвующих данных может существенно исказать результат,
# в связи с этим отбросим записи с отсутсвующими данными,
# к тому же, в каждой из данных строк отсутсвует большое количество информации и 
# по остальным признакам
data = data.dropna(subset=['studytime'], how='all')


# In[572]:


# failures
data.failures.hist()
display_normalized_values_for(data.failures)

# видим что 75% учеников не испытвали внешкольных неудач. 5% данных отсутсвует
# достаточно сложно интерпретировать данные из этого столбца,
# поэтому предлагаю заменить отсутсвующие данные на значение моды (0)
data.failures = data.failures.fillna(0)


# In[573]:


# studytime_granular
data.studytime_granular.hist()
display_normalized_values_for(data.studytime_granular)

# если я правильно понимаю, то этот столбец содержит значение часов из столбцa studytime
# мы уже обработали столбец studytime, по сути эти данные просто дублируются
# (к тому же, почему-то со значением -), поэтому удалим этот столбец

data.drop(['studytime_granular'], inplace=True, axis=1)


# In[574]:


# family_relations
print(data.family_relations.describe())
display_normalized_values_for(data.family_relations)

# видим отрицательное значение -1, предположим что значение должно быть положительным и исправим это
data.loc[(data['family_relations'] == -1), 'family_relations'] = 1

# видим что отсутсвует около 7% данных, что довольно много, рассмотрим их
display_absent_values_for(data.family_relations)

# предлагаю удалить строки, в которых нет данных о семейных отношениях
# и семейной помощи в обучении (family_support), поскольку это достаточно важная информация
# которая оказывает существенное влияние
data = data.dropna(subset=['family_relations', 'family_support'], how='all')

# рассмотрим зависимость семейных отношений и совместного/раздельного проживания родителей
sns.boxplot(x ='parents_status', y = 'family_relations', data = data)

# очевидно, что в семьях где родители живут раздельно, отношения чуть хуже
# однако медианное значение для таких семей = 4.
# поэтому предлагаю заменить отсутсвующие значения в family_relations
# в семьях где родители живут вместе на 5, а где раздельно - на 4
data.loc[(data['parents_status'] == 'A') & (data['family_relations'].isna()), 'family_relations'] = 4
data.loc[(data['parents_status'] == 'T') & (data['family_relations'].isna()), 'family_relations'] = 5


# In[575]:


# freetime
print(data.freetime.describe())
display_normalized_values_for(data.freetime)

# видим что отсутствует небольшая часть значений (2.5%), рассмотрим их
display_absent_values_for(data.freetime)

# полагаю что будет логичным заменить отсутсвующие значения в данном случае
# на значения из столбца go_out. поскольку очевидно что если ты проводишь много времени с друзьями
# то у тебя много свободного времени, и наоборот :)
data.freetime = np.where(data.freetime.isna(), data['go_out'], data['freetime'])


# In[576]:


# go_out
print(data.go_out.describe())
display_normalized_values_for(data.go_out)

# видим что отсутствует небольшая часть значений (1.8%), рассмотрим их
display_absent_values_for(data.go_out)

# поступим аналогично предыдущему случаю, и подставим вместо отсутсвующих значений
# значения из столбца freetime
data.go_out = np.where(data.go_out.isna(), data['freetime'], data['go_out'])


# In[577]:


# health
print(data.health.describe())
display_normalized_values_for(data.health)
data.health.hist()

# видим что бОльшая часть студентов чувствуют себя замечательно
# и можем лишь посочувствовать 22% у которых не все хорошо со здоровьем
# пока нет понимания как можно заменить отсутсвующие значения,
# возможно следовало бы удалить строки с отсутвующими данными по здоровью,
# поскольку это весьма важный фактор, который нельзя не учитывать,
# однако пока предлагаю оставить как есть, возможно в дальнейшем мы как-либо обработаем эти данные


# In[578]:


# absences
print(data.absences.describe())
display_normalized_values_for(data.absences)
data.absences.hist()

# очевидны проблемы с этим столбцом, сразу же явно видны выбросы
# (ну либо они просто перестали ходить в школу :) )
# поэтому избавимся от выбросов
data = data.loc[values_outside_iqr_for(data.absences)]

display_absent_values_for(data.absences)

#отсутвующих значений нет


# In[579]:


# score
print(data.score.describe())
display_normalized_values_for(data.score)
data.score.hist()

# явно видно нормальное распределение оценок, при этом имеются
# выбросы/отсутсвующие данные в виде оценок 0, избавимся от них
data = data.loc[values_outside_iqr_for(data.score)]

# в результате этого мы так же избавились от результата 100
# (верхняя граница 98.125), но предлагаю не рассматривать вундеркиндов,
# а исследовать обычных студентов/школьников :)


# In[580]:


## подведем итоги обработки числовых столбцов
data.info()

# видим что в результате обработки всех числовых столбцов мы исключили ряд записей
# по разным причинам. Осталось 302 записи


# In[581]:


## Обработка строковых столбцов


# In[582]:


# school
display(data.school.value_counts())
display_normalized_values_for(data.school)

# видим, что 88% учеников обучаются в школе GP (что бы это ни значило). отсутсвующих/пустых значений нет


# In[583]:


# sex

display(data.sex.value_counts())
display_normalized_values_for(data.sex)

# видим что соотношение мальчиков и девочек примерно равное. отсутсвующих/пустых значений нет


# In[584]:


# address
display(data.address.value_counts())
display_normalized_values_for(data.address)

# видим что соотношение городских учеников к сельским примерно 3 к 1. Есть 3% пропущенных значений. 
# часть пропусков нами уже была обработана при анализе столбцa traveltime.
# рассмотрим оставшиеся пропуски

display_absent_values_for(data.address)

# для обработки пропусков в данном столбце используем тот же прием, что и при заполнении пропусков
# в столбце traveltime (1 - для городского жителя, 2 - для сельского)
data.loc[(data['traveltime'] == 1) & (data['address'].isna()), 'address'] = 'U'
data.loc[(data['traveltime'] == 2) & (data['address'].isna()), 'address'] = 'R'


# In[585]:


# family size
display(data.family_size.value_counts())
display_normalized_values_for(data.family_size)

# видим что 65% учеников проживают в больших семьях. 5% данных отсутсвуют
# рассмотрим оставшиеся пропуски

display_absent_values_for(data.family_size)

# учитывая что большинство учеников из больших семей, предлагаю заменить отсутсвующие значения 
# на GT3, если родители живут вместе (parents_status == T), и на LE3, если раздельно
# если данные о родителях так же отсутсвуют - удалим такие строки
data = data.dropna(subset=['parents_status', 'family_size'], how='all')
data.loc[(data['parents_status'] == 'A') & (data['family_size'].isna()), 'family_size'] = 'LE3'
data.loc[(data['parents_status'] == 'T') & (data['family_size'].isna()), 'family_size'] = 'GT3'


# In[586]:


# parents_status
display(data.parents_status.value_counts())
display_normalized_values_for(data.parents_status)

# видим что в 80% случаев родители живут вместе. 9% данных отсутсвует, 9% живут раздельно
# рассмотрим оставшиеся пропуски

display_absent_values_for(data.parents_status)

# можно было бы следовать логике, обратной логике с размером семьи, но поскольку
# подавляющее большинство родителей живут вместе, заменим пропуски на значение моды
data.parents_status = data.parents_status.fillna(data.parents_status.mode()[0])


# In[587]:


# m_job
display_normalized_values_for(data.m_job)
data.m_job.hist()

# видим что 4% информации отсутсвует, рассмотрим отсутвующие данные
display_absent_values_for(data.m_job)

# думаю, наиболее правильным вариантом в данном случае будет заменить пропуски в работе матери
# на работу отца. не думаю, что эти значения окажут какое-то сильное влияние на конечный результат.
# в случае отсутсвия обоих данных - удалим такие строки
data = data.dropna(subset=['m_job', 'f_job'], how='all')
data.m_job = np.where(data.m_job.isna(), data['f_job'], data['m_job'])


# In[588]:


# f_job
display_normalized_values_for(data.f_job)
data.f_job.hist()

# видим что 8% информации отсутсвует, рассмотрим отсутвующие данные
display_absent_values_for(data.f_job)

# у меня появилась другая идея, привязать работу родителя к его уровню образования.
# оставим замену для работы матери как есть, поскольку там пропущенных значений в 2 раза меньше
# чем для работы отца, а для работы отца применим подход где
# уровень образования 4 - health, 3 - teacher, 2 - services, 1 - at_home

values_dictionary = {
    4: 'health',
    3: 'teacher',
    2: 'services',
    1: 'at_home'
}
for key, value in values_dictionary.items():
    data.loc[(data['f_education'] == key) & (data['f_job'].isna()), 'f_job'] = value


# In[589]:


# reason
display(data.reason.value_counts())
display_normalized_values_for(data.reason)

# видим что 4% информации отсутсвует, рассмотрим отсутвующие данные
display_absent_values_for(data.reason)

# сложно делать какие-либо выводы об этом столбце, да и не думаю что он имеет отношение
# к оценкам. возможно имеет отношение пункт 'course', то есть образовательная программа,
# но это не значит что в других школах она хуже, просто кто-то руководствуется
# другими причинами при выборе школы. 
# позволю себе удалить данный столбец как нерелевантный
data.drop(['reason'], inplace=True, axis=1)


# In[590]:


# guardian
display(data.guardian.value_counts())
display_normalized_values_for(data.guardian)

# данный столбец мне не совсем понятен. что значит опекун мать/отец
# если семья полная, то очевидно что оба родителя выполняют свои функции, а не кто-то один
# однако все же рассмотрим отсутсвующие значения, возможно найдем способ для замены данных
display_absent_values_for(data.guardian)

# сложно предположить какие-либо варианты для замены. удалим этот столбец
data.drop(['guardian'], inplace=True, axis=1)


# In[591]:


# school_support
display(data.school_support.value_counts())
display_normalized_values_for(data.school_support)

#видим что в большинстве случаев дополнительной поддержки нет. при этом небольшая часть данных (2.5%) пропущена
# рассмотрим отсутсвующие данные

display_absent_values_for(data.school_support)

#  возможно, наличие поддержки связано с типом школы. проверим эту гипотезу
data.groupby(by=['school_support'])['school'].hist()

# видим что поддержка есть только в школе GP, однако в связи с малочисленностью 
# данных о наличии поддержки мы не можем однозначно сказать что все остальные данные можно 
# заменить на наличие поддержки. Поскольку в школе MS данных о поддержке нет
# то в ней мы заменим отсутвующее значение на 'no'
data.loc[(data['school'] == 'MS') & (data['school_support'].isna()), 'school_support'] = 'no'

# заменим yes/no на 1/0 для учета этих данных в таблице корреляции
replace_yes_no_with_integers_in(data.school_support)


# In[592]:


# family_support
display(data.family_support.value_counts())
display_normalized_values_for(data.family_support)

# видим что в отличие от школьной поддержки, более половины учеников занимаются в семье
# 8% данных пропущено, рассмотрим их
display_absent_values_for(data.family_support)

# на данный момент не представляется возможным как-либо заполнить пропуски

# заменим yes/no на 1/0 для учета этих данных в таблице корреляции
replace_yes_no_with_integers_in(data.family_support)


# In[593]:


# paid
display(data.paid.value_counts())
display_normalized_values_for(data.paid)

# видим что распределение учеников с платными занятиями и без них приблизительно равное, при этом 11% данных пропущено
display_absent_values_for(data.paid)

# на данный момент не представляется возможным как-либо заполнить пропуски
# заменим yes/no на 1/0 для учета этих данных в таблице корреляции
replace_yes_no_with_integers_in(data.paid)


# In[594]:


# activities
display(data.activities.value_counts())
display_normalized_values_for(data.activities)

# видим что распределение учеников с доп занятиями и без них приблизительно равное, при этом 4% данных пропущено
display_absent_values_for(data.activities)

# на данный момент не представляется возможным как-либо заполнить пропуски
# заменим yes/no на 1/0 для учета этих данных в таблице корреляции
replace_yes_no_with_integers_in(data.activities)


# In[595]:


# nursery
display(data.nursery.value_counts())
display_normalized_values_for(data.nursery)

# видим что 77% посещали детский сад, 4% данных пропущено.
display_absent_values_for(data.nursery)

# на данный момент не представляется возможным как-либо заполнить пропуски
# заменим yes/no на 1/0 для учета этих данных в таблице корреляции
replace_yes_no_with_integers_in(data.nursery)


# In[596]:


# higher
display(data.higher.value_counts())
display_normalized_values_for(data.higher)

# видим что 92% хотят получать высшее образование (что радует). отсутвует 2.5% данных, рассмотрим их
display_absent_values_for(data.higher)

# думаю, при таком подавляющем большинстве желающих получать высшее образование, можно заменить отсутсвующие
# данные на большинство
data.higher = data.higher.fillna('yes')

# заменим yes/no на 1/0 для учета этих данных в таблице корреляции
replace_yes_no_with_integers_in(data.higher)


# In[597]:


# internet
display(data.internet.value_counts())
display_normalized_values_for(data.internet)

# видим что 76.5% имеют доступ в интернет, 14.7% - нет. отсутвует 8% данных, рассмотрим их
display_absent_values_for(data.internet)

# хотелось бы заполнить все пропуски доступом в интернет, но я не Илон Маск чтобы дарить
# людям бесплатный интернет :)

# заменим yes/no на 1/0 для учета этих данных в таблице корреляции
replace_yes_no_with_integers_in(data.internet)


# In[598]:


# romantic
display(data.romantic.value_counts())
display_normalized_values_for(data.romantic)

# видим что 63% не состоят в отношениях. 8% данных отсутсвует, рассмотрим их

# на данный момент не представляется возможным как-либо заполнить пропуски
# заменим yes/no на 1/0 для учета этих данных в таблице корреляции
replace_yes_no_with_integers_in(data.romantic)


# In[599]:


## данные после обработки всех столбцов 
data.info()

# осталось 298 записей, часть столбцов была удалена


# In[600]:


# рассмотрим таблицу корреляции
data.corr()


# In[601]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize = (14, 4))
    sns.boxplot(x=column, y='score', 
                data=data.loc[data.loc[:, column].isin(data.loc[:, column].value_counts().index[:10])],
               ax=ax)
    ax.set_title('Boxplot for ' + column)
    plt.show()
    
for col in data.columns:
    get_boxplot(col)

