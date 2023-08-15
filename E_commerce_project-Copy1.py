#!/usr/bin/env python
# coding: utf-8

#                                                Подготовка данных для работы.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from datetime import date

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-ivanivani-petrov/olist_customers_dataset.csv')
orders = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-ivanivani-petrov/olist_orders_dataset.csv', 
        parse_dates = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                       'order_delivered_customer_date', 'order_estimated_delivery_date'])
order_items = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-ivanivani-petrov/olist_order_items_dataset.csv',
        parse_dates = ['shipping_limit_date'])


# Проверяем пропущенные значения.

# In[3]:


customers.isna().sum()


# Видим, что в датасете заказов очень много пропущенных дат. Часть связана с тем, что закзы не были доставлены, но есть и те, что вызывают вопросы: не заполнены, потому что это ошибка человека (лень, халатность, забывчивость) или это баг/ошибка данных со стороны приложения?

# In[4]:


orders.isna().sum()


# In[5]:


order_items.isna().sum()


# Проверяем название колонок и далее типы данных (адекватны ли тому, что есть в колонках).

# In[6]:


customers.head()


# *customer_id* — позаказный идентификатор пользователя
# 
# *customer_unique_id* —  уникальный идентификатор пользователя  (аналог номера паспорта)
# 
# *customer_zip_code_prefix* —  почтовый индекс пользователя
# 
# *customer_city* —  город доставки пользователя
# 
# *customer_state* —  штат доставки пользователя

# In[8]:


customers.dtypes


# In[9]:


orders.head()


# *order_id* —  уникальный идентификатор заказа (номер чека)
# 
# *customer_id* —  позаказный идентификатор пользователя
# 
# *order_status* —  статус заказа
# 
# *order_purchase_timestamp* —  время создания заказа
# 
# *order_approved_at* —  время подтверждения оплаты заказа
# 
# *order_delivered_carrier_date* —  время передачи заказа в логистическую службу
# 
# *order_delivered_customer_date* —  время доставки заказа
# 
# *order_estimated_delivery_date* —  обещанная дата доставки

# In[10]:


orders.dtypes


# In[11]:


order_items.head()


# *order_id* —  уникальный идентификатор заказа (номер чека)
# 
# *order_item_id* —  идентификатор товара внутри одного заказа
# 
# *product_id* —  ид товара (аналог штрихкода)
# 
# *seller_id* — ид производителя товара
# 
# *shipping_limit_date* —  максимальная дата доставки продавцом для передачи заказа партнеру по логистике
# 
# *price* —  цена за единицу товара
# 
# *freight_value* —  вес товара

# In[12]:


order_items.dtypes


# Колонки переименовывать не требуется, все данные в колонках в корректных форматах (все даты при чтении файлов были сразу приведены к формату datetime для дальнейшей работы).

# Далее проверим с помощью описательной статистики есть ли у нас какие-то аномалии и выбросы в колонки с ценой и со средним чеком. Описательные данные id товара или веса нам в данном случае не важны.
# По другим датасетам описательная статистика в таком варианте не несет полезной информации, поэтому ее исключил.

# In[38]:


order_items.describe()


# In[33]:


order_items.price.quantile(0.95)


# In[22]:


plt.figure(figsize=[16, 8])
ax = sns.histplot(order_items.price)


# In[23]:


avg_orders = order_items.groupby('order_id', as_index=False).agg({'price':'sum'})


# In[27]:


avg_orders.price.describe()


# In[37]:


avg_orders.price.quantile(0.95)


# In[28]:


plt.figure(figsize=[16, 8])
ax = sns.histplot(avg_orders.price)


# Мы видим, что при средней цене на товар в 120 у.е и 75-м процентиле 134 у.е., есть очень сильные выбросы по цене за товар.
# Та же ситуация со средним чеком: среднее 137.75 у.е., 75 процентиль - 149 у.е. Но при этом есть очень сильный выброс до 13 440 у.е. Проверка 95 процентиля показывает, что 95% средних чеков находятся ниже 400 рублей и 95% цен на товары находятся ниже 350 рублей. Что подтверждается графиком - выбросы есть, но их немного.
# Графики говорят нам о том, что цены на товары и средний чек имеют не нормальное, а экспоненциальное распределение. Это нужно учитывать при проведении статистического анализа (в данном проекте это не потребуется, но EDA все же провести необходимо).

# In[ ]:





#                     Задание № 1. Сколько у нас пользователей, которые совершили покупку только один раз?

# Объединяем 2 таблицы для расчетов (нам нужны заказы, чтобы дать ответ). Исключаем заказы, которые не стали покупкой:
# отменены, недоступны, пока только в статусе "подтвержден" и "выставлен счет", что фактом оплаты - то есть совершения покупки
# не является. Группируем заказы по пользователям, считая количество заказов у каждого уникального пользователя. 
# Переименовываем колонку для лучшего понимания ее значения (она не выводится на  печать, это для себя). Используем условие 
# "только 1 покупка" и считаем к-во строк.
# Вывод: подавляющее большинство клиентов (около 90%) сделали у нас только одну покупку. Причины установить сложно: нужно 
# понимать, что за товар продается.

# In[4]:


customer_orders = customers.merge(orders, on='customer_id') 
one_purchaise = customer_orders.query('order_status != "canceled" and order_status != "unavailable"                                       and order_status != "approved" and order_status != "invoiced"')     .groupby('customer_unique_id', as_index=False)     .agg({'order_id': 'count'})     .rename(columns={'order_id': 'purchaise_number'})     .query('purchaise_number == 1')     .shape
one_purchaise


#        Задание №2. Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам)? 

# Выделяем месяц создания заказа в отдельную колонку для удобства группировки. Выбираем заказы в статусе "отменен" или "недоступен".
# Группируем заказы по месяцам и статусам, считая количество.
# Группируем по статусам и считаем среднее к-во. Переименовываем колонку для лучшего понимания.
# Выводы из полученных данных можно сделать следующие: 
# 1. Большая часть недоставленных заказов - "отмененные", поэтому никто их не доставлял.
# 2. Часть данных по заказам утеряна, поврежедна или что-то еще (они в статусе "недоступен", что бы это не значило).

# In[25]:


customer_orders['year_month'] = customer_orders['order_purchase_timestamp'].dt.to_period("M")
not_delivered_orders = customer_orders.query('order_status == "canceled" or order_status == "unavailable"')     .groupby(['year_month', 'order_status'], as_index=False)     .agg({'order_id': 'count'})     .groupby('order_status', as_index=False)     .agg({'order_id':'mean'})     .rename(columns={'order_id': 'avg_canceled'})     .round(2)


# In[26]:


not_delivered_orders


#                     Задание №3. По каждому товару определить, в какой день недели товар чаще всего покупается.

# Объедининяем 2 таблицы для расчетов (нам нужны даты заказов, чтобы из них получить дни недели, а они в другой таблице).
# Достаем день недели из даты заказа и формируем новый столбец.
# Снова отсекаем заказы, которые не были оплачены, а значит товар не "покупался".
# Группируем данные по товару и дню недели, чтобы получить уникальные пары, далее получаем к-во этих уникальных пар в отдельном
# столбце и переименовываем для лучшего понимания.
# С помощью сортировки по товару и внутри товара по количеству получаем таблицу, где в первой строке напротив уникального товара
# будет максимальное число его покупок в конкретный день недели. Выбираем топ-1. Сортировка для наглядности.

# In[4]:


product_data = orders.merge(order_items, on='order_id')
product_data['weekday'] = product_data['order_purchase_timestamp'].dt.strftime('%A')
product_top_day = product_data.query('order_status != "canceled" and order_status != "unavailable"                                       and order_status != "approved" and order_status != "invoiced"')     .groupby(['product_id', 'weekday'], as_index=False)     .size()     .rename(columns={'size': 'quantity'})     .sort_values(['product_id', 'quantity'], ascending=False)     .groupby('product_id')     .head(1)     .sort_values('quantity', ascending=False)
product_top_day


#                     Задание №4. Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)?

# Снова отсекаем заказы, которые не были оплачены, а значит товар не "покупался".
# Группируем данные по месяцу и клиенту, считаем к-во покупок у клиента в этом месяце.
# Переименовываем колонку для лучшего понимания.
# Добавляем колонку "недели" с помощью lambda-функции (делим к-во дней в месяце на 7).
# Считаем среднее к-во покупок в неделю в месяце. Сортировка для наглядности.
# Данная метрика может быть полезна определенному бизнесу, но судить о чем-то конекретном без данных о товарах,
# самом бизнесе сложно. Также стоит учитывать, что подавляющее больштнство клиентов сделало всего одну покупку за все время.

# In[6]:


avg_purchaises = customer_orders.query('order_status != "canceled" and order_status != "unavailable"                                       and order_status != "approved" and order_status != "invoiced"')     .groupby(['year_month', 'customer_unique_id'], as_index=False)     .agg({'order_id': 'count'})     .rename(columns={'order_id':'quantity'})
avg_purchaises['weeks'] = avg_purchaises['year_month'].apply(lambda x: x.days_in_month / 7)
avg_purchaises['avg_purchases_per_week'] = avg_purchaises['quantity'] / avg_purchaises['weeks']
avg_purchaises.sort_values('avg_purchases_per_week', ascending=False)


#     Задание №5.Используя pandas, проведи когортный анализ пользователей. В период с января по декабрь выяви когорту с самым 
#                                                    высоким retention на 3й месяц.

# В данном случае за retention я взял количество пользователей в когорте, совершивших покупку повторно в течение 3 месяцев.
# Выбраны для расчета были только когорты c января 2017 года по июнь 2018 года, так как в 2016 году данные недостаточные,
# а в 2018 году данные есть только до августа.
# Выделяем месяц создания заказа в отдельную колонку для удобства группировки.
# Снова отсекаем заказы, которые не были оплачены, а значит товар не "покупался". Выделяем когорты (в данном случае выбрано 
# время певого заказа, переведенное в год и месяц для сортировки).
# Так как просто колонку не удалось добавить к изначальному датафрейму, пришлось объединять таблицы.
# Далее идет многократное повторение кода для разбивки на когорты с данными по трем месяцам, расчет процента). 
# Так как в таком варианте обнаружилась "грязь" в итоговых таблицах (начиная с июня "лезут" какие-то странные заказы, которые
# не должны относиться к этой когорте). Чтобы получить нужные три месяца, в ручном режиме меняется цифра в head(). Как это
# исправить, пока не ясно. Из-за этих "грязных" данных вычислять самый высокий показатель приходится вручную.
# После конкатенируем все когорты в одну таблицу, смотрим какие результаты получились. Видим, что самый высокий retention
# 3-го месяца в когорде мая 2017 года: 0.56%. Из всех данных по retention просматривается вывод (он был очвиден итак), что
# наш товар крайне редко покупают второй раз. Вероятно, его специфика такова, поэтому за второй и более покупкой приходит ничтожный процент покупателей.

# In[5]:


customer_orders['year_month'] = customer_orders['order_purchase_timestamp'].dt.to_period("M")
customer_orders_fom = customer_orders.query('order_status != "canceled" and order_status != "unavailable"                                       and order_status != "approved" and order_status != "invoiced"')     .groupby('customer_unique_id', as_index=False)     .agg({'order_purchase_timestamp': 'min'})     .rename(columns={'order_purchase_timestamp':'cohorts'})
customer_orders_fom.cohorts = customer_orders_fom.cohorts.dt.strftime('%Y-%m')
customer_data = customer_orders.merge(customer_orders_fom, on='customer_unique_id')
jan_cohort = customer_data.query('cohorts == "2017-01"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(3)
jan_cohort['percentage'] = (jan_cohort.customers_quantity / jan_cohort.customers_quantity.max() * 100).round(2)
feb_cohort = customer_data.query('cohorts == "2017-02"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(3)
feb_cohort['percentage'] = (feb_cohort.customers_quantity / feb_cohort.customers_quantity.max() * 100).round(2)
mar_cohort = customer_data.query('cohorts == "2017-03"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(3)
mar_cohort['percentage'] = (mar_cohort.customers_quantity / mar_cohort.customers_quantity.max() * 100).round(2)
apr_cohort = customer_data.query('cohorts == "2017-04"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(3)
apr_cohort['percentage'] = (apr_cohort.customers_quantity / apr_cohort.customers_quantity.max() * 100).round(2)
may_cohort = customer_data.query('cohorts == "2017-05"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(3)
may_cohort['percentage'] = (may_cohort.customers_quantity / may_cohort.customers_quantity.max() * 100).round(2)
jun_cohort = customer_data.query('cohorts == "2017-06"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(4)
jun_cohort['percentage'] = (jun_cohort.customers_quantity / jun_cohort.customers_quantity.max() * 100).round(2)
jul_cohort = customer_data.query('cohorts == "2017-07"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(5)
jul_cohort['percentage'] = (jul_cohort.customers_quantity / jul_cohort.customers_quantity.max() * 100).round(2)
aug_cohort = customer_data.query('cohorts == "2017-08"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(4)
aug_cohort['percentage'] = (aug_cohort.customers_quantity / aug_cohort.customers_quantity.max() * 100).round(2)
sep_cohort = customer_data.query('cohorts == "2017-09"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(5)
sep_cohort['percentage'] = (sep_cohort.customers_quantity / sep_cohort.customers_quantity.max() * 100).round(2)
oct_cohort = customer_data.query('cohorts == "2017-10"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(5)
oct_cohort['percentage'] = (oct_cohort.customers_quantity / oct_cohort.customers_quantity.max() * 100).round(2)
nov_cohort = customer_data.query('cohorts == "2017-11"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(4)
nov_cohort['percentage'] = (nov_cohort.customers_quantity / nov_cohort.customers_quantity.max() * 100).round(2)
dec_cohort = customer_data.query('cohorts == "2017-12"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(5)
dec_cohort['percentage'] = (dec_cohort.customers_quantity / dec_cohort.customers_quantity.max() * 100).round(2)
jan_2_cohort = customer_data.query('cohorts == "2018-01"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(4)
jan_2_cohort['percentage'] = (jan_2_cohort.customers_quantity / jan_2_cohort.customers_quantity.max() * 100).round(2)
feb_2_cohort = customer_data.query('cohorts == "2018-02"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(4)
feb_2_cohort['percentage'] = (feb_2_cohort.customers_quantity / feb_2_cohort.customers_quantity.max() * 100).round(2)
mar_2_cohort = customer_data.query('cohorts == "2018-03"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(6)
mar_2_cohort['percentage'] = (mar_2_cohort.customers_quantity / mar_2_cohort.customers_quantity.max() * 100).round(2)
apr_2_cohort = customer_data.query('cohorts == "2018-04"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(6)
apr_2_cohort['percentage'] = (apr_2_cohort.customers_quantity / apr_2_cohort.customers_quantity.max() * 100).round(2)
may_2_cohort = customer_data.query('cohorts == "2018-05"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(5)
may_2_cohort['percentage'] = (may_2_cohort.customers_quantity / may_2_cohort.customers_quantity.max() * 100).round(2)
jun_2_cohort = customer_data.query('cohorts == "2018-06"')     .groupby('year_month', as_index=False)     .agg({'customer_unique_id':'count'})     .rename(columns={'customer_unique_id':'customers_quantity'})     .sort_values('year_month')     .head(4)
jun_2_cohort['percentage'] = (jun_2_cohort.customers_quantity / jun_2_cohort.customers_quantity.max() * 100).round(2)
all_cohorts = pd.concat([jan_cohort, feb_cohort, mar_cohort, apr_cohort, may_cohort, jun_cohort,
                        jul_cohort, aug_cohort, sep_cohort, oct_cohort, nov_cohort, dec_cohort,
                        jan_2_cohort, feb_2_cohort, mar_2_cohort, apr_2_cohort, may_2_cohort, jun_2_cohort])
all_cohorts


#         Задание №6. Используя python, построй RFM-сегментацию пользователей, чтобы качественно оценить свою аудиторию.

# Объединяем три исходных датафрема (два были объединены ранее для расчетов) в новый для проведения расчетов.

# In[6]:


customer_orders_data = customer_orders.merge(order_items, on='order_id')


# Отсекаем заказы, которые не были оплачены. Группируем по времени заказа, номеру заказа и уникальному id пользователя, считаю
# сумму по данным заказам.

# In[38]:


orders = customer_orders_data.query('order_status != "canceled" and order_status != "unavailable"                                       and order_status != "approved" and order_status != "invoiced"')     .groupby(['order_id', 'order_purchase_timestamp', 'customer_unique_id'], as_index=False)     .agg({'price':'sum'})


# Выделяем текущую дату и период, который мы берем для расчетов (с 2016 года по 2023). Хотя это спорный момент, на мой взгляд,
# так как данные здесь уже стали историческими, зачем нам считать от текущего дня, а не от последнего дня покупки (как будто мы
# в том моменте находимся), непонятно. Но раз в задании сказано "до текущей даты", то так и делаем.

# In[37]:


NOW = date.today()
period = 365*7


# Переводим timedelta в date для расчетов. Создаем колонку "дней с последнего заказа" для расчета recency.

# In[45]:


orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp']).dt.date
orders['days_since_order'] = orders['order_purchase_timestamp'].apply(lambda x: (NOW - x).days)


# Честно берем код автора статьи из рекомендации без каких-либо изменений и преобразований, лишь заменив названия столбцов.
# В данной ячейке мы проходим циклом по колонке "дней с последнего заказа" для вычисления минимальной даты при наличии повторных
# заказов у данного id пользователя. Далее проходим циклом по колонке "дата заказа" для определения frequency (частоты или коли-
# чества заказов у данного клиента). Группируем все это по id пользователя и передаем как аргумент в агрегацию. После переименовы-
# выем колонки для лучшего понимания.

# In[223]:


aggr = {
    'days_since_order': lambda x: x.min(),  
    'order_purchase_timestamp': lambda x: len([d for d in x if d >= NOW - timedelta(days=period)])
}
rf = orders.groupby('customer_unique_id').agg(aggr).reset_index()
rf.rename(columns={'days_since_order': 'recency', 'order_purchase_timestamp': 'frequency'}, inplace=True)


# Так как сумма покупок за все время "отказалась" добавляться колонкой к готовому датафрейму, как у автора статьи из примера, то
# пришлось сделать отдельный расчет и после "склеить" датафреймы.

# In[224]:


monetary = orders.groupby('customer_unique_id', as_index=False)     .agg({'price':'sum'})     .rename(columns={'price':'monetary'})
rfm = rf.merge(monetary)


# Рассчитав все нужные данные, делим наши новые стобцы на кластеры. Разделение recency и monetary ведем с помощью квантилей.
# Кластер recency: 1 - от 2163 дней и больше, 2 - от 2048 до 2162 дней, 3 - от 1957 до 2047 дней, 4 - от 1872 до 1956 дней,
#                   5 - от 0 до 1871 дня
# Кластер monetary: 1 - от 0 до 39.9 у.е., 2 - от 40 до 69.9 у.е., 3 - от 70 до 109.9 у.е., 4 - от 110 до 179.9 у.е.,
#                    5 - от 180 и выше у.е.
# Кластер frequency квантилям не поддавался, так как огромное большинство заказчиков купило 1 раз. Поэтому поделил с помощью
# cut на 5 групп. Сразу же присваиваем номера групп в этом делении.
# Кластер frequency: 1 - от 1 до 4 заказов, 2 - от 5 до 7 заказов, 3 - от 8 до 10 заказов, 
#                    4 - от 11 до 13 заказов (в  эту группу никто не попал), 5 - от 14 до 16 заказов.

# In[225]:


quintiles = rfm[['recency', 'monetary']].quantile([.2, .4, .6, .8]).to_dict()
freq_group = pd.cut(rfm['frequency'], bins=5,  labels =[1, 2, 3, 4, 5])


# Присваиваем номера групп кластерам recency, monecy.

# In[226]:


def r_score(x):
    if x <= quintiles['recency'][.2]:
        return 5
    elif x <= quintiles['recency'][.4]:
        return 4
    elif x <= quintiles['recency'][.6]:
        return 3
    elif x <= quintiles['recency'][.8]:
        return 2
    else:
        return 1

def m_score(x):
    if x <= quintiles['monetary'][.2]:
        return 1
    elif x <= quintiles['monetary'][.4]:
        return 2
    elif x <= quintiles['monetary'][.6]:
        return 3
    elif x <= quintiles['monetary'][.8]:
        return 4
    else:
        return 5  


# Формируем новые колонки с присвоенными номерами.
# Примеры интепретации: сегмент 215 - последний заказ от 2048 до 2162 дней, количество заказов от 1 до 4 заказов включительно, 
#                                     сумма заказов от 180 и выше у.е.
#                        сегмент 414 - последний заказ от 1872 до 1956 дней, количество заказов от 1 до 4 заказов включительно, 
#                                     сумма заказов от 110 до 179.9 у.е.

# In[251]:


rfm['r'] = rfm['recency'].apply(lambda x: r_score(x))
rfm['f'] = freq_group
rfm['m'] = rfm['monetary'].apply(lambda x: m_score(x))


# Создаем колонку с названиями сегментов, чтобы разделить итоговые кластеры по группам. Приводим все колонки к строкам, чтобы
# далее заменить сочетания цифровых обозначений на названия группы. Создаем переменную, в которую передаем словарь с сочетаниями
# и их названиями. Первоначально сочетаний было больше (восемь), но три из них так не нашли совпадения в датафрейме, поэтому
# были удалены.

# In[250]:


rfm['segment'] = rfm['r'].astype(str) + rfm['f'].astype(str) + rfm['m'].astype(str)
segt_map = {
    r'[4-5][4-5][3-5]': 'the_best',
    r'[4-5][1-3][3-5]': 'valuable',
    r'[4-5][1-3][1-2]': 'almost_new_common',
    r'[1-3][1-3][3-5]': 'regular_common',
    r'[1-3][1-3][1-2]': 'elder_common'
}

rfm['segment'] = rfm['segment'].replace(segt_map, regex=True)
rfm.head()


# Считаем процентное соотношение получившихся групп.

# In[252]:


segment_percentage = rfm.segment.value_counts(normalize=True) * 100
segment_percentage.round(3)


# Визуализируем результат подсчета процентного соотношения сегментов. Что-то стало нагляднее, но понимание наличия вообще 
# кого-то "живого" из группы "the_best" на графике отсутствует. Слишком уж мизерная доля.

# In[263]:


segment_labels = ['regular_common', 'elder_common', 'valuable', 'almost_new_common', 'the_best']
plt.figure(figsize=(10, 10))
colors = sns.color_palette('pastel')[0:5]
plt.pie(segment_percentage, labels=segment_labels, colors=colors, autopct='%.0f%%')
plt.show()


#                                         По итогу RFM-исследования можно заключить следующее:
#     1. Около 35% наших покупателей относятся к сегменту regular_common: это пользователи, которые соверишили последнюю 
#        покупку от 2163 до 1957 дней назад, оформили от 1 до 10 заказов и принесли за все время от 70 до более чем 180 у.е.
#     2. Около 25% наших покупателей относятся к сегменту elder_common: это пользователи, которые соверишил последнюю покупку
#        от 2163 до 1957 дней назад, оформили от 1 до 10 заказов и принесли за все время от 0 до 69.9 у.е.
#     3. Около 24% наших покупателей относятся к сегменту valuable: это пользователи, которые соверишил последнюю покупку от
#        от 1956 до 1871 дня назад, оформили от 1 до 10 заказов и принесли за все время от 70 до более чем 180 у.е.
#     4. Около 16% наших покупателей относятся к сегменту almost_new_common: это пользователи, которые соверишил последнюю 
#        покупку от 1956 до 1871 дня назад, оформили от 1 до 10 заказов и принесли за все время от от 0 до 69.9 у.е.
#     5. Лишь одна тысячная процента относится к сегменту the best: это пользователи, которые соверишил последнюю покупку                от 1956 до 1871 дня назад, оформили от 11 до 16 заказов и принесли за все время от 70 до более чем 180 у.е.
#     В целом, данное деление не совсем показательно, так как RFM анализ не слишком хорошо работает там, где подавляющее боль-
#     шинство клиентов совершили 1 заказ. Но если таков продукт, то и его можно пытаться анализировать даже таким методом.

# In[ ]:




