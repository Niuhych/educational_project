#!/usr/bin/env python
# coding: utf-8

# Ипортируем нужные библиотеки, читаем данные, смотрим типы, сразу парсим даты как даты на случай использования в расчетах.

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import chi2_contingency
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[99]:


ab_users_data_all = pd.read_csv('C:/Users/Niuh/Courses/ab_users_data.csv', parse_dates=['time', 'date'])
ab_orders = pd.read_csv('C:/Users/Niuh/Courses/ab_orders.csv', parse_dates=['creation_time'])
ab_products = pd.read_csv('C:/Users/Niuh/Courses/ab_products.csv')


# In[100]:


ab_users_data_all.head()


# In[101]:


ab_users_data_all.shape


# In[102]:


ab_users_data_all.dtypes


# Убираем отмененные заказы для дальнейших расчетов.

# In[103]:


ab_users_data = ab_users_data_all.query('action == "create_order" and action != "cancel_order"')
ab_users_data.shape


# In[5]:


ab_orders.head()


# In[6]:


ab_orders.dtypes


# Убираем фигурные скобки, трансформируем все в список, удаляем "лишнюю" колонку и разворачиваем список по отдельному product_id для будущего merge.

# In[34]:


ab_orders['product_id'] = ab_orders['product_ids'].apply(lambda x: x.replace('{', '').replace('}', '').split(', '))


# In[35]:


ab_orders = ab_orders.drop(columns='product_ids', axis=1)


# In[36]:


ab_orders = ab_orders.explode('product_id')
ab_orders['product_id'].astype('Int64')
ab_orders.head()


# In[37]:


ab_uo = ab_users_data.merge(ab_orders, on='order_id')


# In[38]:


ab_uo.head()


# In[39]:


ab_uo['product_id'] = ab_uo['product_id'].astype('Int64')


# In[40]:


ab_products.head()


# In[14]:


ab_products.dtypes


# In[41]:


all_data = ab_uo.merge(ab_products, on='product_id')


# In[42]:


all_data.head()


# In[43]:


all_data.action.value_counts()


# Готовим данные, оставляем нужные колонки для расчетов метрик.

# In[44]:


research_data = all_data.groupby(['user_id', 'group', 'order_id', 'date'], as_index=False) \
    .agg({'product_id':'count', 'price':'sum'}) \
    .rename(columns={'product_id':'number_of_products', 'price':'sum_price'})


# In[45]:


research_data.head()


# Проверка наших групп. По числу уникальных пользователей примерно одинаковы. Будем считать, что система сплитования была корректной, выборки репрезентативны.

# In[46]:


research_data.groupby('group', as_index=False).agg({'user_id':'nunique'})


# Первоначально с помощью обычных группировок и агрегаций вглянем на показатели контрольной и экпериментальной группы. Видно, что общая выручка увеличилась, но увеличилось также и кол-во заказов на одного юзера. А вот средний чек в экспериментальной группе даже слегка уменьшился (далее проверим насколько это изменение статистически значимое). Также не изменилось и среднее к-во товаров в одном заказе.

# In[47]:


data_money = research_data.groupby('group', as_index=False).agg({'sum_price':'sum'})
data_money


# In[48]:


929232.0*100/613488.8 - 100


# In[49]:


research_data.groupby(['group'], as_index=False).agg({'sum_price':'mean'}).round(2)


# In[50]:


data_orders = research_data.groupby('group', as_index=False).agg({'order_id':'nunique'}) \
    .rename(columns={'order_id':'orders'})
data_orders


# Ниже на графике наглядно показан рост количества заказов в течение выбранного периода

# In[51]:


plt.xticks(rotation=45)
data_orders_0 = research_data.query('group == 0').groupby('date').agg({'order_id':'nunique'}).cumsum()
data_orders_1 = research_data.query('group == 1').groupby('date').agg({'order_id':'nunique'}).cumsum()
sns.lineplot(data=data_orders_0, x="date", y="order_id")
sns.lineplot(data=data_orders_1, x="date", y="order_id")
plt.title('Number of orders')
plt.ylabel('orders')


# In[52]:


sns.histplot(data_orders_0)


# In[53]:


data_avg_products = research_data.groupby('group', as_index=False).agg({'number_of_products':'mean'}).round(2)
data_avg_products


# Проводим исследование среднего чека.
# 1. Проверка на гомогенность дисперсий пройдена.
# 2. Распределение же не является нормальным, но и слишком необыного чего-то в нем нет, сильных выбросов тоже.
# 3. После логарифмирования ситуация та же, поэтому использовать логарифмированные данные не будем (они порою усложняют интерпретацию результата только).

# In[54]:


pg.homoscedasticity(data=research_data, dv='sum_price', group='group')


# In[55]:


pg.normality(data=research_data, dv="sum_price", group="group", method="normaltest")


# In[56]:


research_data['sum_price_log'] = np.log(research_data.sum_price)


# In[57]:


pg.normality(data=research_data, dv="sum_price_log", group="group", method="normaltest")


# In[58]:


research_data.query('group == 0').groupby('user_id').agg({'sum_price':'sum'}).hist()


# In[59]:


research_data.query('group == 1').groupby('user_id').agg({'sum_price':'sum'}).hist()


# In[60]:


research_data.query('group == 0').groupby('user_id').agg({'sum_price_log':'sum'}).hist()


# In[61]:


research_data.query('group == 1').groupby('user_id').agg({'sum_price_log':'sum'}).hist()


# Нулевая гипотеза теста: средний чек в тестовой и экспериментальной группе не различаются.
# Альтернативная гипотеза: средний чек в группах статистически значимо различаются.
# 1. Хотя распределение не является нормальным, но данных довольно большое к-во, проверил на графиках насколько сильно различаются внешне распределения. В целом, они удовлетворяют при такой выборке возможность использования t-теста. Тест не показал статистически значимое различие.
# 2. Так как до проведения теста нам не удалось определиться с "коллегами", какой тип теста мы применим, то была проведена проверка с помощью теста Манна-Уитни (средние ранги групп не различаются), который не дал статистически значимого различия.
# 3. Так как тесты показали разные данные, провел еще и bootstrap тест на сравнение средних в группах, который показал, что доверительные интервалы пересекаются, то есть статистически значимых различий нет.
# Итог: ни один из тестов не показал статистически значимого результата, мы не можем отклонить нулевую гипотезу.

# In[65]:


ss.ttest_ind(group_0, group_1)


# In[66]:


group_0 = research_data.query('group == 0')['sum_price']
group_1 = research_data.query('group == 1')['sum_price']
ss.mannwhitneyu(group_0, group_1)


# In[67]:


pg.compute_bootci(group_0, func='mean')


# In[68]:


pg.compute_bootci(group_1, func='mean')


# Следующей метрикой выбрано среднее количество товаров в заказе.
# Ситуация та же, что и со средним чеком: дисперсии гомогенны, но распределение ненормальное, хотя и без критичных выбросов.
# После логарифмирования существенно ничего не изменилось. Как и в первом случае, не будем использовать эти данные, чтобы не было проблем с интерпретацией результата.

# In[69]:


pg.homoscedasticity(data=research_data, dv='number_of_products', group='group')


# In[70]:


pg.normality(data=research_data, dv="number_of_products", group="group", method="normaltest")


# In[49]:


research_data.query('group == 0').groupby('user_id').agg({'number_of_products':'sum'}).hist()


# In[71]:


research_data.query('group == 1').groupby('user_id').agg({'number_of_products':'sum'}).hist()


# In[72]:


research_data['number_of_products_log'] = np.log(research_data.number_of_products)


# In[74]:


research_data.query('group == 0').groupby('user_id').agg({'number_of_products_log':'sum'}).hist()


# In[73]:


research_data.query('group == 1').groupby('user_id').agg({'number_of_products_log':'sum'}).hist()


# Нулевая гипотеза теста: среднее количество товаров в одном заказе не отличается.
# Альтернативная гипотеза: среднее к-во товаров статистически значимо отличается.
# T-тест, тест Манна-Уитни и bootstrap показали, что статистически значимого результата нет, поэтому мы не можем отклонить нулевую гипотезу. Как и в первой метрике, t-тест был использован потому, что выборка достаточно большая, а в распределениях нет ничего критического.

# In[74]:


ss.ttest_ind(group_0, group_1)


# In[75]:


group_0 = research_data.query('group == 0')['number_of_products']
group_1 = research_data.query('group == 1')['number_of_products']

ss.mannwhitneyu(group_0, group_1)


# In[76]:


pg.compute_bootci(group_0, func='mean')


# In[77]:


pg.compute_bootci(group_1, func='mean')


# Подготовил данные для следующей метрики - количество заказов в группах.

# In[78]:


research_data_noford = all_data.groupby(['user_id', 'group'], as_index=False) \
    .agg({'order_id':'nunique'}) \
    .rename(columns={'order_id':'orders'})


# In[77]:


research_data_noford.head()


# In[79]:


research_data_noford.query('group == 0').groupby('user_id').agg({'orders':'sum'}).hist()


# In[80]:


research_data_noford.query('group == 1').groupby('user_id').agg({'orders':'sum'}).hist()


# In[81]:


research_data_noford['orders_log'] = np.log(research_data_noford.orders)


# In[82]:


research_data_noford.query('group == 0').groupby('user_id').agg({'orders_log':'sum'}).hist()


# In[83]:


research_data_noford.query('group == 1').groupby('user_id').agg({'orders':'sum'}).hist()


# Здесь уже ни проверку на гомогенность дисперсий, ни проверку на нормальность распределения выборки не проходят.
# Логарифмирование существенно ухудшает графики, поэтому от него отказываюсь сразу.

# In[84]:


pg.normality(data=research_data_noford, dv="orders_log", group="group", method="normaltest")


# In[85]:


pg.normality(data=research_data_noford, dv="orders_log", group="group", method="normaltest")


# In[86]:


pg.homoscedasticity(data=research_data_noford, dv='orders_log', group='group')


# In[87]:


pg.homoscedasticity(data=research_data_noford, dv='orders', group='group')


# Нулевая гипотеза теста: среднее к-во заказов на пользователя в группах не различаются.
# Альтернативная гипотеза: среднее к-во заказов статистически значимо различается.
# Так как данных достаточно, то также пробуем применить t-тест, но уже с поправкой на негомогенность дисперсий. Здесь также полное согласие у t-теста Манна-Уитни и bootstrap: t-тест и МУ показывают очень низкий p-value, bootstrap указыват, что доверительные интервалы различия средних в группах очень далеки друг от друга. Что позволяет нам отклонить нулевую гипотезу и принять альтернативную. Среднее количество заказов на пользователя действительно значительно увеличилось даже в абсолютных цифрах, чтобы видно еще при первоначальной группировке данных: к-во пользователей в группах у нас примерно одинаковое, а к-во заказов на 900 шт больше (примерно на 56%).

# In[88]:


ss.ttest_ind(group_0, group_1, equal_var=False)


# In[89]:


group_0 = research_data_noford.query('group == 0')['orders']
group_1 = research_data_noford.query('group == 1')['orders']

ss.mannwhitneyu(group_0, group_1)


# In[90]:


pg.compute_bootci(group_0, func='mean')


# In[91]:


pg.compute_bootci(group_1, func='mean')


# Подготовил данные для еще одной метрики - количество отмененных заказов по отношению к созданным заказам.

# Здесь видится наиболее удачным вариантом расчета Хи-квадрат. Составить таблицу сопряжения по группам и по состоянию заказов.
# Нулевая гипотеза теста: среднее к-во отмененных заказов по отношению к созданным не изменилось.
# Альтернативная гипотеза: среднее к-во отмененнх заказов изменилось (мы ждем, что уменьшилось, но это наше желание, а тест покажет, что есть на самом деле).
# P-value очень высокое, что совсем не позволяет нам отклонить нулевую гипотезу. Итог: количество отмененных заказов по отношению к созданным никак не изменилось с введением нового алгоритма. Что можно также отследить и с помощью простого расчета процентного соотношения.

# In[92]:


82*100/1609


# In[93]:


132*100/2514


# In[104]:


pd.crosstab(ab_users_data_all.group, ab_users_data_all.action)


# In[105]:


stat, p, dof, expected = chi2_contingency(pd.crosstab(ab_users_data_all.group, ab_users_data_all.action))


# In[106]:


stat, p


# # Выводы

# После проведенного исследования можно заключить, что большая часть ключевых метрик (средний чек, среднее к-во товаров в заказе, количество отмененных заказов по отношению к созданным) никак не поменялись. В лучшую сторону ушла лишь одна метрика: среднее кол-во заказов на пользователя, что влечет за собой увеличение общей прибыли. Так как результаты довольно спорные, то решить принимать или не принимать новый алгоритм рекомендаций на их основании не представляется возможным. Вопрос дискуссионный, здесь бы я не стал двумя руками против обновления или за него, а пошел бы к продукту, к владельцу и т.п., чтобы обсудить и понять, устраивает ли их прирост в выручке на 52.26% или будем упирать на то, что средний чек никак не поменялся, а это нам важнее.

# In[ ]:




