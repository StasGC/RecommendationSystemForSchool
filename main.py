import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import spatial
from scipy.sparse.linalg import svds
from scipy import sparse
import os

# путь к данным
data_dir = '/'.join(os.getcwd().split('/')[:-2] + ['Google Диск', 'Сириус', 'library_recoms'])

# указываем параметры для нашего алгоритма
rec_week_num = 43
rec_week_year = 2018
week_delta = 8

df = pd.read_csv(data_dir+'/materials_actions.csv.gz', sep = ';', compression = 'gzip')
df = df.reset_index().rename(columns = {'index': 'action_id'}, index = str)
df = df[['action_type', 'material_id', 'material_type', 'action_id', 'action_info', 'action_start', 'action_user_id']]
df = df[df.action_type.isin(['добавление в избранное', 'оценка', 'использование в сценарии урока', 'просмотр'])]
df.action_start = df.action_start.apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))

mc = pd.read_csv(data_dir + '/materials_cis.csv.gz', sep = ';', compression = 'gzip')
mc = mc.rename(columns = {'education_level': 'ed_level'}, index = str)
mc['theme'] = list(map(lambda x, y, z: str(x) + '_' + y + '_' + z, mc.code, mc.subject, mc.ed_level))

lp = pd.read_csv(data_dir + '/lesson_teacher.csv.gz', sep = ';', error_bad_lines = False, compression = 'gzip')
lp.date = lp.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
lp['week_num'] = lp.date.apply(lambda x: x.isocalendar()[1])
lp['year_num'] = lp.date.apply(lambda x: x.isocalendar()[0])
lp['theme'] = list(map(lambda x, y, z: x + '_' + y + '_' + z, lp.code, lp.subject, lp.ed_level))
lp = lp[['teacher_id', 'ed_level', 'code', 'week_num', 'year_num', 'theme', 'subject', 'cis']]

marks = df.loc[df.action_type == 'оценка', ['material_id', 'material_type', 'action_id', 'action_info']]
marks.action_info = marks.action_info.apply(float)
marks = marks.groupby(['material_id', 'material_type', 'action_id']).max().reset_index()
marks = pd.merge(marks.groupby(['material_id', 'material_type']).action_info.mean().reset_index(),
                 marks.groupby(['material_id', 'material_type']).action_id.count().reset_index(),
                 how = 'inner', on = ['material_id', 'material_type']
                ).rename(columns = {'action_info': 'avg_rating', 'action_id': 'marks_qty'}, index = str)

def make_ready_for_recom(rec_week_num, rec_week_year):
    # отобрали все действия пользователей за предшествующие week_delta недель
    tmp = df[df.action_start.apply(lambda x: x.isocalendar()[1] < rec_week_num) & (
        df.action_start.apply(lambda x: x.isocalendar()[1] >= rec_week_num - week_delta)) & (
        df.action_start.apply(lambda x: x.isocalendar()[0] == rec_week_year))]
    # нашли все просматриваемые пользователем материалы
    tmp = tmp.groupby(['material_type', 'material_id', 'action_user_id']
                     ).action_id.count().reset_index().iloc[:, :3]
    # связали их с темами
    tmp = pd.merge(tmp, mc.iloc[:, :6], how = 'inner', on = ['material_type', 'material_id'])
    tmp.drop('name', axis = 1 , inplace = True)
    # получили датафрейм всех пользователей и тем
    tmp = tmp.groupby(['action_user_id', 'code', 'subject', 'ed_level']
                     ).material_id.count().reset_index().iloc[:, :4]                                                                                                                    
    tmp = tmp[tmp.code.apply(lambda x: len(str(x)) > 2)]
    t = tmp.groupby(['code', 'subject', 'ed_level']).count().reset_index()
    # оставили все темы, которые встречались хотя бы у двух пользователей
    tmp = pd.merge(tmp, t.loc[t.action_user_id > 1, ['code', 'subject', 'ed_level']], 
               how = 'inner', on = ['code', 'subject', 'ed_level']) 
    tmp['theme'] = list(map(lambda x, y, z: x + '_' + y + '_' + z, tmp.code, tmp.subject, tmp.ed_level))
    tmp.drop(['code', 'subject', 'ed_level'], axis = 1, inplace = True)
    
    # cоздали датафрейм с темами на требуемой неделе
    next_themes = lp.loc[(lp.week_num == rec_week_num) & (lp.code.apply(lambda x: len(str(x)) > 2)), 
                         ['teacher_id', 'theme']]
    
    # создали таблицу с предыдущими просмотрами и с пройденными поурочными планами
    t1 = lp.loc[(lp.week_num < rec_week_num) & (lp.week_num >= rec_week_num - week_delta) & (
        lp.year_num == rec_week_year) & (lp.teacher_id.isin(next_themes.teacher_id.unique())) & (
        lp.code.apply(lambda x: len(str(x)) > 2)), 
                :].groupby(['teacher_id', 'theme']).week_num.count().reset_index().iloc[:, :2]
    t1 = pd.concat([tmp.rename(columns = {'action_user_id': 'teacher_id'}, index = str), t1])
    t1['qty'] = 1
    t1 = t1.groupby(['teacher_id', 'theme']).count().reset_index().iloc[:, :2]
    
    # формируем спарс матрицы
    user_ref = pd.DataFrame({'teacher_id': t1.teacher_id.unique(), 
                             'user_ind': range(t1.teacher_id.unique().shape[0])})
    theme_ref = pd.DataFrame({'theme': t1.theme.unique(), 
                              'theme_ind': range(t1.theme.unique().shape[0])})
    t1 = pd.merge(t1, user_ref, how = 'inner', on = 'teacher_id')
    t1 = pd.merge(t1, theme_ref, how = 'inner', on = 'theme')
    t1['qty'] = 1
    
    size = t1.shape[0]
    row = np.array(t1.user_ind[:size])
    col = np.array(t1.theme_ind[:size])
    data = np.array(t1.qty[:size])
    X = sparse.csr_matrix((data, (row, col))).asfptype()
    
    # Применяем сингулярное разложение
    (U, S, V) = svds(X, k = 100)
    
    # Применяем метод ближайших соседей(KDtree)
    tree = spatial.KDTree(U)
    return [next_themes, t1, user_ref, theme_ref, tree, U]

[next_themes, t1, user_ref, theme_ref, tree, U] = make_ready_for_recom(rec_week_num, rec_week_year)

def recomm_by_les_and_act(teacher_id):
    nn_user_ind = tree.query(U[user_ref.loc[user_ref.teacher_id == teacher_id, 'user_ind'].values[0], :], k = 100)[1]
    
    mat_list = mc.loc[mc.theme.isin(next_themes[next_themes.teacher_id.isin(
        user_ref.loc[user_ref.user_ind.isin(nn_user_ind), 
                     'teacher_id'].values)].theme.unique()), :
                     ].groupby(['material_type', 'material_id']).count().reset_index().iloc[:, :2]
    rec_final = pd.merge(marks, mat_list, how = 'inner', on = ['material_type', 'material_id']
                        ).sort_values(by = ['avg_rating', 'marks_qty'], ascending = False).head(10)
    return rec_final

def make_recom_by_les_plan(teacher_id):
    # тут отбираем темы, интересные для пользователя на неделе
    tmp = lp.loc[(lp.week_num == rec_week_num) & (lp.year_num == rec_week_year), 
                 ['teacher_id', 'code', 'cis', 'week_num', 'subject', 'ed_level']
                ].groupby(['teacher_id', 'code', 'cis', 'subject', 'ed_level']).count().reset_index().iloc[:, :5]
    tmp = tmp[tmp.code.apply(lambda x: len(str(x)) > 2)]
    # отбираем для пользователя наиболее релевантные материалы по поурочным планам
    rec = pd.merge(pd.merge(tmp.loc[tmp.teacher_id == teacher_id, ['teacher_id', 'code', 'ed_level', 'subject']], 
                            mc, how = 'inner', on = ['code', 'ed_level', 'subject']
                           ).groupby(['material_type', 'material_id']).subject.count().reset_index().iloc[:, :2], 
                   marks, how = 'inner', on = ['material_type', 'material_id'])
    return rec.sort_values(by = ['avg_rating', 'marks_qty'], ascending = False).head(10)

def fin_rec(teacher_id):
    res = make_recom_by_les_plan(teacher_id)
    if res.shape[0] == 0:
        res = recomm_by_les_and_act(teacher_id)
    t = res.reset_index().iloc[:, 1:3]
    return {i : {j:t.loc[i, j] for j in t} for i in t.index}