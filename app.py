import streamlit as st
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import cartoon_html
import test2
# streamlit run app.py
# 侧边栏
st.sidebar.header("数据层级关系图")
st.sidebar.graphviz_chart('''
digraph{
第一层级 -> 第三层级
第二层级 -> 第三层级
第三层级 -> 第四层级
}
''')
st.sidebar.header("基础算法简介")
choose = st.sidebar.selectbox("", ('GBDT算法', 'GA优化算法'))
if choose=='GBDT算法':
    st.sidebar.write('GBDT（Gradient Boosting Decision Tree）是一种迭代的决策树算法，\
    又叫 MART（Multiple Additive Regression Tree)，它通过构造一组弱的学习器（树），\
    并把多颗决策树的结果累加起来作为最终的预测输出。该算法将决策树与集成思想进行了有效的结合。')
    st.sidebar.write('我们需要知道的是，度量任何一个模型最重要的就是这个模型的损失函数。\
    我们训练的目标就是使得损失函数L最小化。')
    st.sidebar.image('1.png', width=350)
    st.sidebar.write('当损失函数是平方损失和指数损失时，每一步优化是很简单的。\
    但对一般损失函数而言，往往每一步优化没那么容易，如绝对值损失函数和Huber损失函数。常见的损失函数及其梯度如下表所示：')
    st.sidebar.image('2.png', width=350)
    st.sidebar.write('如何使得损失函数最小化？调整参数，使得损失沿着梯度方向下降。')
    st.sidebar.write('对于损失函数为平方损失函数的，我们可以使用的是yj-Ti对xj的\
    预测结果作为残差。那么对于其他类型的损失函数我们应该使用什么作为残差以达到最好\
    的效果呢呢？针对这一问题，Freidman提出了梯度提升算法：利用最速下降的近似方法，\
    即利用损失函数的负梯度在当前模型的值。')
    st.sidebar.image('3.png', width=350)
    st.sidebar.write('如果我们对提升树的损失函数求偏导，就可以发现，偏导是等于残\
    差的，见上图。（因为上文的残差是损失函数梯度的一个特例，对应的损失函数为平方损失\
    函数）。因此，对于不同的损失函数，我们可以使用损失函数的偏导作为我们的残差。')
    st.sidebar.write('这就是梯度提升决策树了。')
    pass
else:
    st.sidebar.write('遗传算法基本步骤：')
    st.sidebar.write('1.确定适应度函数的取值范围，确立精度及染色体编码长度。')
    st.sidebar.write('2.初始化操作：染色体编码，确立种群数量，交叉、变异概率等。')
    st.sidebar.write('3.初始化种群：随机生成第一代种群。')
    st.sidebar.write('4.利用适应度函数评价种群，判断是否满足停止条件，若是则停止，输出最优解；否则继续进行操作。')
    st.sidebar.write('5.对种群进行选择、交叉、变异操作，得到下一代种群，回到第4步。')
    st.sidebar.image('4.png', width=350)
    st.sidebar.write('可得到的结果：')
    st.sidebar.write('1. 最佳适应值的个体染色体编码，通过解码操作获取自变量所对应的值。')
    st.sidebar.write('2. 最佳适应度值，也就是算法找到的全局最优解。')
    st.sidebar.write('3. 取得最优解的迭代次数（进化到第几代）。')
    pass


# 正文
cartoon_html.cartoon_html()
st.subheader('1.读入数据')
# 数据描述
# 后面补一个快速加载数据
# 选择预测的层级
genre = st.selectbox(
    "选择你要预测的层级数据",
    ('第一层级', '第二层级', '第三层级', '第四层级'))


@st.cache_data
def load_data():
    df1 = pd.read_excel('第一层级预测.xlsx')
    df1.fillna(0)
    for col in df1.columns:
        df1[col] = df1[col].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    df2 = pd.read_excel('第二层级预测.xlsx')
    df2.fillna(0)
    for col in df2.columns:
        df2[col] = df2[col].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    df3 = pd.read_excel('第三层级预测.xlsx')
    df3.fillna(0)
    for col in df3.columns:
        df3[col] = df3[col].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    df4 = pd.read_excel('第四层级预测.xlsx')
    df4.fillna(0)
    for col in df4.columns:
        df4[col] = df4[col].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df1, df2, df3, df4


df1, df2, df3, df4 = load_data()
if genre == '第一层级':
    st.caption('第一层级数据')
    st.dataframe(df1)
    X = df1.drop(columns=(['HPT同心度（mm）', 'HPT组合不平衡大小（mm）', 'HPT组合不平衡角度（°）']))
    options = st.multiselect(
        '选择你要预测的目标变量',
        ['HPT同心度（mm）', 'HPT组合不平衡大小（mm）', 'HPT组合不平衡角度（°）'])
    flag=0
    agree = st.radio('选择进行GA优化参数',options=['默认参数运行','进行GA优化'])
    if agree=='进行GA优化':
        population_size=st.slider('选择种群大小',min_value=20,max_value=100,key=20)
        iter_num=st.slider('选择迭代次数', min_value=50, max_value=150, key=1)
        pc=st.slider('选择交叉概率阈值', min_value=0.40, max_value=0.99, key=0.01)
        pm=st.slider('选择变异概率阈值', min_value=0.01, max_value=0.10, key=0.02)
        agree1 = st.radio('参数选择完成？', options=['否','是'])
        if agree1=='是':
            st.write('不好意思，跑不动')
            flag = 1
            flag = 0
    for em in options:
        y = df1[em]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        if flag==1:
            chromosome_num = 2
            max_value = 10
            chromosome_length = 20
            res = test2.diaoyong(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
                                 X_train, X_test, y_train, y_test)
            model = GradientBoostingRegressor(random_state=123,n_estimators=res[0],learning_rate=res[1])
        else:
            model = GradientBoostingRegressor(random_state=123)
            model.fit(X_train, y_train)
            # 模型预测与评估
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # 对比预测值和实际值
            chart_data = pd.DataFrame()
            chart_data['预测值'] = list(y_test_pred)
            chart_data['实际值'] = list(y_test)
            st.caption(f"{em}--误差分析")
            wucha = pd.DataFrame()
            # 解释方差分
            wucha['解释方差分'] = [abs(metrics.explained_variance_score(y_test, y_test_pred))]
            # 平均绝对误差
            wucha['平均绝对误差'] = [abs(metrics.mean_absolute_error(y_test, y_test_pred))]
            # 均方误差
            train_err = metrics.mean_squared_error(y_train, y_train_pred)
            test_err = metrics.mean_squared_error(y_test, y_test_pred)
            wucha['训练集误差'] = [abs(train_err)]
            wucha['测试集误差为'] = [abs(test_err)]
            score = model.score(X_test, y_test)
            wucha['预测得分'] = [abs(score)]
            st.write(wucha)

            # 可视化
            # 显示预测数据集真实值和预测值
            st.caption(f"{em}--预测图分析")
            st.line_chart(chart_data)
elif genre == '第二层级':
    st.caption('第二层级数据')
    st.dataframe(df2)
    X = df2.drop(columns=(['HPC同心度（mm）', 'HPC组合不平衡大小（mm）', 'HPC组合不平衡角度（°）']))
    options = st.multiselect(
        '选择你要预测的目标变量',
        ['HPC同心度（mm）', 'HPC组合不平衡大小（mm）', 'HPC组合不平衡角度（°）'])
    flag = 0
    agree = st.radio('选择进行GA优化参数', options=['默认参数运行', '进行GA优化'])
    if agree == '进行GA优化':
        population_size = st.slider('选择种群大小', min_value=20, max_value=100, key=20)
        iter_num = st.slider('选择迭代次数', min_value=50, max_value=150, key=1)
        pc = st.slider('选择交叉概率阈值', min_value=0.40, max_value=0.99, key=0.01)
        pm = st.slider('选择变异概率阈值', min_value=0.01, max_value=0.10, key=0.02)
        agree1 = st.radio('参数选择完成？', options=['否', '是'])
        if agree1 == '是':
            st.write('不好意思，跑不动')
            flag = 1
            flag = 0
    for em in options:
        y = df2[em]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        if flag == 1:
            chromosome_num = 2
            max_value = 10
            chromosome_length = 20
            res = test2.diaoyong(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
                                 X_train, X_test, y_train, y_test)
            model = GradientBoostingRegressor(random_state=123, n_estimators=res[0], learning_rate=res[1])
        else:
            model = GradientBoostingRegressor(random_state=123)
            model.fit(X_train, y_train)
            # 模型预测与评估
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # 对比预测值和实际值
            chart_data = pd.DataFrame()
            chart_data['预测值'] = list(y_test_pred)
            chart_data['实际值'] = list(y_test)
            st.caption(f"{em}--误差分析")
            wucha = pd.DataFrame()
            # 解释方差分
            wucha['解释方差分'] = [abs(metrics.explained_variance_score(y_test, y_test_pred))]
            # 平均绝对误差
            wucha['平均绝对误差'] = [abs(metrics.mean_absolute_error(y_test, y_test_pred))]
            # 均方误差
            train_err = metrics.mean_squared_error(y_train, y_train_pred)
            test_err = metrics.mean_squared_error(y_test, y_test_pred)
            wucha['训练集误差'] = [abs(train_err)]
            wucha['测试集误差为'] = [abs(test_err)]
            score = model.score(X_test, y_test)
            wucha['预测得分'] = [abs(score)]
            st.write(wucha)

            # 可视化
            # 显示预测数据集真实值和预测值
            st.caption(f"{em}--预测图分析")
            st.line_chart(chart_data)
elif genre == '第三层级':
    st.caption('第三层级数据')
    st.dataframe(df3)
    X = df3.drop(columns=(['HPC+HPT组合件初始不平衡大小（gmm）', 'HPC+HPT组合件初始不平衡角度（°）', 'HPC+HPT组合件同心度（mm）']))
    base1 = ['HPT同心度（mm）', 'HPT组合不平衡大小（mm）', 'HPT组合不平衡角度（°）']
    base2 = ['HPC同心度（mm）', 'HPC组合不平衡大小（mm）', 'HPC组合不平衡角度（°）']
    for em in base1:
        X[em] = df1[em]
    for em in base2:
        X[em] = df2[em]
    options = st.multiselect(
        '选择你要预测的目标变量',
        ['HPC+HPT组合件初始不平衡大小（gmm）', 'HPC+HPT组合件初始不平衡角度（°）', 'HPC+HPT组合件同心度（mm）'])
    flag = 0
    agree = st.radio('选择进行GA优化参数', options=['默认参数运行', '进行GA优化'])
    if agree == '进行GA优化':
        population_size = st.slider('选择种群大小', min_value=20, max_value=100, key=20)
        iter_num = st.slider('选择迭代次数', min_value=50, max_value=150, key=1)
        pc = st.slider('选择交叉概率阈值', min_value=0.40, max_value=0.99, key=0.01)
        pm = st.slider('选择变异概率阈值', min_value=0.01, max_value=0.10, key=0.02)
        agree1 = st.radio('参数选择完成？', options=['否', '是'])
        if agree1 == '是':
            st.write('不好意思，跑不动')
            flag = 1
            flag = 0
    for em in options:
        y = df3[em]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        if flag == 1:
            chromosome_num = 2
            max_value = 10
            chromosome_length = 20
            res = test2.diaoyong(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
                                 X_train, X_test, y_train, y_test)
            model = GradientBoostingRegressor(random_state=123, n_estimators=res[0], learning_rate=res[1])
        else:
            model = GradientBoostingRegressor(random_state=123)
            model.fit(X_train, y_train)
            # 模型预测与评估
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # 对比预测值和实际值
            chart_data = pd.DataFrame()
            chart_data['预测值'] = list(y_test_pred)
            chart_data['实际值'] = list(y_test)
            st.caption(f"{em}--误差分析")
            wucha = pd.DataFrame()
            # 解释方差分
            wucha['解释方差分'] = [abs(metrics.explained_variance_score(y_test, y_test_pred))]
            # 平均绝对误差
            wucha['平均绝对误差'] = [abs(metrics.mean_absolute_error(y_test, y_test_pred))]
            # 均方误差
            train_err = metrics.mean_squared_error(y_train, y_train_pred)
            test_err = metrics.mean_squared_error(y_test, y_test_pred)
            wucha['训练集误差'] = [abs(train_err)]
            wucha['测试集误差为'] = [abs(test_err)]
            score = model.score(X_test, y_test)
            wucha['预测得分'] = [abs(score)]
            st.write(wucha)

            # 可视化
            # 显示预测数据集真实值和预测值
            st.caption(f"{em}--预测图分析")
            st.line_chart(chart_data)
else:
    st.caption('第四层级数据')
    st.dataframe(df4)
    X = df4.drop(columns=(['振动V1（mm/s）', '振动V2（mm/s）', '振动V3（mm/s）']))
    base = ['HPC+HPT组合件初始不平衡大小（gmm）', 'HPC+HPT组合件初始不平衡角度（°）', 'HPC+HPT组合件同心度（mm）']
    for em in base:
        X[em] = df3[em]
    options = st.multiselect(
        '选择你要预测的目标变量',
        ['振动V1（mm/s）', '振动V2（mm/s）', '振动V3（mm/s）'])
    flag = 0
    agree = st.radio('选择进行GA优化参数', options=['默认参数运行', '进行GA优化'])
    if agree == '进行GA优化':
        population_size = st.slider('选择种群大小', min_value=20, max_value=100, key=20)
        iter_num = st.slider('选择迭代次数', min_value=50, max_value=150, key=1)
        pc = st.slider('选择交叉概率阈值', min_value=0.40, max_value=0.99, key=0.01)
        pm = st.slider('选择变异概率阈值', min_value=0.01, max_value=0.10, key=0.02)
        agree1 = st.radio('参数选择完成？', options=['否', '是'])
        if agree1 == '是':
            st.write('不好意思，跑不动')
            flag = 1
            flag = 0
    for em in options:
        y = df4[em]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        if flag == 1:
            chromosome_num = 2
            max_value = 10
            chromosome_length = 20
            res = test2.diaoyong(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
                                 X_train, X_test, y_train, y_test)
            model = GradientBoostingRegressor(random_state=123, n_estimators=res[0], learning_rate=res[1])
        else:
            model = GradientBoostingRegressor(random_state=123)
            model.fit(X_train, y_train)
            # 模型预测与评估
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # 对比预测值和实际值
            chart_data = pd.DataFrame()
            chart_data['预测值'] = list(y_test_pred)
            chart_data['实际值'] = list(y_test)
            st.caption(f"{em}--误差分析")
            wucha = pd.DataFrame()
            # 解释方差分
            wucha['解释方差分'] = [abs(metrics.explained_variance_score(y_test, y_test_pred))]
            # 平均绝对误差
            wucha['平均绝对误差'] = [abs(metrics.mean_absolute_error(y_test, y_test_pred))]
            # 均方误差
            train_err = metrics.mean_squared_error(y_train, y_train_pred)
            test_err = metrics.mean_squared_error(y_test, y_test_pred)
            wucha['训练集误差'] = [abs(train_err)]
            wucha['测试集误差为'] = [abs(test_err)]
            score = model.score(X_test, y_test)
            wucha['预测得分'] = [abs(score)]
            st.write(wucha)

            # 可视化
            # 显示预测数据集真实值和预测值
            st.caption(f"{em}--预测图分析")
            st.line_chart(chart_data)
