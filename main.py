import graphviz as graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import os


# 可视化特征重要性并且降序表示
def plot_feature_importances(feature_importances, title, feature_names, normalize=True):
    #     将重要性值标准化
    if normalize:
        feature_importances = 100.0 * (feature_importances / max(feature_importances))

    #     将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))
    #     让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5

    plt.figure(figsize=(16, 9))
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Importance')
    for a, b in zip(pos, feature_importances[index_sorted]):
        plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=20)
    plt.title(title)
    plt.show()


# 将分类变量转化成数字变量，方便后续计算
def trans(x):
    if x == data['Species'].unique()[0]:
        return 0  # Adelie
    if x == data['Species'].unique()[1]:
        return 1  # Gentoo
    if x == data['Species'].unique()[2]:
        return 2  # Chinstrap
    if x == data['Island'].unique()[0]:
        return 0  # Torgersen
    if x == data['Island'].unique()[1]:
        return 1  # Biscoe
    if x == data['Island'].unique()[2]:
        return 2  # Dream
    if x == data['Sex'].unique()[0]:
        return 0  # male
    if x == data['Sex'].unique()[1]:
        return 1  # female
    if x == data['Sex'].unique()[2]:
        return -1  # -1


# python找不到下载的graphviz包
# 路径填自己的
os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin/'

# ---------------------  下方是数据的导入和处理 ---------------------
data = pd.read_csv(open(r'.\data\penguin.csv'))
# 查看数据信息
# print(data.info())
# 补全缺失值
data = data.fillna(-1)
# fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
# value：固定值，可以用固定数字、均值、中位数、众数等，此外还可以用字典，series等形式数据；
# method:填充方法，'bfill','backfill','pad','ffill'
# axis: 填充方向，默认0和index，还可以填1和columns
# inplace:在原有数据上直接修改
# limit:填充个数，如1，每列只填充1个缺失值
# print(data['Sex'].unique())
# print(data['Species'].unique())
# print(data['Island'].unique())

# 将类型变量转换为值变量
data['Species'] = data['Species'].apply(trans)
data['Island'] = data['Island'].apply(trans)
data['Sex'] = data['Sex'].apply(trans)
# ---------------------  上方是数据的导入和处理 ---------------------


feature_data = data[
    ['Island', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex',
     'Age']]
goal_data = data[['Species']]
# 上面两句的另一种写法
# goal_data = data[data['Species'].isin([0, 1, 2])][['Species']]
# feature = data[data['Species'].isin([0, 1, 2])][[
#     'Island', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)',
#     'Body Mass (g)', 'Sex', 'Age'
# ]]

# ---------------------  决策树训练与测试 ---------------------
# 划分训练集 测试集

x_train, x_test, y_train, y_test = train_test_split(feature_data, goal_data, test_size=0.2, random_state=2022)

# 超参数学习曲线 这里只画了一个层数的
test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=i + 1,
                                      random_state=2020,
                                      # 最大深度
                                      splitter='best'
                                      )  # 生成决策树分类器   entropy

    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    test.append(score)

plt.plot(range(1, 11), test, color='red')
plt.ylabel('score')
plt.xlabel('max_depth')
plt.show()
max = test.index(max(test)) + 1
print("该决策树的最佳层数是：", max)

# 训练决策树
penguin_tree = DecisionTreeClassifier(criterion='entropy',
                                      splitter='best',
                                      random_state=2022,
                                      max_depth=max)
penguin_tree.fit(x_train, y_train)

# 返回预测的准确度
print('训练集预测成功率:', penguin_tree.score(x_train, y_train))
print('测试集预测成功率:', penguin_tree.score(x_test, y_test))

# 画决策树
feature_names = ['Island', 'Culmen Length (mm)', 'Culmen Depth (mm)',
                 'Flipper Length (mm)', 'Body Mass (g)', 'Sex', 'Age']
target_names = ['Adelie', 'Gentoo', 'Chinstrap']

plot_feature_importances(penguin_tree.feature_importances_, 'Charcteristic importance',
                         penguin_tree.feature_names_in_,
                         normalize=False)

dot_data = tree.export_graphviz(penguin_tree,
                                feature_names=feature_names,
                                class_names=target_names,
                                out_file=None,
                                filled=True)

graph = graphviz.Source(dot_data)

graph.render("penguin_tree")

# # 在训练集和测试集上分布利用训练好的模型进行预测
# train_predict = penguin_tree.predict(x_train)
# test_predict = penguin_tree.predict(x_test)
#
# ## 利用accuracy 【预测正确的样本数目占总预测样本数目的比例】评估模型效果
# print('训练集预测成功率:', metrics.accuracy_score(y_train, train_predict))
# print('测试集预测成功率:', metrics.accuracy_score(y_test, test_predict))
# ---------------------  决策树训练与测试 ---------------------


# --------------------- 结果可视化 ---------------------
# 查看混淆矩阵
confusion_matrix = metrics.confusion_matrix(penguin_tree.predict(x_test), y_test)
plt.figure()
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
# --------------------- 结果可视化 ---------------------
