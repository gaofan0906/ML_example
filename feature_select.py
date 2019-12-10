import pandas as pd
import numpy as np

df = pd.read_csv('./data.csv')

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from matplotlib import font_manager
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 卡方检验
# X = df.drop(columns=['有无酮症', '采样日期', '病程（年）'])
# y = df['有无酮症']
# # x中不能包含负值
# chi = chi2(X, y)

# ## 递归消除法进行特征选择，无特征重要性
X = df.drop(columns=['有无酮症', '采样日期'])
y = df['有无酮症']
#
# 参数可调，结果不一样
# rf = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=5)
# rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5),
#               scoring='accuracy')
# rfecv.fit(X, y)
#
# # 交叉验证的结果绘图
# print("Optimal number of features : %d" % rfecv.n_features_)
# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# # 每个特征子集的得分，每次消除1个所以一共有37个特征集合，每次递归消除权值系数对应的特征
# # plt.savefig('./feature_select/feature_select1.png')
# plt.show()
#
# ## 将结果保存到新的表格中
# rank = list(rfecv.ranking_)
# sel_index = [i for i, j in enumerate(rank) if j == 1]
# print(sel_index)
# print(len(sel_index))
# temp_data1 = df.iloc[:, 1:2]
# for i in sel_index[1:]:
#     temp_data = X.iloc[:, i:i + 1]
#     temp_data1 = temp_data1.join(temp_data)
#
# temp_data1 = temp_data1.join(y)
# temp_data1.to_csv('./featrue_select_data/se_data1.csv', index=False)

##根据特征重要性选择特征,输出特征排序的图
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="./simsun.ttc")
X = df.drop(columns=['有无酮症', '采样日期'])
Y = df['有无酮症']
forest1 = RandomForestClassifier(n_estimators=20, random_state=0)
forest1.fit(X, Y)
importance = forest1.feature_importances_
imp_result = np.argsort(importance)[::-1]

for f in range(X.shape[1]):
    print("%2d. %-*s %f" % (f + 1, 30, imp_result[f], importance[imp_result[f]]))

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importance[imp_result], color='lightblue', align='center')
plt.xticks(range(X.shape[1]), imp_result, rotation=90, fontproperties=my_font)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
# plt.savefig('./feature_select/feature_select2.png')
plt.show()

model = SelectFromModel(forest1, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
print(imp_result[:X_new.shape[1]])

# 将结果保存到新的表格中
sel_index = sorted(imp_result[:X_new.shape[1]])
print(sel_index)
print(len(sel_index))
temp_data1 = X.iloc[:, 1:2]
for i in sel_index[1:]:
    temp_data = X.iloc[:, i:i + 1]
    temp_data1 = temp_data1.join(temp_data)

temp_data1 = temp_data1.join(Y)
temp_data1.to_csv('./featrue_select_data/se_data2.csv', index=False)

forest2 = ExtraTreesClassifier(n_estimators=20,
                               random_state=0)

forest2.fit(X, Y)
importances = forest2.feature_importances_
# std = np.std([tree.feature_importances_ for tree in forest2.estimators_],
#              axis=0)
# yerr=std[indices],
indices = np.argsort(importances)[::-1]

col_list = df.columns.values.tolist()
col_name = []
for i in indices:
    col_name.append(col_list[i])

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="lightblue", align="center")
plt.xticks(range(X.shape[1]), col_name, rotation=90, fontproperties=my_font)
plt.xlim([-1, X.shape[1]])
# plt.savefig('./feature_select/feature_select3.png')
plt.show()
model = SelectFromModel(forest2,threshold=0.03, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
print(sorted(indices[:X_new.shape[1]]))
print(sorted(imp_result[:X_new.shape[1]]))

## 将结果保存到新的表格中
sel_index = sorted(indices[:X_new.shape[1]])
print(sel_index)
print(len(sel_index))
temp_data1 = X.iloc[:, 1:2]
for i in sel_index[1:]:
    temp_data = X.iloc[:, i:i + 1]
    temp_data1 = temp_data1.join(temp_data)

temp_data1 = temp_data1.join(Y)
# temp_data1.to_csv('./featrue_select_data/se_data3.csv', index=False)
