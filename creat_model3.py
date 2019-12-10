# RF
# %matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve

# df = pd.read_csv('./data.csv')
# data_path='/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data1.csv'
# data_path='/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data2.csv'
data_path='/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data3.csv'
df = pd.read_csv(data_path)

#1.加载数据
# X=df.drop(columns=['有无酮症','采样日期'])
X=df.drop(columns='有无酮症')
y = df['有无酮症']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

rfc = RandomForestClassifier( n_estimators=100,max_depth=3,random_state=0)
## 训练
rfc = rfc.fit(X_train, y_train)
##预测
y_pred = rfc.predict(X_test)

from sklearn.metrics import classification_report,roc_auc_score,accuracy_score
#  mean accuracy
print("Model in train score is:", rfc.score(X_train, y_train))
print("Model in test  score is:", rfc.score(X_test, y_test))
print("report is:", classification_report(y_test, y_pred))

prob_predict_y_validation = rfc.predict_proba(X_test)  # 给出带有概率值的结果，每个点所有label的概率和为1
print(prob_predict_y_validation)
y_score = prob_predict_y_validation[:, 1]  # 预测为正样本的概率值
print('AUC: %.4f' % roc_auc_score(y_test, y_score))

# 画学习曲线
# train_sizes 训练数量
# 全量样本的学习曲线
train_sizes, train_scores, test_scores = learning_curve(estimator=rfc, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
# 统计结果
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制效果
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.6, 1.0])
plt.savefig('./RF/learn_rate.png')
plt.show()

# ROC曲线
# 测试集的ROC曲线
prob_predict_y_validation = rfc.predict_proba(X_test)  # 给出带有概率值的结果，每个点所有label的概率和为1
y_score = prob_predict_y_validation[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
# fpr被错误的分到正类的负样本
# tpr正样本正确分类
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('./RF/fROC.png')
plt.show()
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="./simsun.ttc")
importance = rfc.feature_importances_
# print(importance)
imp_result = np.argsort(importance)[::-1]

col_list=df.columns.values.tolist()
# print(col_list)
col_name=[]
for i in imp_result:
    col_name.append(col_list[i])


for f in range(X.shape[1]):
    print("%2d. %-*s %f" % (f + 1, 30, imp_result[f], importance[imp_result[f]]))

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importance[imp_result], color='lightblue', align='center')
plt.xticks(range(X.shape[1]), col_name, rotation=90,fontproperties=my_font)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('./RF/feature_importance.png')
plt.show()




# 目的是带大家复习一下交叉验证
# 交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
#
# rfc = RandomForestClassifier(n_estimators=25)
# rfc_s = cross_val_score(rfc, X, y, cv=10)
# clf = DecisionTreeClassifier()
# clf_s = cross_val_score(clf, X, y, cv=10)
# plt.plot(range(1, 11), rfc_s, label="RandomForest")
# plt.plot(range(1, 11), clf_s, label="Decision Tree")
# plt.legend()
# plt.show()
# ====================一种更加有趣也更简单的写法===================#
"""
label = "RandomForest"
for model in [RandomForestClassifier(n_estimators=25),DecisionTreeClassifier()]:
score = cross_val_score(model,wine.data,wine.target,cv=10)
print("{}:".format(label)),print(score.mean())
plt.plot(range(1,11),score,label = label)
plt.legend()
label = "DecisionTree"
"""
# rfc_l = []
# clf_l = []
# for i in range(10):
#     rfc = RandomForestClassifier(n_estimators=25)
#     rfc_s = cross_val_score(rfc, X, y, cv=10).mean()
#     rfc_l.append(rfc_s)
#     clf = DecisionTreeClassifier()
#     clf_s = cross_val_score(clf, X, y, cv=10).mean()
#     clf_l.append(clf_s)
# plt.plot(range(1, 11), rfc_l, label="Random Forest")
# plt.plot(range(1, 11), clf_l, label="Decision Tree")
# plt.legend()
# plt.show()
# # 是否有注意到，单个决策树的波动轨迹和随机森林一致？
# # 再次验证了我们之前提到的，单个决策树的准确率越高，随机森林的准确率也会越高
# #####【TIME WARNING: 2mins 30 seconds】#####
# superpa = []
# for i in range(200):
#     rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1)
#     rfc_s = cross_val_score(rfc, X, y, cv=10).mean()
#     superpa.append(rfc_s)
#     print(max(superpa), superpa.index(max(superpa)))
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 201), superpa)
# plt.show()
