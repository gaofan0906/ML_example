# GBDT

import pandas as pd
import numpy as np
# df = pd.read_csv('./data.csv')
# data_path='/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data1.csv'
# data_path='/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data2.csv'
data_path='/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data3.csv'
df = pd.read_csv(data_path)

# 1.加载数据
# X=df.drop(columns=['有无酮症','采样日期'])
X=df.drop(columns='有无酮症')
y = df['有无酮症']
# 数据切分
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# 2.构建模型
from sklearn.ensemble import GradientBoostingClassifier

dtc = GradientBoostingClassifier(learning_rate=0.005)
'''loss='deviance', learning_rate=0.005, n_estimators=100,
                                 subsample=1.0, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                 max_depth=3, init=None, random_state=None,
                                 max_features=None, verbose=0,
                                 max_leaf_nodes=None, warm_start=False,
                                 presort='auto'
'''

# 3.训练模型
dtc.fit(X_train, y_train)
# 3.测试
y_pred = dtc.predict(X_test)
# 4.模型校验
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score
#  mean accuracy
print("Model in train score is:", dtc.score(X_train, y_train))
print("Model in test  score is:", dtc.score(X_test, y_test))
print("report is:", classification_report(y_test, y_pred))

prob_predict_y_validation = dtc.predict_proba(X_test)  # 给出带有概率值的结果，每个点所有label的概率和为1
y_score = prob_predict_y_validation[:, 1]  # 预测为正样本的概率值
print('AUC: %.4f' % roc_auc_score(y_test, y_score))


# 画学习曲线
# train_sizes 训练数量
# 全量样本的学习曲线
train_sizes, train_scores, test_scores = learning_curve(estimator=dtc, X=X_train, y=y_train,
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
# plt.savefig('./GDBT/learn_rate.png')
plt.show()


# ROC曲线
# 测试集的ROC曲线
prob_predict_y_validation = dtc.predict_proba(X_test)  # 给出带有概率值的结果，每个点所有label的概率和为1
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
# plt.savefig('./GDBT/fROC.png')
plt.show()
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="./simsun.ttc")
importance = dtc.feature_importances_
imp_result = np.argsort(importance)[::-1]
col_list=df.columns.values.tolist()
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
# plt.savefig('./GDBT/feature_importance.png')
plt.show()

# # 交叉验证迭代5次
# scores = cross_val_score(dtc, X_train, y_train, cv=10)
# print(scores)  # 打印输出每次迭代的度量值（准确度）
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）

# ---------------------------执行结果--------------------------------------------
# Model in train score is: 0.929343308396
# Model in test  score is: 0.875420875421
# report is:              precision    recall  f1-score   support
#
#           0       0.98      0.91      0.94        55
#           1       0.89      0.87      0.88        55
#           2       0.91      0.79      0.85        52
#           3       0.88      0.88      0.88        56
#           4       0.92      0.91      0.91        64
#           5       0.97      0.81      0.88        73
#           6       0.87      0.95      0.91        57
#           7       0.81      0.92      0.86        62
#           8       0.81      0.92      0.86        52
#           9       0.77      0.82      0.79        68
#
# avg / total       0.88      0.88      0.88       594
