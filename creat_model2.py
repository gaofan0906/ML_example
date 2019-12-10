# XGBOOST

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, \
    classification_report, confusion_matrix

# df = pd.read_csv('./data.csv')
# data_path='/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data1.csv'
# data_path='/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data2.csv'
data_path = '/Users/gaofan/Desktop/糖尿病I型/featrue_select_data/se_data3.csv'
df = pd.read_csv(data_path)

# 1.加载数据
# X=df.drop(columns=['有无酮症','采样日期'])
X = df.drop(columns='有无酮症')
y = df['有无酮症']

# 训练集和测试集
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0)

# 模型初始化设置
import xgboost as xgb
from xgboost import XGBClassifier

# dtrain = xgb.DMatrix(train_x, label=train_y)
# dtest = xgb.DMatrix(test_x)
#
# # booster:
# params = {'booster': 'gbtree',
#           'objective': 'binary:logistic',
#           'eval_metric': 'auc',
#           'max_depth': 3,
#           'lambda': 15,
#           'subsample': 0.75,
#           'colsample_bytree': 0.75,
#           'min_child_weight': 1,
#           'eta': 0.025,
#           'seed': 0,
#           'nthread': 8,
#           'silent': 1,
#           'gamma': 6,
#           'learning_rate': 0.05}
#
# watchlist = [(dtrain, 'train')]
#
# # 建模与预测:NUM_BOOST_round迭代次数和数的个数一致
# bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
# # ypred预测为1的概率
# ypred = bst.predict(dtest)
# # 设置阈值, 输出一些评价指标，>0.5预测为1，其他预测为0
# y_pred = (ypred >= 0.5) * 1
#
# from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, \
#     classification_report, confusion_matrix
# print("Model in train score is:", accuracy_score(train_x, train_y))
# print("Model in test  score is:", accuracy_score( test_y,y_pred))
# print('AUC: %.4f' % roc_auc_score(test_y, ypred))
# print('ACC: %.4f' % accuracy_score(test_y, y_pred))
# print('Recall: %.4f' % recall_score(test_y, y_pred))
# print('F1-score: %.4f' % f1_score(test_y, y_pred))
# print('Precesion: %.4f' % precision_score(test_y, y_pred))
# print("report is:", classification_report(test_y, y_pred))
# confusion_matrix(test_y, y_pred)
#
# fpr, tpr, _ = roc_curve(test_y, ypred)
# # fpr被错误的分到正类的负样本
# # tpr正样本正确分类
# roc_auc = auc(fpr, tpr)
#
# plt.figure()
# lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# # plt.savefig('./rfauc.png')
# plt.show()

# API接口.fit训练模型
model = XGBClassifier(booster='gbtree',
                      objective='binary:logistic',
                      # eval_metric='auc',
                      max_depth=3,
                      # lambda = 15,
                             subsample=0.75,
                             colsample_bytree=0.75,
                             min_child_weight=1,
                             eta=0.025,
                             seed=0,
                             nthread=8,
                             # silent=1,
                             gamma=10,
                             learning_rate=0.05)
eval_set = [(train_x, train_y), (test_x, test_y)]
# model.fit(train_x, train_y,verbose=True,eval_set=eval_set)
# model.fit(train_x, train_y, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
model.fit(train_x, train_y, eval_metric='auc', eval_set=eval_set, verbose=True)

# make predictions for test data
y_pred = model.predict(test_x)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(test_y, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Model in train score is:", model.score(train_x, train_y))
print("Model in test  score is:", model.score(test_x, test_y))

# # retrieve performance metrics
# results = model.evals_result()
# epochs = len(results['validation_0']['error'])
# x_axis = range(0, epochs)
#
# # plot log loss
# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
# ax.legend()
# plt.ylabel('Log Loss')
# plt.title('XGBoost Log Loss')
# plt.show()
#
# # plot classification error
# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['error'], label='Train')
# ax.plot(x_axis, results['validation_1']['error'], label='Test')
# ax.legend()
# plt.ylabel('Classification Error')
# plt.title('XGBoost Classification Error')
# plt.show()

# 评估预测结果
prob_predict_y_validation = model.predict_proba(test_x)  # 给出带有概率值的结果，每个点所有label的概率和为1
y_score = prob_predict_y_validation[:, 1]
print('AUC: %.4f' % roc_auc_score(test_y, y_score))
print('ACC: %.4f' % accuracy_score(test_y, y_pred))
print('Recall: %.4f' % recall_score(test_y, y_pred))
print('F1-score: %.4f' % f1_score(test_y, y_pred))
print('Precesion: %.4f' % precision_score(test_y, y_pred))
print("report is:", classification_report(test_y, y_pred))

# 画学习曲线
# train_sizes 训练数量
# 全量样本的学习曲线
train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=train_x, y=train_y,
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
plt.savefig('./XGBOOST/learn_rate.png')
plt.show()

# ROC曲线
# 测试集的ROC曲线
prob_predict_y_validation = model.predict_proba(test_x)  # 给出带有概率值的结果，每个点所有label的概率和为1
y_score = prob_predict_y_validation[:, 1]
fpr, tpr, _ = roc_curve(test_y, y_score)
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
plt.savefig('./XGBOOST/fROC.png')
plt.show()
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="./simsun.ttc")

importance = model.feature_importances_
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
plt.savefig('./XGBOOST/feature_importance.png')
plt.show()
#
# from pylab import mpl
#
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# print(xgb.plot_importance(bst))
