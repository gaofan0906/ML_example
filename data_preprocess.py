import pandas as pd

##读取数据
df=pd.read_excel('2019-10-16 数据汇总（发医渡云）.xlsx',sheet_name='整合后 (6)')

##剔除编号，诊断，主诉
del_name=['编号','诊断','主诉']
df.drop(columns=del_name,inplace=True)
##处理脏数据
df['性别'].unique()
df['性别'].replace(' 男','男',inplace=True)

df['民族'].unique()
df['民族'].replace(' 汉','汉',inplace=True)
df['民族'].replace('回族','回',inplace=True)


# 找到枚举值的列，阈值设置为10
enum_cols = []
threshold = 10
for col in df.columns:
    num = len(df[col].unique())
    if num <= threshold:
        enum_cols.append(col)
for col in enum_cols:
    df[col] = df[col].astype('category')

# 从object数据中提取float数据列，判断条件是取出每列前5个元素
# 如果能够转换为数字，则认为是float
object_columns = df.dtypes[df.dtypes=='object']
object_columns = object_columns.index
float_columns = []
for col in object_columns:
    style = df[col][df[col].notnull()].unique()[:5]
    try:
        judge = [float(i) for i in style]
        float_columns.append(col)
    except:
        continue
# 识别所有float类型的数据列，将类型转为float，将无效类型转为空
for col in float_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 删除缺失值较多的特征
all_data_na = df.isnull().sum()/len(df)
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
# 删除缺失值大于90的特征,得到temp_data
high_missing_data_cols = missing_data[missing_data['Missing Ratio']>=0.9].index
temp = df
temp_data = temp.drop(high_missing_data_cols, axis=1)

# 对float类型的数据进行填充均值
float_columns = temp_data.dtypes[temp_data.dtypes=='float64'].index
for col in float_columns:
    temp_data[col] = temp_data[col].fillna(temp_data[col].mean())
# 对int类型的数据进行填充均值
int_columns = temp_data.dtypes[temp_data.dtypes=='int64'].index
for col in int_columns:
    temp_data[col] = temp_data[col].fillna(temp_data[col].mean())
# 对category类型的数据填充类别少的一方
category_columns = temp_data.dtypes[temp_data.dtypes=='category'].index
for col in category_columns:
    fill_val=df[col].value_counts().index[-1]
    temp_data[col] = temp_data[col].fillna(fill_val)

print(temp_data.shape)
print(temp_data.dtypes)
# 从temp_data中删除object类型数据，只分析category和float
object_columns = temp_data.dtypes[temp_data.dtypes=='object']
object_columns = object_columns.index
temp_data.drop(object_columns, axis=1, inplace=True)


# 对category类型的数据进行labelEncoder
from sklearn.preprocessing import LabelEncoder
for col in category_columns:
    lbl = LabelEncoder()
    lbl.fit(list(temp_data[col].values))
    temp_data[col] = lbl.transform(list(temp_data[col].values))

# temp_data.to_csv('./data.csv',index=False)
print(temp_data.shape)