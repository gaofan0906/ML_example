import pandas as pd

##读取数据
df=pd.read_excel('2019-10-16 数据汇总（发医渡云）.xlsx',sheet_name='整合后 (6)')

##剔除编号，诊断，主诉
del_name=['编号','诊断','主诉']
df.drop(columns=del_name,inplace=True)

enum_cols=[]
threshold = 10
for col in df.columns:
    num=len(df[col].unique())
    if num<threshold:
        enum_cols.append(col)
for col_name in enum_cols:
    df[col_name]=df[col_name].astype('category')

# 可视化定性数据
from matplotlib import pyplot as plt
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname="./simsun.ttc")
draw_name=['性别',
 '民族',
 'MA (mgl/L)',
 ' SLC30A8 SNP',
 '胰岛自身抗体（RLA法）ZnT8A',
 '胰岛自身抗体（RLA法）GADA',
 '胰岛自身抗体（RLA法）IA-2A',
 '胰岛自身抗体（ECL法）GADA',
 '胰岛自身抗体（ECL法）IAA',
 '胰岛自身抗体（ECL法）IA-2A',
 'TGA']
for col_name in draw_name:
    df[col_name] = df[col_name].cat.add_categories(['None'])
    df[col_name] = df[col_name].fillna('None')

    x = df[col_name].unique()
    y = df[col_name].value_counts()

    B=plt.bar(x, y, align='center')

    plt.title('数据展示',fontproperties=my_font, size=15)
    plt.ylabel('个数',fontproperties=my_font, size=15)
    plt.xlabel('{}'.format(col_name),fontproperties=my_font, size=15)

    xtick_labels = df[col_name].unique()
    plt.xticks(x,xtick_labels,fontproperties=my_font)
    for b in B:
        h=b.get_height()
        plt.text(b.get_x()+b.get_width()/2,h,'%d'%int(h),ha='center',va='bottom')
    # plt.savefig("./pictures/{}.png".format(col_name))
    # plt.show()


df['胰岛抗体阳性个数(RLA法)'] = df['胰岛抗体阳性个数(RLA法)'].cat.add_categories(['None'])
df['胰岛抗体阳性个数(RLA法)'] = df['胰岛抗体阳性个数(RLA法)'].fillna('None')

x=['1','0','2','3','None']
y = df['胰岛抗体阳性个数(RLA法)'].value_counts()
B = plt.bar(x, y, align='center')

plt.title('数据展示', fontproperties=my_font, size=15)
plt.ylabel('个数', fontproperties=my_font, size=15)
plt.xlabel('胰岛抗体阳性个数(RLA法)', fontproperties=my_font, size=15)

xtick_labels = df['胰岛抗体阳性个数(RLA法)'].unique()
plt.xticks(x, xtick_labels, fontproperties=my_font)
for b in B:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2, h, '%d' % int(h), ha='center', va='bottom')
# plt.savefig("./pictures/胰岛抗体阳性个数(RLA法).png")
# plt.show()

draw_name2=[
 '并发症.糖尿病肾病',
 '并发症.糖尿病视网膜病变',
 '并发症.糖尿病神经病变',
 '心脑血管病变',
 '有无其他自身免疫性疾病',
 '有无酮症',
 '有无酮症酸中毒']
for col_name in draw_name2:
    df[col_name] = df[col_name].cat.add_categories(['None'])
    df[col_name] = df[col_name].fillna('None')

    x=['0','1','None']
    y = df[col_name].value_counts()

    B=plt.bar(x, y, align='center')

    plt.title('数据展示',fontproperties=my_font, size=15)
    plt.ylabel('个数',fontproperties=my_font, size=15)
    plt.xlabel('{}'.format(col_name),fontproperties=my_font, size=15)

    xtick_labels = df[col_name].unique()
    plt.xticks(x,xtick_labels,fontproperties=my_font)
    for b in B:
        h=b.get_height()
        plt.text(b.get_x()+b.get_width()/2,h,'%d'%int(h),ha='center',va='bottom')
    # plt.savefig("./pictures/{}.png".format(col_name))
    # plt.show()


object_columns = df.dtypes[df.dtypes=='object']
object_columns = object_columns.index
float_columns_s = []
for col in object_columns:
    style = df[col][df[col].notnull()].unique()[:5]
    try:
        judge = [float(i) for i in style]
        float_columns_s.append(col)
    except:
        continue

for col in float_columns_s:
    df[col] = pd.to_numeric(df[col], errors='coerce')

a=df.describe()

all_data_na = df.isnull().sum()/len(df)
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
b=missing_data.T
c=a.append(b)
import xlsxwriter
bb = pd.ExcelWriter('data_describe' + '.xlsx',engine='xlsxwriter')
c.to_excel(bb, sheet_name='Sheet1')
bb.save()


# df['MA (mgl/L)'] = df['MA (mgl/L)'].cat.add_categories(['None'])
# df['MA (mgl/L)'] = df['MA (mgl/L)'].fillna('None')

# x = df['MA (mgl/L)'].unique()
# y = df['MA (mgl/L)'].value_counts()
# B = plt.bar(x, y, align='center')
#
# plt.title('数据展示', fontproperties=my_font, size=15)
# plt.ylabel('个数', fontproperties=my_font, size=15)
# plt.xlabel('MA (mgl/L)', fontproperties=my_font, size=15)
#
# xtick_labels = df['MA (mgl/L)'].unique()
# plt.xticks(x, xtick_labels, fontproperties=my_font)
# for b in B:
#     h = b.get_height()
#     plt.text(b.get_x() + b.get_width() / 2, h, '%d' % int(h), ha='center', va='bottom')
# plt.savefig("./pictures/MA.png")
# plt.show()