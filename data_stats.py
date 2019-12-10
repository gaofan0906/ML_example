import pandas as pd
import numpy as np

df = pd.read_csv('./data.csv')
object_columns = df.dtypes[df.dtypes=='object']
object_columns = object_columns.index
df.drop(object_columns, axis=1, inplace=True)

category_columns = df.dtypes[df.dtypes=='int64'].index
for i in category_columns:
    print(df[i].value_counts())