import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
df = pd.read_csv('./data.csv')

X = df.drop(columns=['有无酮症', '采样日期','病程（年）'])
y = df['有无酮症']
chi=chi2(X, y )

X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
X_new.shape

