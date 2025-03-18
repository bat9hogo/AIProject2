import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('traincompleted.csv')

print(df.head())

df.drop(['PassengerId', 'Name'], axis=1, inplace=True)
df.head()

catСols = df.select_dtypes(include=['object']).columns

labelEncoders = {}
for col in catСols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    labelEncoders[col] = le

df['Transported'] = df['Transported'].astype(int)
X = df.drop('Transported', axis=1)
y = df['Transported']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sbn
import matplotlib.pyplot as plt

logReg = LogisticRegression(max_iter=2000)
logReg.fit(X_train, y_train)

yPredLogReg = logReg.predict(X_test)

print("Логистическая регрессия:")
print(f'Accuracy: {accuracy_score(y_test, yPredLogReg):.2f}')
print(f'Precision: {precision_score(y_test, yPredLogReg):.2f}')
print(f'Recall: {recall_score(y_test, yPredLogReg):.2f}')
print(f'F1 Score: {f1_score(y_test, yPredLogReg):.2f}')

cmLogReg = confusion_matrix(y_test, yPredLogReg)
sbn.heatmap(cmLogReg, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Transported', 'Transported'], yticklabels=['Not Transported', 'Transported'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

from sklearn.linear_model import LinearRegression

linReg = LinearRegression()
linReg.fit(X_train, y_train)

yPredLinReg = linReg.predict(X_test)

yPredLinReg = [1 if y > 0.5 else 0 for y in yPredLinReg]

print("\nЛинейная регрессия:")
print(f'Accuracy: {accuracy_score(y_test, yPredLinReg):.2f}')
print(f'Precision: {precision_score(y_test, yPredLinReg):.2f}')
print(f'Recall: {recall_score(y_test, yPredLinReg):.2f}')
print(f'F1 Score: {f1_score(y_test, yPredLinReg):.2f}')

cmLinReg = confusion_matrix(y_test, yPredLinReg)
sbn.heatmap(cmLinReg, annot=True, fmt='d', cmap='Reds', xticklabels=['Not Transported', 'Transported'], yticklabels=['Not Transported', 'Transported'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Linear Regression')
plt.show()