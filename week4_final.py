
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 데이터 불러오기
train = pd.read_csv("/content/drive/MyDrive/titanic_data/train.csv")
test = pd.read_csv("/content/drive/MyDrive/titanic_data/test.csv")
test_ids = test['PassengerId']

# 2. 전처리 및 Feature Engineering 함수
def preprocess(df):
    df = df.copy()

    # 성별 인코딩
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Title 추출 및 간소화
    def extract_title(name):
        match = re.search(r' ([A-Za-z]+)\.', name)
        return match.group(1) if match else "None"

    df['Title'] = df['Name'].apply(extract_title)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess','Capt','Col','Don','Dr','Major',
         'Rev','Sir','Jonkheer','Dona'], 'Rare'
    )
    df['Title'] = df['Title'].apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs', 'Master'] else 'Rare')

    # 결측치 처리
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # IsAlone 파생 변수
    df['IsAlone'] = ((df['SibSp'] + df['Parch']) == 0).astype(int)

    # Fare 구간화
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

    # 불필요한 열 제거
    df = df.drop(columns=['Cabin', 'Ticket', 'Name'])

    # 원핫 인코딩
    df = pd.get_dummies(df, columns=['Title', 'Embarked'], drop_first=True)

    return df

# 3. 전처리 실행
train_proc = preprocess(train)
test_proc = preprocess(test)

X = train_proc.drop(columns=['Survived', 'PassengerId'])
y = train_proc['Survived']
X_test_final = test_proc.drop(columns=['PassengerId'])
X_test_final = X_test_final[X.columns]

# 4. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# 5. 모델 정의
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# 6. 모델 평가
results = {}
for name, model in models.items():
    print(f"\n▶ Evaluating: {name}")
    preds = cross_val_predict(model, X_scaled, y, cv=5)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)

    model.fit(X_scaled, y)

    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

# 7. 최고 성능 모델 선택
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_acc = results[best_model_name]['accuracy']
print(f"\n✅ Best model selected: {best_model_name} (Accuracy: {best_acc:.4f})")

# 8. 테스트셋 예측 및 제출 파일 생성
preds = best_model.predict(X_test_scaled)
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': preds
})
