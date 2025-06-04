
# 📝 Titanic 생존자 예측 프로젝트: 분석 & 개선 과정 요약

---

## 1. 문제 정의 및 목표

- Kaggle에서 제공하는 타이타닉 생존자 예측 데이터를 활용하여, 승객의 정보를 기반으로 생존 여부(`Survived`)를 예측하는 이진 분류 모델을 구축하고, 가장 높은 **Accuracy**를 가진 모델을 선택하여 결과를 제출하는 것이 목표였습니다.

---

## 2. 초기 모델 설계

- 초기에는 기본 변수(`Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `SibSp`, `Parch`)만 사용하여 모델을 구성하였습니다.
- 모델 성능 비교를 위해 **Logistic Regression**, **Random Forest**, **XGBoost** 3가지 모델을 사용하였고, XGBoost가 가장 좋은 결과를 보였습니다.

---

## 3. Feature Engineering 시도

- 정확도를 높이기 위해 다양한 **파생 변수**를 실험하였습니다.
  - 🎩 **`Title`**: 이름에서 호칭(Mr, Miss 등)을 추출하여 계층 정보 활용
  - 👨‍👩‍👧‍👦 **`FamilySize`**: `SibSp + Parch + 1`
  - ✅ **`IsAlone`**: 혼자인 승객 여부
  - 💰 **`FareBin`**: 요금을 4구간으로 나눈 이산형 변수
  - 📶 **`FamilyGroup`**: FamilySize를 Small/Medium/Large로 구분
  - 🛏️ **`Deck`, `TicketPrefix`** 등도 후보로 고려

---

## 4. 문제 발생 및 분석

- `Title`과 `FamilyGroup`을 **너무 세분화**했을 때 성능이 오히려 낮아지는 문제가 발생
  - Rare한 타이틀이 너무 많아져 학습이 과적합되었음
  - FamilyGroup이 의미 있는 구분을 제공하지 않음
- Label Encoding으로 처리한 범주형 변수가 **모델에 순서를 부여하는 오해**를 줄 수 있었음

---

## 5. 최적화 방향 결정

- 오히려 변수 수를 줄이고, 검증된 주요 변수만 사용하는 방향으로 수정
- 최종 반영된 변수:
  - `Title` (Mr, Miss, Mrs, Master, Rare)
  - `IsAlone`
  - `FareBin`
  - `Sex`, `Pclass`, `Embarked` 등 기본 변수 유지
- 범주형 변수는 모두 **One-Hot Encoding**으로 처리

---

## 6. 최종 결과 및 제출

- 가장 높은 성능을 보인 모델은 **XGBoost**
- 최종 정확도는 약 **0.85 ~ 0.87**
- 제출 파일: `titanic_best_simplified_XGBoost_acc0.85xx.csv`

---

## ✅ 느낀 점

- 무조건 많은 파생 변수가 성능을 높이지는 않으며, **"좋은 변수 몇 개가 더 중요하다"**는 사실을 실험을 통해 확인할 수 있었습니다.
- 실제 모델 성능은 **적절한 Feature Engineering + 과적합 방지 + 모델 선택**이 균형을 이룰 때 가장 높아졌습니다.
