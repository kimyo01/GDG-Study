
# 📝 Titanic 생존자 예측 프로젝트: 분석 & 개선 과정 요약

## 1. 초기 모델 설계

- 초기에는 기본 변수(`Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `SibSp`, `Parch`)만 사용하여 모델을 구성하였습니다.
- 모델 성능 비교를 위해 **Logistic Regression**, **Random Forest**, **XGBoost** 3가지 모델을 사용하였고, XGBoost가 가장 좋은 결과를 보였습니다.

## 2. Feature Engineering 시도

- 정확도를 높이기 위해 다양한 파생변수(Title, IsAlone -> FamilyGroup 확장)을 사용
- 그러나 `Title`과 `FamilyGroup`을 **너무 세분화**했을 때 성능이 오히려 낮아지는 문제가 발생
  - 타이틀이 너무 많아져 학습이 과적합되었음
  - FamilyGroup이 의미 있는 구분을 제공하지 않음

## 3. 최적화 방향 결정

- 오히려 변수 수를 줄이고, 검증된 주요 변수만 사용하는 방향으로 수정
- 최종 반영된 변수:
  - `Title` (Mr, Miss, Mrs, Master, Rare)
  - `IsAlone`
  - `FareBin`
  - `Sex`, `Pclass`, `Embarked` 등 기본 변수 유지

## 4. 최종 결과 및 제출

- 가장 높은 성능을 보인 모델은 XGBoost
- 최종 정확도는 약 0.8249

## ✅ 느낀 점

- 무조건 많은 파생 변수가 성능을 높이지는 않으며, 주요한 파생 변수를 선정하는 것이 중요하다고 느낌
