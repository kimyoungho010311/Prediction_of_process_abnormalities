# 🏭 Semiconductor Manufacturing Process Anomaly Detection (반도체 공정 이상 탐지)

## 📖 프로젝트 개요 (Overview)

반도체 제조 공정은 수백 개의 복잡한 센서 데이터로 이루어져 있으며, 불량품(Fail)을 조기에 탐지하는 것은 수율(Yield) 향상과 비용 절감에 필수적입니다.
본 프로젝트는 **UCI SECOM 데이터셋**을 활용하여 반도체 공정 데이터를 분석하고, **불균형 데이터(Imbalanced Data)** 문제를 해결하여 공정의 Pass(양품)/Fail(불량)을 예측하는 머신러닝 분류 모델을 구축했습니다.

## 📂 데이터셋 (Dataset)

  * **출처:** UCI Machine Learning Repository (SECOM Data)
  * **크기:** 1567 rows × 592 columns
  * **특징:**
      * **높은 차원:** 590개 이상의 센서 데이터 존재.
      * **결측치(NaN):** 다수의 센서 데이터에 결측치 존재.
      * **클래스 불균형:** Pass(93.4%) vs Fail(6.6%)로 불량이 매우 적은 극심한 불균형 데이터.

## ⚙️ 데이터 전처리 및 의사결정 (Preprocessing & Rationale)

데이터의 특성을 고려하여 모델 성능을 최적화하기 위해 다음과 같은 전처리 전략을 수립했습니다.

### 1\. 결측치 처리 (Handling Missing Values)

  * **Action:** 데이터셋 내 모든 `NaN` 값을 `0`으로 대체.
  * **Why:** 일반적인 통계적 대치(평균, 중앙값) 대신 도메인 특성을 고려했습니다. 반도체 센서 데이터에서 값이 비어있다는 것은 '측정 오류'가 아닌 '신호 없음(No Signal)'을 의미한다고 가정하는 것이 더 타당하다고 판단했습니다.

### 2\. 다중공선성 제거 (Feature Selection)

  * **Action:** 변수 간 상관계수(Correlation)가 **0.7 이상**인 경우, 중복된 정보로 간주하여 하나의 변수만 남기고 제거 (592개 → 306개로 차원 축소).
  * **Why:**
      * **다중공선성 방지:** 상관관계가 지나치게 높은 변수들은 선형 모델(Logistic Regression, Lasso)의 가중치 계산을 불안정하게 만듭니다.
      * **과적합 방지 및 연산 효율성:** 불필요한 중복 변수를 제거함으로써 모델의 일반화 성능을 높이고 학습 속도를 개선했습니다.

### 3\. 데이터 표준화 (Standardization)

  * **Action:** `StandardScaler`를 적용하여 모든 특성의 평균을 0, 분산을 1로 변환.
  * **Why:** 센서마다 측정 단위와 값의 범위(Scale)가 매우 다릅니다(예: 온도 vs 압력). 스케일링을 하지 않을 경우, 값의 범위가 큰 특정 센서가 모델 학습을 주도(Dominate)해버리는 문제를 방지하기 위함입니다.

## ⚖️ 불균형 데이터 해결 전략 (Handling Imbalance)

초기 모델링 결과, 정확도(Accuracy)는 93%로 높았으나 **Recall(재현율)이 0**에 수렴하여 불량품을 전혀 탐지하지 못하는 문제가 발생했습니다. 이를 해결하기 위해 두 가지 샘플링 기법을 적용했습니다.

1.  **Undersampling:** 다수 클래스(Pass) 데이터를 소수 클래스(Fail) 수에 맞춰 무작위 제거.
2.  **Oversampling (SMOTE):** 소수 클래스(Fail) 데이터를 합성하여 증강.

## 📊 모델링 및 성능 평가 (Modeling & Evaluation)

다음 4가지 알고리즘을 사용하여 비교 분석을 수행했습니다.

  * **XGBoost Classifier** (GridSearch CV로 하이퍼파라미터 튜닝)
  * **Random Forest Classifier**
  * **Logistic Regression**
  * **Lasso**

### 📈 실험 결과 요약

불균형 데이터를 처리하기 전과 후의 성능(F1-Score, Recall)을 비교했습니다.

| Model | Sampling Method | Accuracy | F1-Score | 비고 |
|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | Normal | 93.2% | 0.0 | Fail 예측 실패 |
| **Random Forest** | Normal | 93.2% | 0.0 | Fail 예측 실패 |
| **Logistic Regression** | **SMOTE (Over)** | **84.1%** | **0.719** | **Best Performance** |
| **Lasso** | SMOTE (Over) | - | 0.714 | - |
| **Random Forest** | Undersampling | 50.7% | 0.652 | - |

### 🔍 주요 발견 (Key Findings)

1.  **Normal 데이터:** 모든 모델이 Fail을 거의 예측하지 못함 (Accuracy의 함정).
2.  **Undersampling:** Recall은 상승했으나 데이터 손실로 인해 전체적인 모델 신뢰도 하락.
3.  **Oversampling (SMOTE):** 가장 우수한 성능을 보임. 특히 **Logistic Regression**과 결합했을 때 **F1-Score 0.719**로 가장 균형 잡힌 성능을 기록함.
4.  **Feature Importance:** XGBoost 분석 결과, `Feature 126`, `Feature 135`, `Feature 45`가 공정 결과에 가장 큰 영향을 미치는 주요 인자로 식별됨.

## 📝 결론 (Conclusion)

  * 반도체 공정 데이터의 특성상 단순 정확도보다는 **불량품을 놓치지 않는 Recall(재현율)과 F1-Score**가 중요합니다.
  * 본 프로젝트에서는 **SMOTE 오버샘플링** 기법과 **Logistic Regression**을 결합했을 때 불량 탐지 성능이 가장 최적화됨을 확인했습니다.
  * 향후 과제로 중요 특성(Feature Importance 상위 변수) 중심의 정밀 분석과 딥러닝(Autoencoder 등) 기반의 이상 탐지 기법 적용을 고려해볼 수 있습니다.

## 🚀 How to Run

```bash
# 필수 라이브러리 설치
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

# Jupyter Notebook 실행
jupyter notebook Semiconductor_manufacturing_project.ipynb
```
