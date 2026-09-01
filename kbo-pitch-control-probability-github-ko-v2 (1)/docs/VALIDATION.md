# 검증 설계와 누수 방지

## 행 독립성

평가 데이터의 다른 행을 이용해 rolling, expanding, 누적 차분을 만들지 않았다. production 예측은 다음 정보만 사용했다.

1. 현재 평가 행의 열
2. 학습 시점에 고정한 모델과 상수
3. cutoff 이전 공식 이력으로 만든 선수 lookup

`tests/test_core.py`에는 입력 행 순서를 뒤집어도 각 행의 feature가 동일한지 확인하는 테스트가 있다.

## 시간 분할

무작위 K-fold 하나로 끝내지 않았다.

- 2022: 더 이른 시즌으로 학습, pitcher-disjoint fold 평가
- 2023: 2022까지의 데이터로 평가
- 2024: 구조와 가중치를 고정한 뒤 마지막 seal
- 2025 배포: 2024까지의 데이터로 production 모델 학습

Futures는 라벨 regime이 달라 별도 가중 정책을 사용했다.

## 후보 통과 기준

대형 후보의 기본 gate는 다음과 같았다.

- 개발 연도 2022·2023 모두 양수
- 2024 FULL 위 gain ≥ +4
- 5개 fold 중 최소 4개 양수
- worst fold > -0.5

작은 후보는 더 낮은 gain 문턱을 허용하되, OOF correction 간 상관과 worst fold를 확인한 후에만 조합했다.

## 왜 강한 기준선 위에서 봤나

V7은 단독으로 +15.88처럼 보였지만 강한 V8×TrackMan 위에서는 +0.3~0.6으로 압축됐다. 반대로 Pattern16은 강한 기준선 위에서도 +5.46이 남았다. 따라서 `standalone score`가 아니라 `current FULL 위 marginal gain`을 축의 가치로 정의했다.

## 위약 대조

열을 추가하면 정보가 없어도 정규화 효과로 점수가 움직일 수 있었다. 그래서 같은 열 수의 shuffled feature를 추가한 placebo와 비교했다. 실제 24열 묶음의 연도별 gain과 placebo를 함께 재어 순수 정보 효과를 분리했다.
