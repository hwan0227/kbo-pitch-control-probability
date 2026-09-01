# 실험 이력: 0에서 1160까지

## 1. 출발점 — 문제와 데이터부터 다시 정의

행 하나는 투구 하나이고, 목표는 제구 성공 확률이다. 첫 작업은 모델 선택이 아니라 데이터 무결성 검사였다.

- 투수·타자의 `asof_n`이 시간순으로 증가하는지 확인
- 누적 횟수 × 누적 비율이 정수 event count와 맞는지 확인
- count, base, out, score, win expectancy의 대수적 일관성 검사
- 평가 행은 자기 열만 사용하도록 feature builder를 row-local 함수로 제한
- Regular와 Futures가 서로 다른 라벨 체계를 가질 수 있음을 분리 진단

이 단계에서 48개 원열을 선수·현재 상태·매치업·구종 프로필을 포함한 약 80개 core feature로 확장했다.

## 2. 선수 관점 — 과거 능력을 안전하게 요약

초기 모델은 “이 선수는 지금까지 얼마나 잘했는가”에 집중했다.

- 투수·타자 career/as-of 성공률
- 최근 1/3/5경기 form과 career 대비 gap
- count, 상대 손, base/out별 target encoding
- 구종 비중과 entropy
- 표본 수에 따른 league-average shrinkage
- 최근 시즌을 5/4/3으로 가중하는 Marcel prior

LightGBM, XGBoost, CatBoost, RealMLP를 사용했고, 분류와 squared-loss regression을 함께 섞어 비슷한 모델 사이에도 링크 함수 다양성을 만들었다. 이 구조가 1070.1118의 첫 강한 점수를 만들었다.

## 3. 라벨 구조 복원 — 1089.91

`asof_rate × asof_n`의 누적 차분으로 성공/실패뿐 아니라 실패 유형을 복원했다. 이진 모델과 5분류 softmax, 실패 채널을 다시 합치는 reconstruction branch를 섞었다. 같은 입력을 다른 target과 loss로 풀어 예측 상관을 낮춘 것이 핵심이었다.

## 4. 상태 수준 보정 — 1136.81

Brier score는 확률의 수준 오차에도 민감하다. Futures, 일부 팀 관여, base/out 24칸처럼 여러 해에 걸쳐 같은 방향으로 남는 잔차를 교차검증으로 찾고, 지나치게 세분화하지 않은 작은 보정표로 반영했다.

이 단계에서 큰 폭의 LB 상승이 있었지만, 후반부에는 LB probe보다 out-of-time 안정성을 더 엄격히 사용했다. 공개 저장소에는 probe 생성·해법 코드를 포함하지 않았다.

## 5. TrackMan — 1140.50

공식 선수 이력과 TrackMan을 cutoff-safe하게 연결해 구속·회전·무브먼트·릴리스·extension·구종 분포를 만들었다. 평균 프로필은 기존 80열과 많이 겹쳤지만, 투수 내 표준편차는 재구성 R² 0.03~0.24로 프레임 밖의 정보였다. 이 산포 피처와 산술 interaction을 일부 모델 슬롯에 추가해 세 개발 연도 모두 양의 fold gain을 얻었다.

## 6. 관점 전환 — Pattern16, 1154.81

가장 큰 모델링 변화였다. 기존에는 `X → success`를 바로 학습했다. Pattern16은 네 사건을 joint class로 만들어 `X → 사건 구조 → success`로 풀었다.

| bit | 사건 |
|---:|---|
| 1 | reverse |
| 2 | middle |
| 4 | ball |
| 8 | strike |

16-class 단독 예측은 direct binary보다 약했지만 상관이 약 0.77로 달랐다. 강한 TM baseline에 0.10만 섞었을 때 **+5.4643**, 4/5 fold 양수, worst `-0.1141`로 살아남았다. 실제 제출 점수는 1154.8135였다.

## 7. 작은 독립축 조립 — 1157.9984

Pattern16 위에 다음 correction을 작게 더했다.

- Joint6 CatBoost
- Joint6 XGBoost
- Futures trusted-period weighting

조합은 로컬에서 +2.2992, 5/5 양수, worst +1.2637이었다. 서로 비슷한 대형 모델을 다시 평균내기보다 작은 correction의 OOF delta correlation을 확인한 것이 중요했다.

## 8. disagreement gate — 1160.1624

P48 후보를 항상 같은 세기로 넣지 않고 direct, P16, P48이 얼마나 불일치하는지에 따라 correction 강도를 제한적으로 조절했다. 최종 조합은 로컬 +4.2063, 5/5 양수, worst +1.4608이었다. 최종 Public LB는 1160.1624441988이었다.

## 9. 1160 이후 연구

대회 막판에는 Cascade, adaptive router, temporal mixture, dynamic state, low-rank graph, reconciliation, capacity sweep, Walsh/ECOC, continuation target 등 여러 구조를 실제 FULL 위에서 검증했다. 대부분 2022/23에서는 커 보였지만 2024에서 뒤집혔다. 이는 새로운 이름의 모델보다 **연도 전달과 강한 기준선 위 marginal gain**이 더 어려운 문제였음을 보여준다.
