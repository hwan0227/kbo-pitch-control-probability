# 포트폴리오·면접용 요약

## 30초 설명

KBO 투구 한 행의 제구 성공 확률을 예측하는 대회에서 데이터 검증부터 모델 앙상블까지 전체 파이프라인을 구축했습니다. 투수·타자 이력과 TrackMan 물리 정보를 시간 누수 없이 결합했고, 이진 정답을 네 사건의 16개 joint class로 재표현한 Pattern16을 개발했습니다. 단독 점수보다 강한 기준선 위 marginal gain과 연도별 worst fold를 기준으로 실험을 관리해 Public LB를 1070.11에서 1160.16까지 높였습니다.

## 핵심 기여를 말하는 방식

- 약 147만 투구의 누적 통계·상태 일관성을 검사하고 cutoff-safe as-of feature pipeline 설계
- 공식 데이터만 사용한 TrackMan 선수 매핑과 물리·구종·산포 profile 구축
- LightGBM/XGBoost/CatBoost/RealMLP 및 binary/regression/multiclass target 다양화
- `reverse/middle/ball/strike`를 16-class joint target으로 만든 Pattern16 branch 설계
- 2022/23 개발 → 2024 seal, pitcher-disjoint fold, worst-fold gate를 포함한 실험 규율 정립
- 실패 실험까지 구조화하여 중복 연구를 조기에 중단하는 experiment bank 운영

## 이 프로젝트에서 배운 점

가장 큰 개선은 더 깊은 모델이나 더 많은 파생변수가 아니라 **정답을 보는 관점의 변경**에서 나왔다. 반면 로컬에서 좋아 보인 시간가중, subgroup correction, dynamic state는 실제 미래 연도나 리더보드에서 쉽게 깨졌다. 따라서 시계열 tabular 문제에서는 모델 복잡도보다 cutoff 설계, 독립적인 target representation, 강한 기준선 위 marginal 검증이 중요하다는 결론을 얻었다.

## 면접에서 강조할 점

1. 단순 모델 튜닝이 아니라 데이터 검증부터 최종 앙상블까지 전체 흐름을 구축했다.
2. TrackMan은 현재 투구의 정답 정보가 아니라 평가 시점 이전의 선수 프로필로만 사용했다.
3. 가장 큰 개선은 모델 크기보다 Pattern16이라는 새로운 정답 표현에서 나왔다.
4. 성공한 실험뿐 아니라 실패한 축도 같은 기준으로 기록하고 중단 조건을 정했다.
