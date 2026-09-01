# 실패 실험과 배운 점

이 프로젝트의 포트폴리오 가치는 성공 모델 하나보다 **왜 수많은 그럴듯한 아이디어가 전달되지 않았는지 측정한 과정**에 있다.

| 축 | 로컬/2024 결과 | 결론 |
|---|---|---|
| Base-state correction | 로컬 약 +2.71, 실제 LB -4.63 | subgroup correction은 LB 전달 위험이 큼 |
| Whole-core temporal decay | 2024 +0.856, 5/5 양수, 실제 LB 약 -0.155 | 안정적인 한 해도 미래 전달 보장은 아님 |
| P16 Cascade | 최고 +0.334, 3/5 | flat joint model이 hierarchy보다 강했음 |
| Adaptive router | +0.017, worst -4.54 | 유연한 row-wise stacking은 과적합 |
| Dynamic latent state | 최고 -4.224, worst -13.74 | 2022/23의 큰 gain이 2024에서 역전 |
| Low-rank player graph | 최고 -0.453 | 관계 임베딩이 강한 player history를 이기지 못함 |
| P16 reconciliation | 최고 +0.052 | marginal event signal이 이미 P16에 흡수됨 |
| Core/P16/P48 capacity | 모두 2024 음수 | depth와 tree 수가 병목이 아니었음 |
| Walsh-Hadamard P16 | +0.156, 5/5 | 안정적이지만 너무 작고 개발 전달 실패 |
| Continuation value | 최고 -0.089 | state geometry가 기존 context와 중복 |

## 반복해서 나타난 패턴

1. **오래된 연도에서 큰 gain이 나도 최근 연도에서 뒤집힐 수 있다.**
2. **기존 FULL이 이미 강하면 standalone 대박은 대부분 압축된다.**
3. **모델 용량을 키우는 것보다 target representation을 바꾼 Pattern16이 훨씬 컸다.**
4. **작은 gain은 실제 제출 전에 더 강한 안정성 기준이 필요하다.**
5. **실패축을 닫는 명확한 stop rule이 GPU 시간보다 중요하다.**
