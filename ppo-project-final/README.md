# PPO를 이용한 CartPole-v1 & MountainCar-v0 실험

이 저장소는 **Stable-Baselines3의 PPO(Proximal Policy Optimization)** 알고리즘을 사용하여  
`CartPole-v1`과 `MountainCar-v0` 두 가지 환경에서 정책을 학습시키고,  
하이퍼파라미터 설정에 따른 성능 차이를 분석한 프로젝트입니다.

---

## 1. 프로젝트 개요

- **강화학습 알고리즘**: PPO (Stable-Baselines3)
- **환경**:  
  - CartPole-v1  
  - MountainCar-v0
- **정책 네트워크**: MLP (MlpPolicy)
- **실험 설정**:
  - 공통 기본 하이퍼파라미터 (base)
  - 엔트로피 보상 강화(`ent_coef=0.01`)
  - 클리핑 범위 축소(`clip_range=0.1`)
- **Seed**: 0, 1, 2 (총 3개 시드에 대해 반복)

각 실험에 대해:
- 에피소드별 리턴(episode return)을 기록하여 `.npz` 파일로 저장
- 학습된 PPO 모델을 `.zip` 파일로 저장
- 이후 스크립트를 통해 학습 곡선 및 최종 성능을 시각화

---

## 2. 폴더 및 파일 구조

```text
.
├── README.md
├── requirements.txt
├── train_ppo.py           # PPO 학습 및 모델/리턴 저장
├── analyze_results.py     # 결과 로드, 학습 곡선 및 성능 비교 그래프 생성
├── run_trained_model.py   # 저장된 모델 로드 후 환경에서 실행 데모
├── ppo_results/           # 학습 결과(.npz) 및 모델(.zip)이 저장되는 폴더
│   └── models/
└── ppo_plots/             # 학습 곡선 및 성능 비교 이미지(.png) 저장 폴더
