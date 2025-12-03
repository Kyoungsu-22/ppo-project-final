import os
from typing import Dict, List

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed


# ============================
# 1. Callback: 에피소드 리턴 로깅
# ============================

class EpisodeRewardCallback(BaseCallback):
    """
    에피소드가 끝날 때마다 reward 합계를 저장하는 콜백.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        # info dict 안에 'episode' 키가 있으면 에피소드 종료
        for info in self.locals.get("infos", []):
            if "episode" in info.keys():
                r = info["episode"]["r"]
                self.episode_rewards.append(r)
        return True


# ============================
# 2. 환경 생성 함수
# ============================

def make_env(env_id: str, seed: int = 0):
    """
    Monitor 래핑 + seed 설정된 Gymnasium 환경 생성.
    """
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init()


# ============================
# 3. 한 실험(config, env, seed)에 대한 학습 함수
# ============================

def train_single_run(
    env_id: str,
    config_name: str,
    config_params: Dict,
    seed: int,
    total_timesteps: int,
    log_dir: str
):
    """
    env_id: "CartPole-v1" 또는 "MountainCar-v0"
    config_name: "A_base", "B_entropy" 등 설정 이름
    config_params: PPO 하이퍼파라미터 dict
    seed: random seed
    """
    os.makedirs(log_dir, exist_ok=True)

    # 랜덤 시드 고정
    set_random_seed(seed)

    # 환경 생성
    env = make_env(env_id, seed=seed)

    # 콜백 설정
    callback = EpisodeRewardCallback()

    # PPO 모델 생성
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        **config_params,
        seed=seed,
    )

    # 학습
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    # ===== 모델 저장 =====
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f"{env_id}_config-{config_name}_seed-{seed}"
    )
    # 실제 파일명은 f"{model_path}.zip" 으로 저장됨
    model.save(model_path)

    env.close()

    # ===== episode reward 저장 =====
    rewards = np.array(callback.episode_rewards, dtype=np.float32)
    save_path = os.path.join(
        log_dir,
        f"{env_id}_config-{config_name}_seed-{seed}.npz"
    )
    np.savez(save_path, episode_rewards=rewards)
    print(
        f"[DONE] {env_id} | config={config_name} | seed={seed} "
        f"| episodes={len(rewards)} saved to {save_path}"
    )
    print(f"[MODEL SAVED] {model_path}.zip")


# ============================
# 4. 전체 실험 실행
# ============================

def main():
    # 실험 환경
    env_ids = ["CartPole-v1", "MountainCar-v0"]

    # 환경별 학습 스텝
    total_timesteps_per_env = {
        "CartPole-v1": 150_000,   # CartPole: 150k
        "MountainCar-v0": 500_000 # MountainCar: 500k
    }

    # PPO 하이퍼파라미터 설정들 (공통)
    base_params = dict(
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    configs = {
        "A_base": {  # 기본값
            **base_params
        },
        "B_entropy": {  # 탐색 강화
            **base_params,
            "ent_coef": 0.01,
        },
        "C_clip_small": {  # clip 범위 축소
            **base_params,
            "clip_range": 0.1,
        },
    }

    # seed 3개
    seeds = [0, 1, 2]
    log_dir = "./ppo_results"

    for env_id in env_ids:
        total_timesteps = total_timesteps_per_env[env_id]
        for config_name, params in configs.items():
            for seed in seeds:
                train_single_run(
                    env_id=env_id,
                    config_name=config_name,
                    config_params=params,
                    seed=seed,
                    total_timesteps=total_timesteps,
                    log_dir=log_dir,
                )


if __name__ == "__main__":
    main()