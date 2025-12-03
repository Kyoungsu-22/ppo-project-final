import os
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO


MODEL_DIR = "./ppo_results/models"


def run_trained_model(
    env_id: str,
    config_name: str,
    seed: int = 0,
    n_episodes: int = 5,
    render: bool = True,
):
    """
    저장된 PPO 모델을 불러와서 해당 환경에서 n_episodes 만큼 실행해보는 함수.

    env_id: "CartPole-v1" 또는 "MountainCar-v0"
    config_name: "A_base", "B_entropy", "C_clip_small"
    seed: 학습 때 사용했던 seed (0, 1, 2 중 선택)
    n_episodes: 실행할 에피소드 수
    render: True면 화면 렌더링 (윈도우 팝업)
    """
    model_path = os.path.join(
        MODEL_DIR,
        f"{env_id}_config-{config_name}_seed-{seed}.zip"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Gymnasium 환경 생성 (render_mode 설정)
    render_mode: Optional[str] = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    # 모델 로드 (환경도 같이 넘겨줌)
    model = PPO.load(model_path, env=env)

    print(f"=== Running {env_id} | config={config_name} | seed={seed} ===")
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            # deterministic=True 로 policy의 대표 행동을 사용
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Gymnasium에서 render_mode="human"이면 env.render()는 보통 필요 없지만,
            # 환경에 따라 필요할 수도 있어 한 번 더 호출
            if render:
                env.render()

            done = terminated or truncated

        print(
            f"[{env_id}] Episode {ep + 1}/{n_episodes} "
            f"| steps={step_count} | return={total_reward:.2f}"
        )

    env.close()
    print(f"=== Finished {env_id} | config={config_name} | seed={seed} ===\n")


def main():
    """
    CartPole-v1 / MountainCar-v0 두 환경에 대해
    학습된 모델을 실행해보는 간단한 데모.
    필요하면 config_name / seed / n_episodes 수정해서 사용.
    """
    # 데모용 설정 (원하는 config / seed 로 바꿔도 됨)
    demo_config_name = "B_entropy"  # "A_base", "B_entropy", "C_clip_small" 중 하나
    demo_seed = 0
    n_episodes = 3
    render = True

    # CartPole 데모
    run_trained_model(
        env_id="CartPole-v1",
        config_name=demo_config_name,
        seed=demo_seed,
        n_episodes=n_episodes,
        render=render,
    )

    # MountainCar 데모
    run_trained_model(
        env_id="MountainCar-v0",
        config_name=demo_config_name,
        seed=demo_seed,
        n_episodes=n_episodes,
        render=render,
    )


if __name__ == "__main__":
    main()
