import os
import glob
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = "./ppo_results"
SAVE_DIR = "./ppo_plots"


def load_rewards(env_id: str, config_name: str) -> List[np.ndarray]:
    """
    env_id와 config_name에 해당하는 seed별 episode reward 배열들을 불러온다.
    파일 패턴: {env_id}_config-{config_name}_seed-*.npz
    """
    pattern = os.path.join(
        RESULTS_DIR,
        f"{env_id}_config-{config_name}_seed-*.npz"
    )
    files = sorted(glob.glob(pattern))
    rewards_list = []
    for f in files:
        data = np.load(f)
        rewards = data["episode_rewards"]
        rewards_list.append(rewards)
    return rewards_list


def pad_and_stack(rewards_list: List[np.ndarray]) -> np.ndarray:
    """
    각 seed마다 episode 수가 다를 수 있으므로,
    최대 길이에 맞춰 NaN padding 후 (num_seeds, max_episodes) 배열로 쌓는다.
    """
    max_len = max(len(r) for r in rewards_list)
    arr = np.full((len(rewards_list), max_len), np.nan, dtype=np.float32)
    for i, r in enumerate(rewards_list):
        arr[i, :len(r)] = r
    return arr


def moving_average(x: np.ndarray, window: int = 20) -> np.ndarray:
    """
    간단한 이동 평균 (NaN 무시).
    x: (num_seeds, num_episodes)
    """
    out = np.copy(x)
    for i in range(x.shape[0]):
        y = x[i]
        valid_idx = ~np.isnan(y)
        if valid_idx.sum() == 0:
            continue
        vals = y[valid_idx]
        if len(vals) < window:
            out[i, valid_idx] = vals
            continue
        cumsum = np.cumsum(np.insert(vals, 0, 0))
        smoothed = (cumsum[window:] - cumsum[:-window]) / window
        # 길이 맞춰 붙이기
        front = vals[: window - 1]
        smoothed_full = np.concatenate([front, smoothed])
        out[i, valid_idx] = smoothed_full
    return out


def plot_env(env_id: str, configs: Dict[str, str], window: int = 20):
    """
    한 env_id(CartPole or MountainCar)에 대해
    config별 학습곡선을 그린다.

    - 얇은 선: seed별 학습곡선 (smoothed)
    - 굵은 선: config별 seed 평균
    - 그림자 영역: 평균 ± 1 표준편차
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    plt.figure()
    ax = plt.gca()

    # 색상 사이클 가져오기
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (config_name, label) in enumerate(configs.items()):
        rewards_list = load_rewards(env_id, config_name)
        if len(rewards_list) == 0:
            print(f"[WARN] No data for {env_id}, config={config_name}")
            continue

        arr = pad_and_stack(rewards_list)
        arr_smooth = moving_average(arr, window=window)

        mean = np.nanmean(arr_smooth, axis=0)
        std = np.nanstd(arr_smooth, axis=0)
        episodes = np.arange(len(mean))

        color = colors[idx % len(colors)]

        # seed별 곡선 (얇고 투명하게)
        for i in range(arr_smooth.shape[0]):
            ax.plot(
                episodes,
                arr_smooth[i],
                alpha=0.3,
                linewidth=1.0,
                color=color
            )

        # 평균 곡선 (굵게)
        ax.plot(
            episodes,
            mean,
            label=label,
            linewidth=2.5,
            color=color
        )

        # 신뢰 대역 (±1 std)
        ax.fill_between(
            episodes,
            mean - std,
            mean + std,
            alpha=0.15,
            color=color
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title(f"PPO Learning Curves: {env_id}")
    ax.legend()
    ax.grid(True)

    save_path = os.path.join(SAVE_DIR, f"{env_id}_ppo_learning_curves.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[SAVED] {save_path}")
    plt.close()


def compute_final_scores(
    env_id: str,
    config_name: str,
    last_n: int
) -> np.ndarray:
    """
    각 seed별로 '마지막 last_n 에피소드' 평균 리턴을 계산해서
    (num_seeds,) 배열로 반환.
    """
    rewards_list = load_rewards(env_id, config_name)
    scores = []
    for r in rewards_list:
        if len(r) == 0:
            continue
        if len(r) < last_n:
            score = float(r.mean())
        else:
            score = float(r[-last_n:].mean())
        scores.append(score)
    if len(scores) == 0:
        return np.array([], dtype=np.float32)
    return np.array(scores, dtype=np.float32)


def plot_final_performance(
    env_id: str,
    configs: Dict[str, str],
    last_n: int
):
    """
    환경 env_id에 대해 config별 최종 성능 bar chart를 그린다.
    - 각 bar: seed 평균 (마지막 last_n 에피소드 평균 리턴)
    - error bar: seed 간 표준편차

    추가로, env_id별 / config별 / seed별 점수를 콘솔에 출력한다.
    (여기 출력에서는 평균/표준편차는 따로 표시하지 않음.)
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    labels = []
    means = []
    stds = []

    print(f"\n==== Final scores for {env_id} (last {last_n} episodes) ====")

    for config_name, label in configs.items():
        scores = compute_final_scores(env_id, config_name, last_n=last_n)
        if scores.size == 0:
            print(f"[WARN] No scores for {env_id}, config={config_name}")
            continue

        # per-config / per-seed 점수 출력
        print(f"{env_id} | config={config_name}: per-seed scores = {scores}")

        labels.append(label)
        means.append(scores.mean())
        stds.append(scores.std())

    if len(labels) == 0:
        print(f"[WARN] No data to plot final performance for {env_id}")
        return

    x = np.arange(len(labels))

    plt.figure()
    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.8,
    )

    plt.xticks(x, labels, rotation=15)
    plt.ylabel(f"Mean return (last {last_n} episodes)")
    plt.title(f"Final Performance (last {last_n} episodes): {env_id}")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path = os.path.join(
        SAVE_DIR,
        f"{env_id}_ppo_final_performance_last{last_n}.png"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[SAVED] {save_path}")
    plt.close()


# 🔥 추가: env_id + config별로 seed 곡선만 따로 보는 상세 학습곡선
def plot_detailed_per_config(
    env_id: str,
    configs: Dict[str, str],
    window: int = 20
):
    """
    env_id별, config별로 seed의 세부 학습곡선을 별도 PNG로 저장.

    예시:
    - CartPole-v1 + A_base => CartPole-v1_config-A_base_per_seed_learning_curves.png
    - MountainCar-v0 + B_entropy => MountainCar-v0_config-B_entropy_per_seed_learning_curves.png
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    for config_name, label in configs.items():
        # 해당 env + config 조합의 seed별 파일 로딩
        pattern = os.path.join(
            RESULTS_DIR,
            f"{env_id}_config-{config_name}_seed-*.npz"
        )
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            print(f"[WARN] No data for detailed plot: {env_id}, config={config_name}")
            continue

        plt.figure()
        ax = plt.gca()

        for f in files:
            # 파일명에서 seed 추출 (예: ..._seed-0.npz)
            base = os.path.basename(f)
            # "seed-" 이후, ".npz" 이전
            try:
                seed_str = base.split("seed-")[1].split(".")[0]
            except Exception:
                seed_str = "?"

            data = np.load(f)
            rewards = data["episode_rewards"]

            # 1D -> (1, T)로 만들어서 기존 moving_average 재사용
            arr = rewards.reshape(1, -1)
            arr_smooth = moving_average(arr, window=window)
            curve = arr_smooth[0]

            episodes = np.arange(len(curve))

            ax.plot(
                episodes,
                curve,
                label=f"seed {seed_str}",
                linewidth=1.5
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode return")
        ax.set_title(
            f"PPO Learning Curves (per-seed)\n{env_id} | {label} ({config_name})"
        )
        ax.legend()
        ax.grid(True)

        save_path = os.path.join(
            SAVE_DIR,
            f"{env_id}_config-{config_name}_per_seed_learning_curves.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"[SAVED] {save_path}")
        plt.close()


def main():
    configs = {
        "A_base": "Base",
        "B_entropy": "Entropy (ent_coef=0.01)",
        "C_clip_small": "Clip small (clip=0.1)",
    }

    for env_id in ["CartPole-v1", "MountainCar-v0"]:
        # 1) env별 전체 학습곡선 (config 3개가 한 그림)
        plot_env(env_id, configs, window=20)

        # 2) env별 last_n 설정 분리
        if env_id == "CartPole-v1":
            last_n = 300   # CartPole: 마지막 300 에피소드
        else:
            last_n = 50    # MountainCar: 마지막 50 에피소드

        # 3) 최종 성능 bar chart + per-seed 점수 출력
        plot_final_performance(env_id, configs, last_n=last_n)

        # 4) 🔥 env_id + config별로 seed 곡선만 따로 보는 상세 학습곡선
        plot_detailed_per_config(env_id, configs, window=20)


if __name__ == "__main__":
    main()
