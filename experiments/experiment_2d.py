"""Benchmark planners in the 2D polygon environment."""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from envs.collisions import PolygonEnvironment
from planners.prm import PRM
from planners.rrt import RRT, _COLLISION, _NEAR, _NEIGH, _TOTAL


VARIANT_FILENAMES = {
    "rrt": "results_rrt.csv",
    "connect": "results_connect.csv",
    "bidirectional": "results_bi.csv",
    "star": "results_star.csv",
    "prm": "results_prm.csv",
}


def sample_counts():
    return list(range(100, 1000, 100)) + list(range(1000, 5001, 1000))


def path_length(path):
    if path is None:
        return 0.0

    return sum(np.linalg.norm(np.asarray(path[i]) - np.asarray(path[i + 1])) for i in range(len(path) - 1))


def load_environment(env_file):
    environment = PolygonEnvironment()
    environment.read_env(env_file)
    environment.set_start_goal_config()
    return environment


def build_rrt(environment, num_samples, robot_name):
    return RRT(
        num_samples=num_samples,
        num_dimensions=environment.lims.shape[0],
        step_length=1,
        lims=environment.lims,
        connect_prob=0.05,
        collision_func=environment.test_collisions,
        radius=2,
        name=robot_name,
    )


def build_prm(environment, num_samples, robot_name):
    return PRM(
        num_samples=num_samples,
        num_dimensions=environment.lims.shape[0],
        local_planner="line",
        lims=environment.lims,
        collision_func=environment.test_collisions,
        connect_prob=0.05,
        radius=2,
        epsilon=1,
        name=robot_name,
    )


def run_rrt_trial(num_samples, env_file, variant):
    environment = load_environment(env_file)
    model1 = build_rrt(environment, num_samples, "robot1")
    model2 = build_rrt(environment, num_samples, "robot2")

    if variant == "rrt":
        plan1 = model1.build_rrt(environment.robot1.start, environment.robot1.goal)
        plan2 = model2.build_rrt(environment.robot2.start, environment.robot2.goal)
    elif variant == "connect":
        plan1 = model1.build_rrt_connect(environment.robot1.start, environment.robot1.goal)
        plan2 = model2.build_rrt_connect(environment.robot2.start, environment.robot2.goal)
    elif variant == "bidirectional":
        plan1 = model1.build_bidirectional_rrt_connect(environment.robot1.start, environment.robot1.goal)
        plan2 = model2.build_bidirectional_rrt_connect(environment.robot2.start, environment.robot2.goal)
    elif variant == "star":
        plan1 = model1.build_rrt_star(environment.robot1.start, environment.robot1.goal)
        plan2 = model2.build_rrt_star(environment.robot2.start, environment.robot2.goal)
    else:
        raise ValueError(f"Unknown RRT variant: {variant}")

    return {
        "r1_collision": model1.time_table[_COLLISION],
        "r1_near": model1.time_table[_NEAR],
        "r1_neighbors": model1.time_table[_NEIGH],
        "r1_total": model1.time_table[_TOTAL],
        "r1_path": path_length(plan1),
        "r2_collision": model2.time_table[_COLLISION],
        "r2_near": model2.time_table[_NEAR],
        "r2_neighbors": model2.time_table[_NEIGH],
        "r2_total": model2.time_table[_TOTAL],
        "r2_path": path_length(plan2),
    }


def run_prm_trial(num_samples, env_file):
    environment = load_environment(env_file)
    model1 = build_prm(environment, num_samples, "robot1")
    model2 = build_prm(environment, num_samples, "robot2")

    start_time = time.time()
    model1.build_prm()
    r1_build = time.time() - start_time

    start_time = time.time()
    model2.build_prm()
    r2_build = time.time() - start_time

    _, plan1, _ = model1.query(environment.robot1.start, environment.robot1.goal)
    _, plan2, _ = model2.query(environment.robot2.start, environment.robot2.goal)

    return {
        "r1_build": r1_build,
        "r1_collision": 0.0,
        "r1_near": 0.0,
        "r1_neighbors": 0.0,
        "r1_total": r1_build,
        "r1_path": path_length(plan1),
        "r2_build": r2_build,
        "r2_collision": 0.0,
        "r2_near": 0.0,
        "r2_neighbors": 0.0,
        "r2_total": r2_build,
        "r2_path": path_length(plan2),
    }


def run_trial(num_samples, env_file, variant):
    if variant == "prm":
        return run_prm_trial(num_samples, env_file)
    return run_rrt_trial(num_samples, env_file, variant)


def benchmark_variant(num_samples, env_file, variant):
    rows = []
    for sample_count in num_samples:
        row = {"num_samples": sample_count}
        row.update(run_trial(sample_count, env_file, variant))
        rows.append(row)
        print(f"{variant}: num_samples={sample_count}")

    return rows


def write_variant_csv(rows, output_dir, variant):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / VARIANT_FILENAMES[variant]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path, df


def build_parser():
    parser = argparse.ArgumentParser(description="Run 2D benchmark sweeps")
    parser.add_argument("--env", default="envs/env.txt", help="Environment file")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_FILENAMES),
        default=tuple(VARIANT_FILENAMES),
        help="Planner variants to benchmark",
    )
    parser.add_argument(
        "--output_dir",
        default="results/2d",
        help="Directory for generated CSV files",
    )
    parser.add_argument(
        "--sample_counts",
        nargs="+",
        type=int,
        help="Optional explicit sample counts for quick tests",
    )
    return parser


def main(args):
    counts = args.sample_counts if args.sample_counts else sample_counts()
    for variant in args.variants:
        rows = benchmark_variant(counts, args.env, variant)
        output_path, df = write_variant_csv(rows, args.output_dir, variant)
        print(output_path)
        print(df)


if __name__ == "__main__":
    main(build_parser().parse_args())
