"""Benchmark planners in the CoppeliaSim environment."""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

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


def load_environment():
    from envs.vrep import VrepWrapper

    environment = VrepWrapper()
    environment.set_start_goal_config()
    return environment


def build_rrt(environment, num_samples, robot_name):
    return RRT(
        num_samples=num_samples,
        num_dimensions=environment.lims.shape[0],
        step_length=0.1,
        lims=environment.lims,
        connect_prob=0.05,
        collision_func=environment.test_collisions,
        radius=0.1,
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
        radius=0.1,
        epsilon=0.1,
        name=robot_name,
    )


def run_rrt_trial(num_samples, variant):
    environment = load_environment()
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


def run_prm_trial(num_samples):
    environment = load_environment()
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


def run_trial(num_samples, variant):
    if variant == "prm":
        return run_prm_trial(num_samples)
    return run_rrt_trial(num_samples, variant)


def benchmark_variant(num_samples, variant):
    rows = []
    for sample_count in num_samples:
        row = {"num_samples": sample_count}
        row.update(run_trial(sample_count, variant))
        rows.append(row)
        print(f"{variant}: num_samples={sample_count}")

    return rows


def trace_star(num_samples):
    environment = load_environment()
    model1 = build_rrt(environment, num_samples, "robot1")
    model2 = build_rrt(environment, num_samples, "robot2")

    _, records1 = model1.build_rrt_star_trace(environment.robot1.start, environment.robot1.goal)
    _, records2 = model2.build_rrt_star_trace(environment.robot2.start, environment.robot2.goal)
    return records1, records2


def write_variant_csv(rows, output_dir, variant):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / VARIANT_FILENAMES[variant]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path, df


def write_trace_csv(records1, records2, output_dir):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df1 = pd.DataFrame(records1).rename(
        columns={
            "collision": "r1_collision",
            "near": "r1_near",
            "neighbors": "r1_neighbors",
            "total": "r1_total",
            "path": "r1_path",
        }
    )
    df2 = pd.DataFrame(records2).rename(
        columns={
            "collision": "r2_collision",
            "near": "r2_near",
            "neighbors": "r2_neighbors",
            "total": "r2_total",
            "path": "r2_path",
        }
    )

    path1 = output_dir / "results_star_r1.csv"
    path2 = output_dir / "results_star_r2.csv"
    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)
    return path1, path2


def build_parser():
    parser = argparse.ArgumentParser(description="Run CoppeliaSim benchmark sweeps")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_FILENAMES),
        default=tuple(VARIANT_FILENAMES),
        help="Planner variants to benchmark",
    )
    parser.add_argument(
        "--output_dir",
        default="results/3d",
        help="Directory for generated CSV files",
    )
    parser.add_argument(
        "--trace_star",
        action="store_true",
        help="Also write RRT* per-iteration trace CSV files",
    )
    parser.add_argument(
        "--trace_samples",
        type=int,
        default=1000,
        help="Number of samples for --trace_star",
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
        rows = benchmark_variant(counts, variant)
        output_path, df = write_variant_csv(rows, args.output_dir, variant)
        print(output_path)
        print(df)

    if args.trace_star:
        records1, records2 = trace_star(args.trace_samples)
        path1, path2 = write_trace_csv(records1, records2, args.output_dir)
        print(path1)
        print(path2)


if __name__ == "__main__":
    main(build_parser().parse_args())
