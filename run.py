#!/usr/bin/env python
"""Command-line entry point for robot-to-robot handover planning."""

import argparse
import time


ROBOT_1 = "robot1"
ROBOT_2 = "robot2"


def parse_vector(value):
    import numpy as np

    return np.array([float(item) for item in value.split()])


def load_environment(problem):
    if problem == "vrep":
        from envs.vrep import VrepWrapper

        return VrepWrapper()

    from envs.collisions import PolygonEnvironment

    environment = PolygonEnvironment()
    environment.read_env(problem)
    return environment


def planner_dimensions(environment):
    return environment.lims.shape[0]


def build_rrt(environment, args, robot_name):
    from planners.rrt import RRT

    return RRT(
        num_samples=args.num_samples,
        num_dimensions=planner_dimensions(environment),
        step_length=args.step_length,
        lims=environment.lims,
        connect_prob=args.connect_prob,
        collision_func=environment.test_collisions,
        radius=args.radius,
        name=robot_name,
    )


def build_prm(environment, args, robot_name):
    from planners.prm import PRM

    return PRM(
        num_samples=args.num_samples,
        num_dimensions=planner_dimensions(environment),
        lims=environment.lims,
        local_planner=args.local_planner,
        collision_func=environment.test_collisions,
        connect_prob=args.connect_prob,
        radius=args.radius,
        epsilon=args.step_length,
        name=robot_name,
    )


def apply_start_goal_overrides(environment, args):
    if bool(args.start) != bool(args.goal):
        raise ValueError("--start and --goal must be provided together")

    if args.start is None:
        return environment.start, environment.goal

    start = parse_vector(args.start)
    goal = parse_vector(args.goal)
    environment.start = start
    environment.goal = goal
    return start, goal


def print_run_header(args, start, goal):
    print(
        f"""
[START]--------------------
method: {args.method}
problem: {args.problem}
#samples: {args.num_samples}
step length: {args.step_length}
goal bias %: {args.connect_prob}
connect: {args.connect}
bidirection: {args.bidirection}
rrt*: {args.star}
local planner: {args.local_planner}
radius: {args.radius}
start-goal: ({start})-({goal})
---------------------------
"""
    )


def print_plan(label, plan):
    print(f"{label}:")
    if not plan:
        print(None)
        return

    for point in plan:
        print(tuple(f"{value:+2.02f}" for value in point))


def draw_plan(environment, plan1, plan2, planner1, planner2, args):
    if args.no_draw and (args.problem == "vrep" or args.no_save):
        return

    kwargs = {
        "plan1": plan1,
        "plan2": plan2,
        "planner1": planner1,
        "planner2": planner2,
        "dynamic_tree": False,
        "dynamic_plan": not args.no_draw,
        "show": not args.no_draw,
    }

    if args.problem != "vrep":
        kwargs["save"] = not args.no_save

    environment.draw_plan(**kwargs)


def run_rrt(environment, args):
    model1 = build_rrt(environment, args, ROBOT_1)
    model2 = build_rrt(environment, args, ROBOT_2)

    environment.set_start_goal_config()

    if args.connect:
        plan1 = model1.build_rrt_connect(environment.robot1.start, environment.robot1.goal)
        plan2 = model2.build_rrt_connect(environment.robot2.start, environment.robot2.goal)
    elif args.bidirection:
        plan1 = model1.build_bidirectional_rrt_connect(
            environment.robot1.start,
            environment.robot1.goal,
        )
        plan2 = model2.build_bidirectional_rrt_connect(
            environment.robot2.start,
            environment.robot2.goal,
        )
    elif args.star:
        plan1 = model1.build_rrt_star(environment.robot1.start, environment.robot1.goal)
        plan2 = model2.build_rrt_star(environment.robot2.start, environment.robot2.goal)
    else:
        plan1 = model1.build_rrt(environment.robot1.start, environment.robot1.goal)
        plan2 = model2.build_rrt(environment.robot2.start, environment.robot2.goal)

    print_plan("robot1's plan", plan1)
    print_plan("robot2's plan", plan2)
    draw_plan(environment, plan1, plan2, model1, model2, args)


def run_prm(environment, args):
    model1 = build_prm(environment, args, ROBOT_1)
    model2 = build_prm(environment, args, ROBOT_2)

    environment.set_start_goal_config()

    print("Building PRM")
    model1.build_prm()
    model2.build_prm()

    print("Finding plan")
    local_plan_1, plan_1, _ = model1.query(environment.robot1.start, environment.robot1.goal)
    local_plan_2, plan_2, _ = model2.query(environment.robot2.start, environment.robot2.goal)

    print(
        "robot1's plan:",
        [tuple(round(value, 2) for value in point) for point in plan_1]
        if plan_1
        else None,
    )
    print(
        "robot2's plan:",
        [tuple(round(value, 2) for value in point) for point in plan_2]
        if plan_2
        else None,
    )

    draw_plan(environment, plan_1, plan_2, model1, model2, args)

    if args.local_planner == "rrt":
        draw_plan(environment, local_plan_1, local_plan_2, model1, model2, args)


def main(args):
    if bool(args.start) != bool(args.goal):
        raise ValueError("--start and --goal must be provided together")

    if args.problem == "vrep":
        args.step_length = 0.05

    environment = load_environment(args.problem)
    start, goal = apply_start_goal_overrides(environment, args)

    print_run_header(args, start, goal)
    start_time = time.time()

    if args.method == "prm":
        run_prm(environment, args)
    elif args.method == "rrt":
        run_rrt(environment, args)

    print("run_time =", time.time() - start_time)

    if args.problem == "vrep":
        environment.vrepReset()
        time.sleep(10)
        environment.vrepStop()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run PRM or RRT based method with a specified environment file"
    )
    parser.add_argument(
        "--method",
        choices=("rrt", "prm"),
        default="rrt",
        help="Method to use (default: rrt)",
    )
    parser.add_argument(
        "--problem",
        default="envs/env.txt",
        help="Path to the environment file, or 'vrep' for CoppeliaSim (default: envs/env.txt)",
    )

    rrt_variant = parser.add_mutually_exclusive_group()
    rrt_variant.add_argument("--connect", action="store_true", help="Use RRT-Connect")
    rrt_variant.add_argument(
        "--bidirection",
        action="store_true",
        help="Use Bidirectional RRT-Connect",
    )
    rrt_variant.add_argument("--star", action="store_true", help="Use RRT*")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to use in the planner (default: 5000)",
    )
    parser.add_argument(
        "--step_length",
        type=float,
        default=0.15,
        help="Step length to use in the planner (default: 0.15)",
    )
    parser.add_argument(
        "--connect_prob",
        type=float,
        default=0.05,
        help="Goal connection probability (default: 0.05)",
    )
    parser.add_argument(
        "--local_planner",
        choices=("line", "rrt"),
        default="line",
        help="Local planner to use in PRM (default: line)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Radius to use in PRM/RRT* (default: 2.0)",
    )
    parser.add_argument(
        "--start",
        help='Start position for the planner, for example: "75 50"',
    )
    parser.add_argument(
        "--goal",
        help='Goal position for the planner, for example: "10 20"',
    )
    parser.add_argument(
        "--no_draw",
        action="store_true",
        help="Skip interactive display",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save 2D frames or assets/robot_motion.gif",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    try:
        main(parser.parse_args())
    except ValueError as exc:
        parser.error(str(exc))
