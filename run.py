# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:17:29 2019
Edited on Fri Mar 21 23:59:59 2025

@author: tabor
@editor: seongil
"""
from collisions import PolygonEnvironment
import time
# import vrepWrapper
from rrt import *
from prm import *
import argparse


def main(args):
    # load arguments
    method = args.method
    problem = args.problem
    num_samples = args.num_samples
    step_length = args.step_length
    connect_prob = args.connect_prob
    connect = args.connect
    bidirection = args.bidirection
    local_planner = args.local_planner
    radius = args.radius
    start = np.array([float(i) for i in args.start.split()]) if args.start else None
    goal = np.array([float(i) for i in args.goal.split()]) if args.goal else None

    # load problem
    if problem == "vrep":
        environment = vrepWrapper.vrepWrapper()
        step_length = 0.1
    else:
        environment = PolygonEnvironment()
        environment.read_env(problem)

    dims = len(environment.start)

    # set start and goal using argmnuet
    if start is not None:
        environment.start = start
        environment.goal = goal

    print(
        f"""
[START]--------------------
method: {method}
problem: {problem}
#samples: {num_samples}
step lenght: {step_length} 
goal bias %: {connect_prob} 
connect: {connect}
bidirection: {bidirection}
local planner: {local_planner}
radius: {radius}
start-goal: ({start})-({goal})
---------------------------
"""
    )

    start_time = time.time()

    # PRM
    if method == "prm":
        # Set Local Planner
        if local_planner == "line":
            local_planner = StraightLinePlanner(
                step_length, environment.test_collisions
            )
        elif local_planner == "rrt":
            local_planner = RRT(
                num_samples=num_samples,
                num_dimensions=dims,
                step_length=step_length,
                lims=environment.lims,
                connect_prob=0.5,
                collision_func=environment.test_collisions,
            )
        # Set Model
        model = PRM(
            num_samples=num_samples,
            local_planner=local_planner,
            num_dimensions=dims,
            radius=radius,
            epsilon=step_length,
            lims=environment.lims,
            collision_func=environment.test_collisions,
        )

        print("Builing PRM")
        model.build_prm()
        build_time = time.time() - start_time
        print("Build time", build_time)

        starts, goals = [], []
        starts.append(environment.start)
        goals.append(environment.goal)

        print("Finding Plan")

        # Get new start & goal points
        while True:
            INPUT = input("Input Start and Goal (ex: -50 50 / 75 80 or exit): ")

            if not INPUT or INPUT.strip() == "exit":
                break
            
            start, goal = ([float(p) for p in point.split()] for point in INPUT.split("/"))
            starts.append(start)
            goals.append(goal)
        for start, goal in zip(starts, goals):
            time_stamp = time.time()
            local_plan, plan, visited = model.query(start, goal)

            print(
                "plan:",
                [tuple(round(v, 2) for v in point) for point in plan] if plan else None,
            )
            print("plan_time =", time.time() - time_stamp)

            environment.start, environment.goal = start, goal

            # draw plan (global)
            environment.draw_plan(
                plan=plan,
                planner=model,
                dynamic_tree=False,
                dynamic_plan=True,
                show=True,
            )

            # draw local plan
            if local_planner.__class__.__name__ == "RRT":
                environment.draw_plan(
                    plan=local_plan,
                    planner=model,
                    dynamic_tree=False,
                    dynamic_plan=True,
                    show=True,
                )

        run_time = time.time() - start_time
        print("run_time =", run_time)

    # RRT
    elif method == "rrt":
        # Load Model
        model = RRT(
            num_samples=num_samples,
            num_dimensions=dims,
            step_length=step_length,
            lims=environment.lims,
            connect_prob=connect_prob,
            collision_func=environment.test_collisions,
        )
        if connect:
            plan = model.build_rrt_connect(environment.start, environment.goal)
        elif bidirection:
            plan = model.build_bidirectional_rrt_connect(environment.start, environment.goal)
        else:
            plan = model.build_rrt(environment.start, environment.goal)

        run_time = time.time() - start_time
        print(
            "plan:",
            [tuple(round(v, 2) for v in point) for point in plan] if plan else None,
        )
        print("run_time =", run_time)

        # Draw plan
        environment.draw_plan(
            plan=plan, planner=model, dynamic_tree=False, dynamic_plan=True, show=True
        )

    if problem == "vrep":
        environment.vrepReset()
        time.sleep(10)
        environment.vrepStop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PRM or RRT based method with a specified environment file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rrt",
        help="Method to use (default: rrt, options: rrt, prm)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="env0.txt",
        help="Path to the environment file (default: env0.txt, options: env0.txt, env1.txt, vrep)",
    )
    parser.add_argument(
        "--connect", 
        action="store_true", 
        help="Use RRT-Connect"
    )
    parser.add_argument(
        "--bidirection", action="store_true", help="Use Bidirectional RRT-Connect"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to use in the RRT (default: 5000)",
    )
    parser.add_argument(
        "--step_length", type=float, default=2, help="Step length to use in the RRT"
    )
    parser.add_argument(
        "--connect_prob",
        type=float,
        default=0.05,
        help="Connection probability for RRT-Connect (default: 0.05)",
    )
    parser.add_argument(
        "--local_planner",
        type=str,
        default="line",
        help="Local planner to use in PRM (default: line, options: line, rrt)",
    )
    parser.add_argument(
        "--radius", 
        type=float, 
        default=2.0, 
        help="Radius to use in PRM (default: 2.0)"
    )
    parser.add_argument(
        "--start", 
        type=str, 
        help="Start position for the planner (default: None,  ex: `75 50`)"
    )
    parser.add_argument(
        "--goal", 
        type=str,
         help="Goal position for the planner (default: None, ex: `10 20`)"
    )

    args = parser.parse_args()

    main(args)
