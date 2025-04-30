# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:17:29 2019
Edited on Fri Mar 21 23:59:59 2025

@author: tabor
@editor: seongil
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from collisions import PolygonEnvironment
import time
from vrep import VrepWrapper
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
    star = args.star
    local_planner = args.local_planner
    radius = args.radius
    start = np.array([float(i) for i in args.start.split()]) if args.start else None
    goal = np.array([float(i) for i in args.goal.split()]) if args.goal else None

    # load problem
    if problem == "vrep":
        environment = VrepWrapper()
        step_length = 0.05
    else:
        environment = PolygonEnvironment()
        environment.read_env(problem)

    dims = len(environment.start)

    # set start and goal using argmnuet
    if start is not None:
        environment.start = start
        environment.goal = goal
    else:
        start = environment.start
        goal = environment.goal

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
        # Set Model
        model1 = PRM(
            num_samples=num_samples,
            num_dimensions=dims,
            lims=environment.lims,
            local_planner=local_planner,
            collision_func=environment.test_collisions,
            connect_prob=connect_prob,
            radius=radius,
            epsilon=step_length,
            name="robot1",
        )

        model2 = PRM(
            num_samples=num_samples,
            num_dimensions=dims,
            lims=environment.lims,
            local_planner=local_planner,
            collision_func=environment.test_collisions,
            connect_prob=connect_prob,
            radius=radius,
            epsilon=step_length,
            name="robot2",
        )


        environment.set_start_goal_config()

        print("Builing PRM")
        model1.build_prm()
        model2.build_prm()


        print("Finding Plan")
        local_plan_1, plan_1, _ = model1.query(environment.robot1.start, environment.robot1.goal)
        local_plan_2, plan_2, _ = model1.query(environment.robot2.start, environment.robot2.goal)

        print(
            "robot1's plan:",
            [tuple(round(v, 2) for v in point) for point in plan_1] if plan_1 else None,
        )
        print(
            "robot2's plan:",
            [tuple(round(v, 2) for v in point) for point in plan_2] if plan_2 else None,
        )

        # draw plan (global)
        environment.draw_plan(
            plan1=plan_1,
            plan2=plan_2,
            planner1=model1,
            planner2=model2,
            dynamic_tree=False,
            dynamic_plan=True,
            show=True,
        )

        # draw local plan
        if local_planner.__class__.__name__ == "RRT":
            environment.draw_plan(
                plan1=local_plan_1,
                plan2=local_plan_2,
                planner1=model1,
                planner2=model2,
                dynamic_tree=False,
                dynamic_plan=True,
                show=True,
            )
            
        run_time = time.time() - start_time
        print("run_time =", run_time)

    # RRT
    elif method == "rrt":
        # Load Model
        model1 = RRT(
            num_samples=num_samples,
            num_dimensions=dims,
            step_length=step_length,
            lims=environment.lims,
            connect_prob=connect_prob,
            collision_func=environment.test_collisions,
            name="robot1"
        )       
        model2 = RRT(
            num_samples=num_samples,
            num_dimensions=dims,
            step_length=step_length,
            lims=environment.lims,
            connect_prob=connect_prob,
            collision_func=environment.test_collisions,
            name="robot2"
        )

        environment.set_start_goal_config()

        if connect:
            plan_robot1 = model1.build_rrt_connect(environment.robot1.start, environment.robot1.goal)
            plan_robot2 = model2.build_rrt_connect(environment.robot2.start, environment.robot2.goal)
        elif bidirection:
            plan_robot1 = model1.build_bidirectional_rrt_connect(environment.robot1.start, environment.robot1.goal)
            plan_robot2 = model2.build_bidirectional_rrt_connect(environment.robot2.start, environment.robot2.goal)
        elif star:
            plan_robot1 = model1.build_rrt_star(environment.robot1.start, environment.robot1.goal)
            plan_robot2 = model2.build_rrt_star(environment.robot2.start, environment.robot2.goal)
        else:
            plan_robot1 = model1.build_rrt(environment.robot1.start, environment.robot1.goal)
            plan_robot2 = model2.build_rrt(environment.robot2.start, environment.robot2.goal)

        run_time = time.time() - start_time
        
        print("robot1's plan:")
        if plan_robot1:
            for point in plan_robot1:
                print(tuple(f"{v:+2.02f}" for v in point))
        else:
            print(None)

        print("robot2's plan:")
        if plan_robot2:
            for point in plan_robot2:
                print(tuple(f"{v:+2.02f}" for v in point))
        else:
            print(None)

        # Draw plan
        environment.draw_plan(
            plan1=plan_robot1, 
            plan2=plan_robot2, 
            planner1=model1, 
            planner2=model2, 
            dynamic_tree=False, 
            dynamic_plan=True, 
            show=True
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
        default="env.txt",
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
        "--star", action="store_true", help="Use RRT*"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to use in the RRT (default: 5000)",
    )
    parser.add_argument(
        "--step_length", type=float, default=0.15, help="Step length to use in the RRT"
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
