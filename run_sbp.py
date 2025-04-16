import numpy as np
import matplotlib.pyplot as plotter
from math import pi
from collisions import PolygonEnvironment
import time
import argparse
from rrt import *
from prm import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', '-e', type=str)
    parser.add_argument('--algorithm', '-a', type=str, choices=['rrt', 'rrt-c', 'b-rrt', 'prm'])
    parser.add_argument('--num_samples', '-n', type=int, default=0)
    parser.add_argument('--step_length', '-step_length', type=float, default=0)
    parser.add_argument('--connect_prob', '-connect_prob', type=float, default=0.05)
    parser.add_argument('--start', '-start', nargs='+', type=float, default=[-999999])
    parser.add_argument('--goal', '-goal', nargs='+', type=float, default=[-999999])
    parser.add_argument('--local_planner', '-local_planner', type=str, choices=['straight', 'rrt'], default = 'straight')
    parser.add_argument('--radius', '-r', type=float, default=20)
    parser.add_argument('--dynamic_tree', '-dynamic_tree', type=bool, default = False)
    parser.add_argument('--dynamic_plan', '-dynamic_plan', type=bool, default = False)

    args = parser.parse_args()

    env = args.environment
    algorithm = args.algorithm
    num_samples = args.num_samples
    step_length = args.step_length
    connect_prob = args.connect_prob
    start = args.start
    goal = args.goal
    radius = args.radius
    dynamic_tree = args.dynamic_tree
    dynamic_plan = args.dynamic_plan

    environment = PolygonEnvironment()
    environment.read_env(env)

    if step_length == 0:
        if environment.start[0] == 0:
            step_length = 0.15
        else:
            step_length = 2

    if radius == 0:
        if environment.start[0] == 0:
            radius = 1
        else:
            radius = 20

    if num_samples == 0:
        if algorithm == 'prm':
            num_samples = 500
        else:
            num_samples = 5000
    
    dims = len(environment.start)

    if start[0] == -999999:
        start = environment.start
    else:
        environment.start = start
    if goal[0] == -999999:
        goal = environment.goal
    else:
        environment.goal = goal

    #if environment.test_collisions(start):
    #    print("Starting position is in collision")
    #if environment.test_collisions(goal):
    #    print("Goal position is in collision")

    start_time = time.time()

    if algorithm == 'prm':
        
        if args.local_planner == 'rrt':
            local_planner = RRTPlanner(step_length, dims, environment.lims, environment.test_collisions)
        else:
            local_planner = StraightLinePlanner(step_length, environment.test_collisions)

        sbp = PRM(num_samples, local_planner, dims, environment.lims, environment.test_collisions, radius, step_length)

        sbp.build_prm()
        plan, visited = sbp.query(start, goal)

    else:

        sbp = RRT(num_samples,
          dims,
          step_length,
          lims = environment.lims,
          connect_prob = connect_prob,
          collision_func=environment.test_collisions)
        
        if algorithm == 'rrt':
            plan = sbp.build_rrt(start, goal)
        elif algorithm == 'rrt-c':
            plan = sbp.build_rrt_connect(start, goal)
        elif algorithm == 'b-rrt':
            plan = sbp.build_bidirectional_rrt_connect(start, goal)

    run_time = time.time() - start_time
    print('plan:', plan)
    print('run_time =', run_time)

    if algorithm != 'prm':
        debug = environment.draw_plan(plan, sbp, dynamic_tree, dynamic_plan, True)
        plotter.show(block=True)
    else:
        environment.draw_env(show=False)
        environment.draw_plan(plan, sbp,False,dynamic_plan,True)
        plotter.show(block=True)