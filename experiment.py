import pandas as pd
from vrep import VrepWrapper
from rrt import *
from prm import *
import numpy as np

_COLLISION = 0
_NEAR = 1
_NEIGH = 2
_TOTAL = 3



def time_test(num_samples):
    environment = VrepWrapper()

    method = 'rrt'
    step_length = 0.1
    connect_prob = 0.05
    connect = False
    bidirection = False
    star = False
    radius = 0.1
    # local_planner = local_planner
    # start = np.array([float(i) for i in start.split()]) if start else None
    # goal = np.array([float(i) for i in goal.split()]) if goal else None
    dims = 3


    environment = VrepWrapper()

    model1 = RRT(
        num_samples=num_samples,
        num_dimensions=dims,
        step_length=step_length,
        lims=environment.lims,
        connect_prob=connect_prob,
        collision_func=environment.test_collisions,
        radius=radius,
        name="robot1"
    )       
    model2 = RRT(
        num_samples=num_samples,
        num_dimensions=dims,
        step_length=step_length,
        lims=environment.lims,
        connect_prob=connect_prob,
        collision_func=environment.test_collisions,
        radius=radius,
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

    return model1.time_table, model2.time_table, plan_robot1, plan_robot2

def compute_path(path):
    # Compute the path from the start to the goal
    if path is None:
        return 0

    dist = 0
    for i in range(len(path) - 1):
        start = path[i]
        goal = path[i + 1]

        dist+=np.linalg.norm(start-goal)  
    return dist

df = pd.DataFrame(columns=['r1_collision', 'r1_near', 'r1_neighbors', 'r1_total',  'r1_total',
                           'r2_collision', 'r2_near', 'r2_neighbors', 'r2_total'])

num_samples = list(range(100, 1000, 100))
rows = []  # Temporary list to store rows before concatenation

for num_sample in num_samples:
    t1, t2, p1, p2 = time_test(num_sample)
    
    # Assuming model1.time_table and model2.time_table are lists or arrays
    
    # Create a dictionary for the current row
    row = {
        'r1_collision': t1[_COLLISION],
        'r1_near': t1[_NEAR],
        'r1_neighbors': t1[_NEIGH],
        'r1_total': t1[_TOTAL],
        'r1_path': compute_path(p1),
        'r2_collision': t2[_COLLISION],
        'r2_near': t2[_NEAR],
        'r2_neighbors': t2[_NEIGH],
        'r2_total': t2[_TOTAL],
        'r2_path':  compute_path(p2),
    }
    rows.append(row)  # Add the row to the list

# Concatenate all rows into the DataFrame
df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

# Display the resulting DataFrame
df.to_csv('results.csv', index=False)

print(df)




