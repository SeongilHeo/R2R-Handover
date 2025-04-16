You can run any algorithm from the command line by running run_sbp.py with these arguments:

- '-e': **Environment File** (required)
  - 'env0.txt'
  - 'env1.txt'

- '-a': **Algorithm** (required)
  - 'rrt' 
  - 'rrt-c' (RRT Connect)
  - 'b-rrt' (Bi-Directional RRT)
  - 'prm'

- '-n': **Number of Samples** (optional)
  default = 5000 for rrt, 500 for prm
  number of samples to stop expanding RRT tree or PRM roadmap

- '-step_length': **Step Length** (optional)
  default = 0.15 for env1, 2 for env0

  RRT step length or local planner step length in PRM

- '-connect_prob': **Sampling Bias** (optional)
  default = 0.05 (5%)
  probability of sampling the goal in RRT

- '-start': **Start Location** (optional)
  default = start provided by environment file
  
  otherwise enter as space separated values 
  e.g. -start -50 50

- '-goal': **Goal Location** (optional)
  default = goal provided by environment file

  otherwise enter as space separated values
  e.g. -goal 75 80

- '-local_planner': **Local Planner to use in PRM** (optional)
  - 'straight' (default) (straight line planner)
  - 'rrt' (rrt local planner)

- '-r': **Connect Radius** (optional)
  default = 20 for env0, 1 for env1

  Radius to try to connect nodes to in PRM

- '-dynamic_tree': **Draw the tree all at once or not** (optional)
  default = 0
  if 1, tree will be drawn node by node as it was built

- '-dynamic_plan': **Draw the plan all at once or not** (optional)
  default = 0
  if 1, plan will be drawn step by step


examples:

# basic rrt on env0, with default parameters
python run_sbp.py -e env0.txt -a rrt 

# rrt connect on env0, with default parameters but new goal and start locations
python run_sbp.py -e env0.txt -a rrt-c -start 0 0 -goal 90 140

# bi directional rrt on env0, with a step length of 5 and a goal sampling probability of 50%
python run_sbp.py -e env0.txt -a b-rrt -step_length 5 -connect_prob 0.5 

# prm on env1 taking 100 total samples and connecting within a radius of 1.5, default start and goal locations, default straight line planner
python run_sbp.py -e env1.txt -a prm -n 100 -r 1.5

  