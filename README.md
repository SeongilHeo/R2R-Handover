# Robot to Robot Handover (Team 8)
> ROBOT 6200 / CS 6370 Motion Planning, 2025 Spring

In this project, we aim to develop a system in which two robotic arms collaborate to pass an object from one to another efficiently and safely. The objective is to focus on two main aspects: grasp strategy and motion planning.

## Docs
- Proposal:  [Overleaf](https://www.overleaf.com/read/srddvwfsjmrw#7eb712).

## Requirements
```
python
...
```

## Files
```
.
├── README.md
├── env.txt
├── R2R.ttt
├── run.py
├── vrep.py
├── collisions.py
├── rrt.py
├── prm.py
└── robot_motion.gif
```

## Usage

python run.py [options]

### Options

| Option                          | Description                                                                                  |
|---------------------------------|----------------------------------------------------------------------------------------------|
| `-h`, `--help`                  | Show this help message and exit.                                                             |
| `--method METHOD`               | Method to use. (Default: `rrt`, Options: `rrt`, `prm`)                                       |
| `--problem PROBLEM`             | Path to the environment file. (Default: `env.txt`, Options: `env.txt`, `vrep`)               |
| `--connect`                     | Use RRT-Connect.                                                                             |
| `--bidirection`                 | Use Bidirectional RRT-Connect.                                                               |
| `--star`                        | Use RRT*.                                                                                    |
| `--num_samples NUM_SAMPLES`     | Number of samples to use in the RRT. (Default: `5000`)                                       |
| `--step_length STEP_LENGTH`     | Step length to use in the RRT.                                                               |
| `--connect_prob CONNECT_PROB`   | Connection probability for RRT-Connect. (Default: `0.05`)                                    |
| `--local_planner LOCAL_PLANNER` | Local planner for PRM. (Default: `line`, Options: `line`, `rrt`)                             |
| `--radius RADIUS`               | Radius used in PRM for connection. (Default: `2.0`)                                          |
<!-- | `--start START`                 | Start position. Format: `"x y"` (Example: `"75 50"`)                                         |
| `--goal GOAL`                   | Goal position. Format: `"x y"` (Example: `"10 20"`)                                          | -->


### Examples

1. **Check options:**
    ```sh
    $ python run.py --help
    ```
2. **Run on 2D Plannar:**
    ```sh
    $ python run.py --problem env.txt
    $ python run.py --problem vrep
    ```
3. **Run on 3D Coppeliasim:**

    1. Start Coppeliasim.
    2. File > Open Scence. `R2R2.ttt`
    3. Run model.

```sh
  $ python run.py --problem vrep
  ```


## Collaborator
-  Seongil Heo 
    - Computer Science
    - u1527760@utah.edu
- Jesse Jenkins
    - Computer Science and Computer Engineering
    - u1355879@utah.edu
- Cameron Monson
    - Mechanical Engineering
    - u1356158@utah.edu
- Corbin Gurnee
  - Computer Science
  - u1261969@utah.edu

