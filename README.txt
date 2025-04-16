# CS6370 Motion Planning - HW2

- Name: Seongil Heo
- UID: u1527760
- Date: Mar 24, 2025

## Files
```
.
├── README.txt
├── collisions.py
├── env0.txt
├── env1.txt
├── lbr4_testscript.py
├── prm.py
├── rrt.py
├── run.py
├── vrepWrapper.py
└── vrepfiles
    └── vrepfiles
        ├── ...
        └── vrepConst.py
```

## Usage

python run.py [options]

### Options

| Option                          | Description                                                                                  |
|---------------------------------|----------------------------------------------------------------------------------------------|
| `-h`, `--help`                  | Show this help message and exit.                                                             |
| `--method METHOD`               | Method to use. (Default: `rrt`, Options: `rrt`, `prm`)                                       |
| `--problem PROBLEM`             | Path to the environment file. (Default: `env0.txt`, Options: `env0.txt`, `env1.txt`, `vrep`) |
| `--connect`                     | Use RRT-Connect.                                                                             |
| `--bidirection`                 | Use Bidirectional RRT-Connect.                                                               |
| `--num_samples NUM_SAMPLES`     | Number of samples to use in the RRT. (Default: `5000`)                                       |
| `--step_length STEP_LENGTH`     | Step length to use in the RRT.                                                               |
| `--connect_prob CONNECT_PROB`   | Connection probability for RRT-Connect. (Default: `0.05`)                                    |
| `--local_planner LOCAL_PLANNER` | Local planner for PRM. (Default: `line`, Options: `line`, `rrt`)                             |
| `--radius RADIUS`               | Radius used in PRM for connection. (Default: `2.0`)                                          |
| `--start START`                 | Start position. Format: `"x y"` (Example: `"75 50"`)                                         |
| `--goal GOAL`                   | Goal position. Format: `"x y"` (Example: `"10 20"`)                                          |


### Examples

1. **Check options:**
    ```
    python run.py --help
    ```
2. **Run RRT on env0.txt:**
    ```
    python run.py --method rrt --problem env0.txt
    ```
3. **Run RRT connect on env0.txt:**
    ```
    python run.py --method rrt --connect --problem env0.txt
    ```
4. **Run bidirectional RRT-Connect on env1.txt with step length 0.15:**
    ```
    python run.py --method rrt --bidirection --problem env1.txt --step_length 0.15
    ```
5. **Specify start and goal positions:**
    ```
    python run.py --method rrt --problem env0.txt --start "75 50" --goal "0 0"
    ```
6. **Run PRM on env0.txt with radius 20 and 300 samples:**
    ```
    python run.py --method prm --problem env0.txt --radius 20 --num_samples 300
    ```
7. **Run PRM with RRT as a local planner**:
    ```
    python run.py --method prm --local_planner rrt --problem env0.txt --radius 20 --num_samples 300 --step_length 2
    ```
8. **Run RRT on coppelia simulater:**
    ```
    python run.py --method rrt --problem vrep
    ```
9. **Run RRT connect with sample 10000 and 2 step length:**
    ```
    python run.py --method rrt --connect --num_samples 10000 --step_length 2
    ```
10. **Run RRT with 5% goal bias (goal connection probability):**
    ```
    python run.py --method rrt --problem env0.txt --connect_prob 0.05
    ```

