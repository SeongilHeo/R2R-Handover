#!/usr/bin/env python
'''
Package providing implementation of a probabilistic roadmap algorithm
'''

import random
import numpy as np
import matplotlib.pyplot as plotter
from collisions import PolygonEnvironment
import time
import heapq

from rrt import RRT

_DEBUG = False
_DEBUG_END = True

def fake_in_collision(q):
    '''
    We never collide with this function!
    '''
    return False

def euclidean_heuristic(self, s, goal):
    '''
    Euclidean heuristic function

    s - configuration vector
    goal - goal vector

    returns - floating point estimate of the cost to the goal from state s
    '''
    return np.linalg.norm(s - goal)

class PriorityQ:
    '''
    Priority queue implementation with quick access for membership testing
    Setup currently to only with the SearchNode class
    '''
    def __init__(self):
        '''
        Initialize an empty priority queue
        '''
        self.l = [] # list storing the priority q
        self.s = set() # set for fast membership testing

    def __contains__(self, x):
        '''
        Test if x is in the queue
        '''
        return x in self.s

    def push(self, x, cost):
        '''
        Adds an element to the priority queue.
        If the state already exists, we update the cost
        '''
        if tuple(x.state.tolist()) in self.s:
            return self.replace(x, cost)
        heapq.heappush(self.l, (cost, x))
        self.s.add(tuple(x.state.tolist()))

    def pop(self):
        '''
        Get the value and remove the lowest cost element from the queue
        '''
        x = heapq.heappop(self.l)
        self.s.remove(tuple(x[1].state.tolist()))
        return x[1]

    def peak(self):
        '''
        Get the value of the lowest cost element in the priority queue
        '''
        x = self.l[0]
        return x[1]

    def __len__(self):
        '''
        Return the number of elements in the queue
        '''
        return len(self.l)

    def replace(self, x, new_cost):
        '''
        Removes element x from the q and replaces it with x with the new_cost
        '''
        for y in self.l:
            if tuple(x.state.tolist()) == tuple(y[1].state.tolist()):
                self.l.remove(y)
                self.s.remove(tuple(y[1].state.tolist()))
                break
        heapq.heapify(self.l)
        self.push(x, new_cost)

    def get_cost(self, x):
        for y in self.l:
            if tuple(x.state.tolist()) == tuple(y[1].state.tolist()):
                return y[0]

    def __str__(self):
        return str(self.l)

def backpath(node):
    '''
    Function to determine the path that lead to the specified search node

    node - the SearchNode that is the end of the path

    returns - a tuple containing (path, action_path) which are lists respectively of the states
    visited from init to goal (inclusive) and the actions taken to make those transitions.
    '''
    maxPath = 100
    path = []
    while node.parent is not None and maxPath > 0:
        print(node.state)
        path.append(node.state)
        node = node.parent
        maxPath -= 1
    path.append(node.state)
    path.reverse()
    return path

class StraightLinePlanner:
    def __init__(self, step_size, collision_func = None):
        self.in_collision = collision_func
        self.epsilon = step_size
        if collision_func is None:
            self.in_collision = fake_in_collision

    def plan(self, start, goal):
        '''
        Check if edge is collision free, taking epsilon steps towards the goal
        Returns: None / False if edge in collsion
                 Plan / True if edge if free
        '''
        # return True
        # Get unit vector for the direction to go in
        dir = np.array(goal) - np.array(start)
        magnitude = np.linalg.norm(dir)
        if magnitude == 0:
            dir = np.zeros_like(dir)  # Return a zero vector if points are identical
        dir = dir / magnitude * self.epsilon

        point = np.array(start)
        # print("Starting")
        count = magnitude / self.epsilon
        while count >= 0:
            # print("Still going")
            # print(count)
            count -= 1
            if self.in_collision(point):
                return False
            point += dir
        return True
        raise NotImplementedError('Check if straight line edge between neighbours are collision free')


class RRTPlanner:
    def __init__(self, num_samples=500, step_length=2, env='./env0.txt'):
        pe = PolygonEnvironment()
        pe.read_env(env)

        dims = len(pe.start)
        start_time = time.time()
        
        self.rrt = RRT(num_samples,
                dims,
                step_length,
                lims = pe.lims,
                connect_prob = 0.05,
                collision_func=pe.test_collisions)

    def plan(self, start, goal):
        '''
        Check if edge is collision free, taking epsilon steps towards the goal
        Returns: None / False if edge in collsion
                 Plan / True if edge if free
        '''
        # return True
        self.planned = self.rrt.build_rrt(start, goal)
        if (self.planned[-1] == np.array(goal)).all():
            # print("Yahy!!")
            return True
        else:
            return False
    
        raise NotImplementedError('Check if straight line edge between neighbours are collision free')

class RoadMapNode:
    '''
    Nodes to be used in a built RoadMap class
    '''
    def __init__(self, state, cost=0, parent=None):
        self.state = np.array(state)
        self.neighbors = []
        self.cost = 0
        self.parent = parent

    def add_neighbor(self, n_new):
        '''
        n_new - new neighbor
        '''
        self.neighbors.append(n_new)

    def is_neighbor(self, n_test):
        '''
        Test if n_test is already our neighbor
        '''
        for n in self.neighbors:
            if np.linalg.norm(n.state - n_test.state) == 0.0:
                return True
        return False

    def __eq__(self, other):
        return np.linalg.norm(self.state - other.state) == 0.0

class RoadMap:
    '''
    Class to store a built roadmap for searching in our multi-query PRM
    '''
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node, neighbors):
        '''
        Add a node to the roadmap. Connect it to its neighbors
        '''
        # Avoid adding duplicates
        self.nodes.append(node)
        for n in neighbors:
            node.add_neighbor(n)
            if not n.is_neighbor(node):
                n.add_neighbor(node)
                self.edges.append((n.state, node.state))
                
    def get_states_and_edges(self):
        states = np.array([n.state for n in self.nodes])
        return (states, self.edges)

class PRM:
    def __init__(self, num_samples, local_planner, num_dimensions, lims = None,
                 collision_func = None, radius=2.0, epsilon=0.1, isRRT = False):
        self.local_planner = local_planner
        self.r = radius
        self.N = num_samples
        self.n = num_dimensions
        self.epsilon = epsilon
        self.isRRT = isRRT

        self.in_collision = collision_func
        if collision_func is None:
            self.in_collision = fake_in_collision

        # Setup range limits
        self.limits = lims
        if self.limits is None:
            self.limits = []
            for n in range(num_dimensions):
                self.limits.append([0,100])
            self.limits = np.array(self.limits)

        self.ranges = self.limits[:,1] - self.limits[:,0]

        # Build the roadmap instance
        self.T = RoadMap()

    def build_prm(self, reset=False):
        '''
        reset - empty the current roadmap if requested
        '''
        if reset:
            self.T = RoadMap()
        
        count = 0
        self.samples = []
        for s in range(self.N):
            if count == self.N/10:
                print(s/self.N, "%")
                count = 0
            count += 1
            sample = self.sample()

            if not self.in_collision(sample):
                randNode = RoadMapNode(sample)
                neighbors = self.find_valid_neighbors(randNode, self.samples, self.r)
                self.samples.append(randNode)
                self.T.add_node(randNode, neighbors)
                

            

        # raise NotImplementedError('Sample configurations and build a roadmap')

    def find_valid_neighbors(self, n_query, samples, r):
        '''
        Find the nodes that are close to n_query and can be attached by the local planner
        returns - list of neighbors reached by the local planner
        '''
        valid_neighbors = []
        for sample in samples:
            dir = np.array(sample.state) - np.array(n_query.state)
            if np.linalg.norm(dir) <= r:
                if self.local_planner.plan(sample.state, n_query.state):
                    valid_neighbors.append(sample)
        # raise NotImplementedError('Find samples withing radius r of n_query that is collision free')
        return valid_neighbors
    
    def query(self, start, goal):
        '''
        Generate a path from start to goal using the built roadmap
        returns - Path of configurations if in roadmap, None otherwise
        '''
        start_node = RoadMapNode(start)
        goal_node = RoadMapNode(goal)

        # Add neighbors for start node
        neighbors = self.find_valid_neighbors(start_node, self.samples, self.r)
        self.T.add_node(start_node, neighbors)

        # Add neighbors for goal node
        neighbors = self.find_valid_neighbors(goal_node, self.samples, self.r)
        self.T.add_node(goal_node, neighbors)

        # raise NotImplementedError('Attach start and goal node to the roadmap self.T')

        def is_goal(x):
            '''
            Test if a sample is at the goal
            '''
            return np.linalg.norm(x - goal) < self.epsilon

        # Run search on the roadmap to find a plan
        start_node.parent = None
        plan, visited = self.uniform_cost_search(start_node, is_goal)
        return plan, visited


    def uniform_cost_search(self, init_node, is_goal):
        '''
        Perform graph search on the roadmap
        '''
        cost = 0
        frontier = PriorityQ()
        frontier.push(init_node, cost)
        visited = set()
        #You need to modify your graph search from HW1 to expand neighbors instead of actions

        h = euclidean_heuristic  # (state, goal) -> float


        while len(frontier) > 0:
            n_i = frontier.pop()
            if tuple(n_i.state) not in visited:

                visited.add(tuple(n_i.state))
                if is_goal(n_i.state):
                    print("Found Goal!!")
                    print(n_i.state)
                    return (backpath(n_i), visited)
                else:

                    for neighbor in n_i.neighbors:
                        cost_add = np.linalg.norm(n_i.state - neighbor.state)
                        n_prime = neighbor
                        
                        if (not visited.__contains__(tuple(n_prime.state.tolist())) and (not frontier.__contains__(tuple(n_prime.state.tolist())) or frontier.get_cost(n_prime) > n_i.cost + cost_add)):
                            n_prime.cost = n_i.cost + cost_add
                            n_prime.parent = n_i
                            frontier.push(n_prime, n_prime.cost)
        print("Returning None")
        # return (([], []), visited.keys())


        # raise NotImplementedError('Add in your favorite optimal graph search from HW1')
        return None, visited
    
    def sample(self):
        '''
        Sample a new configuration
        Returns a configuration of size self.n bounded in self.limits
        '''

        dimentionsSample = []
        for n in range(self.n):
            # print("lower bounds: ", self.limits[n][0])
            # print("Upper bounds: ", self.limits[n][1])
            rand = random.uniform(self.limits[n][0], self.limits[n][1])
            dimentionsSample.append(rand)
        return dimentionsSample

        raise NotImplementedError('Sample a new configuration')


def saveFig(name,close = True):
    plotter.savefig(name + ".png")
    if(close):
        plotter.close()

  
def test_prm_env(num_samples=500, step_length=1, env='./env0.txt'):
    pe = PolygonEnvironment()
    pe.read_env(env)

    dims = len(pe.start)
    start_time = time.time()

    # For RRT Local Planner
    # local_planner = RRTPlanner(num_samples=num_samples, step_length=step_length, env=env)
    # For Straight Line Local Planner
    local_planner = StraightLinePlanner(step_length, pe.test_collisions)

    prm = PRM(num_samples,
              local_planner,
              dims,
              radius = 20,
              epsilon = step_length,
              lims = pe.lims,
              collision_func=pe.test_collisions,
              isRRT = True)
    print('Builing PRM')
    prm.build_prm()
    build_time = time.time() - start_time
    print('Build time', build_time)
    pe.draw_plan(None, prm,False,True,True)
    plotter.pause(3)
    print('Finding Plan')
    plan, visited = prm.query(pe.start, pe.goal)
    pe.draw_env(show=False)
    pe.draw_plan(plan, prm,False,True,True)

    run_time = time.time() - start_time
    print('plan:', plan)
    print('run_time =', run_time)


    # plotter.pause(3)

    # For Plan 2
    print('Finding Plan2')
    # env0
    pe.start = [-75, -40]
    pe.goal = [-75, 90]
    # env1
    # pe.start = [-0.5, 0.15, -0.3]
    # pe.goal = [0.4, 0.15, 0.3]
    plan, visited = prm.query(pe.start, pe.goal)
    pe.draw_env(show=False)
    pe.draw_plan(plan, prm,False,True,True)

    run_time = time.time() - start_time
    print('plan:', plan)
    print('run_time =', run_time)


    # plotter.pause(3)
    
    # For Plan 3
    print('Finding Plan3')
    # env0
    pe.start = [80, 25]
    pe.goal = [-20, -10]
    # env1
    # pe.start = [-2, 0.15, 0.3]
    # pe.goal = [-3, -1, -0.3]
    plan, visited = prm.query(pe.start, pe.goal)
    pe.draw_env(show=False)
    pe.draw_plan(plan, prm,False,True,True)

    run_time = time.time() - start_time
    print('plan:', plan)
    print('run_time =', run_time)


    return plan, prm, visited
if __name__== "__main__":
    test_prm_env()

    plotter.show(block=True)



# env0


# env1
# Start: -0.4 0.15 -0.3
# Goal: 0.4 0.15 0.3

# Start: -2 0.15 0.3
# Goal: -3 -1 -0.3