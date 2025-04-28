#!/usr/bin/env python
'''
Package providing helper classes and functions for performing graph search operations for planning.
'''
import numpy as np
import matplotlib.pyplot as plotter
from math import pi
from collisions import PolygonEnvironment
import time
import random

_DEBUG = False

_TRAPPED = 'trapped'
_ADVANCED = 'advanced'
_REACHED = 'reached'

class TreeNode:
    '''
    Class to hold node state and connectivity for building an RRT
    '''
    def __init__(self, state, parent=None, cost=0.0):
        self.state = state
        self.children = []
        self.parent = parent

        self.cost = cost

    def add_child(self, child):
        '''
        Add a child node
        '''
        self.children.append(child)

class RRTSearchTree:
    '''
    Searh tree used for building an RRT
    '''
    def __init__(self, init):
        '''
        init - initial tree configuration
        '''
        self.root = TreeNode(init)
        self.nodes = [self.root]
        self.edges = []

    def find_nearest(self, s_query):
        '''
        Find node in tree closets to s_query
        returns - (nearest node, dist to nearest node)
        '''
        min_d = 1000000
        nn = self.root
        for n_i in self.nodes:
            d = np.linalg.norm(s_query - n_i.state)
            if d < min_d:
                nn = n_i
                min_d = d
        return (nn, min_d)

    def add_node(self, node, parent, cost=None):
        '''
        Add a node to the tree
        node - new node to add
        parent - nodes parent, already in the tree
        '''
        if cost is None:
            node.cost = getattr(parent, 'cost', 0.0) + np.linalg.norm(node.state - parent.state)
        else:
            node.cost = cost

        self.nodes.append(node)
        self.edges.append((parent.state, node.state))
        node.parent = parent
        parent.add_child(node)

    def get_states_and_edges(self):
        '''
        Return a list of states and edgs in the tree
        '''
        states = np.array([n.state for n in self.nodes])
        return (states, self.edges)

    def get_back_path(self, n):
        '''
        Get the path from the root to a specific node in the tree
        n - node in tree to get path to
        '''
        path = []
        while n.parent is not None:
            path.append(n.state)
            n = n.parent
        path.append(n.state)
        path.reverse()
        return path

class RRT(object):
    '''
    Rapidly-Exploring Random Tree Planner
    '''
    def __init__(self, num_samples, num_dimensions=2, step_length = 1, lims = None,
                 connect_prob = 0.05, collision_func=None):
        '''
        Initialize an RRT planning instance
        '''
        self.K = num_samples
        self.n = num_dimensions
        self.epsilon = step_length
        self.connect_prob = connect_prob

        self.in_collision = collision_func
        if collision_func is None:
            self.in_collision = self.fake_in_collision

        # Setup range limits
        self.limits = lims
        if self.limits is None:
            self.limits = []
            for n in range(num_dimensions):
                self.limits.append([0,100])
            self.limits = np.array(self.limits)

        self.ranges = self.limits[:,1] - self.limits[:,0]
        self.found_path = False

    

    def build_rrt(self, init, goal):
        '''
        Build the rrt from init to goal
        Returns path to goal or None
        '''
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build tree and search
        self.T = RRTSearchTree(init)

        # Sample and extend
        raise NotImplementedError('Expand RRT tree and return plan')

        return None

    def build_rrt_connect(self, init, goal):
        '''
        Build the rrt connect from init to goal
        Returns path to goal or None
        '''
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build tree and search
        self.T = RRTSearchTree(init)

        # Sample and extend
        raise NotImplementedError('Expand RRT tree and return plan')

        return None

    def build_bidirectional_rrt_connect(self, init, goal):
        '''
        Build two rrt connect trees from init and goal
        Growing towards each oter
        Returns path to goal or None
        '''
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build trees and search
        self.T_init = RRTSearchTree(init)
        self.T_goal = RRTSearchTree(goal)

        # Sample and extend
        raise NotImplementedError('Expand RRT trees and return plan')

        return None
    
    def build_rrt_star(self, init, goal):
        """
        RRT* implementation: samples, steers, chooses best parent among neighbors,
        then rewires nearby nodes for lower-cost connections.
        Returns: the (near-)optimal path as a list of states, or None.
        """
        self.goal = np.array(goal)
        self.init = np.array(init)
        # create tree and set root cost
        self.T = RRTSearchTree(self.init)
        self.T.root.cost = 0.0

        for i in range(self.K):
            # 1) sample
            q_rand = self.sample()

            # 2) find nearest & steer
            nearest, dist = self.T.find_nearest(q_rand)
            if dist == 0:
                continue
            direction = (q_rand - nearest.state) / dist
            new_state = nearest.state + min(self.epsilon, dist) * direction

            # 3) collision check
            if not self.collision_free(nearest.state, new_state):
                continue

            # 4) find neighbors within radius
            r = getattr(self, 'neighbor_radius', self.epsilon * 5.0)
            neighbors = [n for n in self.T.nodes
                         if np.linalg.norm(n.state - new_state) <= r]

            # 5) choose best parent among neighbors
            best_cost, best_parent = float('inf'), None
            for n in neighbors:
                if self.collision_free(n.state, new_state):
                    c = n.cost + np.linalg.norm(n.state - new_state)
                    if c < best_cost:
                        best_cost, best_parent = c, n
            # fallback to nearest if no neighbor is valid
            if best_parent is None:
                if not self.collision_free(nearest.state, new_state):
                    continue
                best_parent, best_cost = nearest, nearest.cost + np.linalg.norm(nearest.state - new_state)

            # 6) add the new node
            new_node = TreeNode(new_state)
            self.T.add_node(new_node, best_parent, cost=best_cost)

            # 7) rewire: see if going through new_node improves any neighbors
            for n in neighbors:
                if n is new_node:
                    continue
                new_cost = new_node.cost + np.linalg.norm(n.state - new_node.state)
                if new_cost < n.cost and self.collision_free(new_node.state, n.state):
                    # detach from old parent
                    old_parent = n.parent
                    try:
                        old_parent.children.remove(n)
                        self.T.edges.remove((old_parent.state, n.state))
                    except:
                        pass
                    # attach to new_node
                    n.parent = new_node
                    new_node.children.append(n)
                    n.cost = new_cost
                    self.T.edges.append((new_node.state, n.state))

            # 8) check if we can connect to goal
            if np.linalg.norm(new_state - self.goal) < self.epsilon:
                if self.collision_free(new_state, self.goal):
                    goal_node = TreeNode(self.goal)
                    goal_cost = new_node.cost + np.linalg.norm(new_node.state - self.goal)
                    self.T.add_node(goal_node, new_node, cost=goal_cost)
                    return self.T.get_back_path(goal_node)

        # no path found
        return None

    def sample(self):
        '''
        Sample a new configuration uniformly in the given limits,
        with probability connect_prob returning the goal.
        '''
        if random.random() < self.connect_prob:
            return self.goal.copy()
        else:
            return np.array([random.uniform(self.limits[i,0], self.limits[i,1])
                             for i in range(self.n)])
        
    def collision_free(self, q1, q2, resolution=0.1):
        '''
        Check if the path from q1 to q2 is collision free.
        We interpolate between q1 and q2 with steps of size (resolution * epsilon).
        '''
        dist = np.linalg.norm(q2 - q1)
        if dist == 0:
            return True
        num_steps = int(dist / (resolution * self.epsilon))
        if num_steps < 2:
            num_steps = 2
        for i in np.linspace(0, 1, num_steps):
            q = q1 + i * (q2 - q1)
            if self.in_collision(q):
                return False
        return True

    def extend(self, T, q):
        '''
        Perform rrt extend operation.
        q - new configuration to extend towards
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED
        '''
        raise NotImplementedError('Extend the tree towards q')

    def fake_in_collision(self, q):
        '''
        We never collide with this function!
        '''
        return False

def test_rrt_env(num_samples=500, step_length=2, env='./env0.txt', connect=False):
    '''
    create an instance of PolygonEnvironment from a description file and plan a path from start to goal on it using an RRT

    num_samples - number of samples to generate in RRT
    step_length - step size for growing in rrt (epsilon)
    env - path to the environment file to read
    connect - If True run rrt_connect

    returns plan, planner - plan is the set of configurations from start to goal, planner is the rrt used for building the plan
    '''
    pe = PolygonEnvironment()
    pe.read_env(env)

    dims = len(pe.start)
    start_time = time.time()
    
    rrt = RRT(num_samples,
              dims,
              step_length,
              lims = pe.lims,
              connect_prob = 0.05,
              collision_func=pe.test_collisions)
    if connect:
        plan = rrt.build_rrt_connect(pe.start, pe.goal)
    else:
        plan = rrt.build_rrt(pe.start, pe.goal)
    run_time = time.time() - start_time
    print('plan:', plan)
    print( 'run_time =', run_time)

    pe.draw_env(show=False)
    pe.draw_plan(plan, rrt, True, True, True)

    return plan, rrt
