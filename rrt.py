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

#random.seed()

class TreeNode:
    '''
    Class to hold node state and connectivity for building an RRT
    '''
    def __init__(self, state, parent=None):
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child):
        '''
        Add a child node
        '''
        self.children.append(child)

class RRTSearchTree:
    '''
    Searh tree used for building an RRT
    '''
    def __init__(self, init=None):
        '''
        init - initial tree configuration
        '''
        if init is not None:
            self.root = TreeNode(init)
            self.nodes = [self.root]
        else:
            self.nodes = []
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

    def add_node(self, node: TreeNode):
        '''
        Add a node to the tree
        node - new node to add
        parent - nodes parent, already in the tree
        '''
        self.nodes.append(node)
        self.edges.append((node.parent.state, node.state))
        node.parent.add_child(node)

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
        samples = 0

        while samples < self.K:
            # sample and extend
            status, new_node = self.extend(self.T, self.sample(self.goal))
            samples += 1

            # get a new sample if collision
            if status == _TRAPPED:
                continue

            # add node to tree
            self.T.add_node(new_node)

            # return path if the new node is the goal
            if status == _REACHED and (new_node.state==self.goal).all():
                return self.T.get_back_path(new_node)
            
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
        samples = 0

        while samples < self.K:
            # sample and extend
            sample = self.sample(self.goal)
            status, new_node = self.extend_connect(self.T, sample)
            samples += 1

            if status == _TRAPPED:
                continue

            if (new_node.state==self.goal).all():
                return self.T.get_back_path(new_node)
        
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
        self.T = RRTSearchTree()
        self.T.nodes.append(TreeNode(init))
        self.T.nodes.append(TreeNode(goal))

        samples = 0

        expanding_tree = self.T_init
        target_tree = self.T_goal

        while samples < self.K:
            # sample and extend
            sample = self.sample(target_tree.nodes[0].state)
            _, target_node = self.extend_connect_bidirectional(expanding_tree, sample)
            samples += 1

            # try to connect trees
            status, new_node = self.extend_connect_bidirectional(target_tree, target_node.state)

            if status == _REACHED: # trees connected
                if new_node in self.T_init.nodes:
                    init_node = new_node
                    goal_node = target_node
                else:
                    init_node = target_node
                    goal_node = new_node
                return self.T_init.get_back_path(init_node) + self.T_goal.get_back_path(goal_node)[::-1]

            # swap trees and go again
            temp = expanding_tree
            expanding_tree = target_tree
            target_tree = temp

        return None

    def sample(self, target):
        '''
        Sample a new configuration
        Returns a configuration of size self.n bounded in self.limits
        '''
        # Return goal with connect_prob probability
        if random.random() <= self.connect_prob:
            return target
        
        while True:
            sample = np.array([random.uniform(self.limits[i][0], self.limits[i][1]) for i in range(self.n)])
            if not self.in_collision(sample):
                break
        return sample

    def extend(self, T: RRTSearchTree, q, nearest_node=None):
        '''
        Perform rrt extend operation.
        q - new configuration to extend towards
        nearest_node - the nearest node if it is already known
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED
        '''

        # find the nearest node to q
        if nearest_node == None:
            nearest_node, distance = T.find_nearest(q)
        else:
            distance = np.linalg.norm(q - nearest_node.state)

        # extend in the direction of q
        if distance < self.epsilon:
            new_state = q
            status = _REACHED
        else:
            direction = q - nearest_node.state
            new_state = nearest_node.state + direction * self.epsilon / distance
            if self.in_collision(new_state):
                status = _TRAPPED
            else:
                status = _ADVANCED

        return (status, TreeNode(new_state, nearest_node))
    
    def extend_connect(self, T: RRTSearchTree, q):
        '''
        Perform rrt extend operation for rrt connect.
        q - new configuration to extend towards
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED
        '''

        nearest_node, distance = T.find_nearest(q)
        direction = q - nearest_node.state
        step = direction*self.epsilon/distance

        prev_node = nearest_node
    
        while distance >= self.epsilon:

            new_state = prev_node.state + step

            if self.in_collision(new_state):
                return (_TRAPPED, prev_node)
            
            new_node = TreeNode(new_state, prev_node)
            T.add_node(new_node)
            prev_node = new_node
            distance -= self.epsilon

        new_node = TreeNode(q, prev_node)
        T.add_node(new_node)
        return (_REACHED, new_node)
    
    def extend_connect_bidirectional(self, T: RRTSearchTree, q):
        '''
        Perform rrt extend operation for rrt connect.
        q - new configuration to extend towards
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED
        '''

        nearest_node, distance = T.find_nearest(q)
        direction = q - nearest_node.state
        step = direction*self.epsilon/distance

        prev_node = nearest_node
    
        while distance >= self.epsilon:

            new_state = prev_node.state + step

            if self.in_collision(new_state):
                return (_TRAPPED, prev_node)
            
            new_node = TreeNode(new_state, prev_node)
            T.add_node(new_node)
            self.T.edges.append(T.edges[-1])
            prev_node = new_node
            distance -= self.epsilon

        new_node = TreeNode(q, prev_node)
        T.add_node(new_node)
        self.T.edges.append(T.edges[-1])
        return (_REACHED, new_node)

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
