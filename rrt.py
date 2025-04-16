#!/usr/bin/env python
'''
Package providing helper classes and functions for performing graph search operations for planning.
'''
import math
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
            # print(n_i.state)
            d = np.linalg.norm(s_query - n_i.state)
            if d < min_d:
                nn = n_i
                min_d = d
        return (nn, min_d)

    def add_node(self, node, parent):
        '''
        Add a node to the tree
        node - new node to add
        parent - nodes parent, already in the tree
        '''
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

    def build_rrt(self, init, goal, enviroment):
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
        for k in range(self.K):
            xRand = self.sample()
            # print("xRand: ", xRand)
            status, node = self.extend(self.T, xRand)
            # if node.state == self.goal.tolist():
            #     print("Goal!!!")
            #     break
            # print("Testing")
            # print("Test: ", (enviroment.robot.fk(node.state[:int(self.n/2)])))
            # print("Final: ", (enviroment.robot.fk(node.state[:int(self.n/2)]))[-1])
            # print("something: ", np.linalg.norm((enviroment.robot.fk(node.state[:int(self.n/2)]))[-1] + (enviroment.robot.fk(node.state[int(self.n/2):]))[-1]))
            # np.linalg.norm(s_query - n_i.state)
            # break
            if np.linalg.norm((enviroment.robot.fk(node.state[:int(self.n/2)]))[-1] - (enviroment.robot.fk(node.state[int(self.n/2):]))[-1]) < 20: # Goal is an array of positions [(x, y), (x, y)]
                print("Goal!!!")
                break
        nearGoal, dist = self.T.find_nearest(self.goal)
        return self.T.get_back_path(nearGoal)
        
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
        status = _REACHED
        xRand = self.sample()
        for k in range(self.K):
            if status != _ADVANCED:
                xRand = self.sample()
                # print(status)
                # print("Not Advancing")
            status, node = self.extend(self.T, xRand)
            if node.state == self.goal.tolist():
                print("Goal!!!")
                break
        nearGoal, dist = self.T.find_nearest(self.goal)
        return self.T.get_back_path(nearGoal)

        raise NotImplementedError('Expand RRT tree and return plan')

        return None

    def build_bidirectional_rrt_connect(self, init, goal, goalPosition):
        '''
        Build two rrt connect trees from init and goal
        Growing towards each oter
        Returns path to goal or None
        '''
        self.goal = np.array(goal)
        self.init = np.array(init)

        self.goalPosition = goalPosition

        self.found_path = False

        # Build trees and search
        self.T_init = RRTSearchTree(self.init)
        self.T_goal = RRTSearchTree(self.goal)

        # Sample and extend
        treeA = self.T_init
        treeB = self.T_goal
        initIsTreeA = True
        status = _REACHED
        xRand = self.BidirectionalSample(treeA, treeB)
        for k in range(self.K):
            if status != _ADVANCED:
                xRand = self.BidirectionalSample(treeA, treeB)
            else:
                initIsTreeA = not initIsTreeA
                # print(status)
                # print("Not Advancing")
            status1, node1 = self.extend(treeA, xRand)
            if status1 != _TRAPPED:
                status2, node2 = self.extend(treeB, node1.state)
                if status2 == _REACHED:
                    endPath, dist = self.T_init.find_nearest(np.array(node2.state))
                    initPath = self.T_init.get_back_path(endPath)
                    endPath, dist = self.T_goal.find_nearest(np.array(node2.state))
                    goalPath = self.T_goal.get_back_path(endPath)
                    goalPath.reverse()
                    states, edges = self.T_goal.get_states_and_edges()

                    # Add Path from goal Tree to init Tree
                    parentNode, dist = self.T_init.find_nearest(np.array(node2.state))
                    for node in goalPath:
                        newNode = TreeNode(node)
                        self.T_init.add_node(newNode, parentNode)
                        parentNode = newNode
                    self.T = self.T_init
                    nearGoal, dist = self.T_init.find_nearest(goal)
                    returnPath = self.T_init.get_back_path(nearGoal)

                    # Add All goal states to init tree for representation
                    for i in range(len(edges)):
                        node = edges[i][1]
                        parentNode = TreeNode(edges[i][0])
                        newNode = TreeNode(node)
                        self.T_init.add_node(newNode, parentNode)
                    
                    return returnPath

            if initIsTreeA:
                treeA = self.T_goal
                treeB = self.T_init
            else:
                treeA = self.T_init
                treeB = self.T_goal
            initIsTreeA = not initIsTreeA

        
        self.T = self.T_init
        return None
        nearGoal, dist = self.T.find_nearest(self.goal)
        return self.T.get_back_path(nearGoal)

        raise NotImplementedError('Expand RRT trees and return plan')

        return None

    def sample(self):
        '''
        Sample a new configuration
        Returns a configuration of size self.n bounded in self.limits
        '''
        # Return goal with connect_prob probability
        # self.n is an integer for size
        # self.limits looks like [[x_0 min, x_0 max], [x_1 min, x_1 max]]
        chance = random.randint(1,100)
        if chance <= self.connect_prob*100:
            
            goal = []
            for d in range(self.n):
                goal.append(self.goal[d])
            # print("Goal: ", goal)
            return goal

        dimentionsSample = []
        # print(int(self.n/2))
        # print("size: ", self.n)
        for n in range(int(self.n/2)):
            # print("lower bounds: ", self.limits[n][0])
            # print("Upper bounds: ", self.limits[n][1])
            rand = random.uniform(self.limits[n][0], self.limits[n][1])
            dimentionsSample.append(rand)
        for n in range(int(self.n/2)):
            # print("lower bounds: ", self.limits[n][0])
            # print("Upper bounds: ", self.limits[n][1])
            rand = random.uniform(self.limits[n][0], self.limits[n][1])
            dimentionsSample.append(rand)
        return dimentionsSample
        raise NotImplementedError('Sample a new configuration, or return goal')
    

    def BidirectionalSample(self, advancingTree, goalTree):
        '''
        Sample a new configuration
        Returns a configuration of size self.n bounded in self.limits
        '''
        # Return goal with connect_prob probability
        # self.n is an integer for size
        # self.limits looks like [[x_0 min, x_0 max], [x_1 min, x_1 max]]
        chance = random.randint(1,100)
        if chance <= self.connect_prob*100:
            
            # Go towards closest Node on other tree
            closestPoint, dist = goalTree.find_nearest(advancingTree.root.state)
            goal = []
            for d in range(self.n):
                goal.append(closestPoint.state[d])
                # print(closestPoint.state[d])
            return goal

        dimentionsSample = []
        for n in range(self.n):
            # print("lower bounds: ", self.limits[n][0])
            # print("Upper bounds: ", self.limits[n][1])
            rand = random.uniform(self.limits[n][0], self.limits[n][1])
            dimentionsSample.append(rand)
        return dimentionsSample
        raise NotImplementedError('Sample a new configuration, or return goal')

    def extend(self, T, q):
        '''
        Perform rrt extend operation.
        q - new configuration to extend towards
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED
        '''
        # print(q)
        parentNode, dist = T.find_nearest(np.array(q))
        xNear = parentNode.state

        dir = []
        for d in range(self.n):
            temp = q[d] - xNear[d]
            dir.append(temp)

        for d in range(self.n):
            dir[d] = dir[d]/dist

        newNode = []
        for d in range(self.n):
            if dist > self.epsilon:
                newNode.append(xNear[d] + dir[d] * self.epsilon)
            else:
                newNode.append(xNear[d] + dir[d] * dist)
        node = TreeNode(newNode)
        if not self.in_collision(newNode):
            T.add_node(node, parentNode)
            
            if newNode == q:
                # print("REached")
                return (_REACHED, node)
            else:
                return (_ADVANCED, node)
        return (_TRAPPED, node)

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

    pe.draw_env(show=True)
    pe.draw_plan(plan, rrt, True, True, True)

    return plan, rrt
