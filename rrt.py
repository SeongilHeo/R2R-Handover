#!/usr/bin/env python
"""
Package providing helper classes and functions for performing graph search operations for planning.
"""
import numpy as np

_TRAPPED = "trapped"
_ADVANCED = "advanced"
_REACHED = "reached"


class TreeNode:
    """
    Class to hold node state and connectivity for building an RRT
    """

    def __init__(self, state, parent=None):
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child):
        """
        Add a child node
        """
        self.children.append(child)


class RRTSearchTree:
    """
    Searh tree used for building an RRT
    """

    def __init__(self, init):
        """
        init - initial tree configuration
        """
        self.root = TreeNode(init)
        self.nodes = [self.root]
        self.edges = []

    def find_nearest(self, s_query):
        """
        Find node in tree closets to s_query
        returns - (nearest node, dist to nearest node)
        """
        min_d = 1000000
        nn = self.root
        for n_i in self.nodes:
            d = np.linalg.norm(s_query - n_i.state)
            if d < min_d:
                nn = n_i
                min_d = d
        return (nn, min_d)

    def add_node(self, node, parent):
        """
        Add a node to the tree
        node - new node to add
        parent - nodes parent, already in the tree
        """
        self.nodes.append(node)
        self.edges.append((parent.state, node.state))
        node.parent = parent
        parent.add_child(node)

    def get_states_and_edges(self):
        """
        Return a list of states and edgs in the tree
        """
        states = np.array([n.state for n in self.nodes])
        return (states, self.edges)

    def get_back_path(self, n):
        """
        Get the path from the root to a specific node in the tree
        n - node in tree to get path to
        """
        path = []
        while n.parent is not None:
            path.append(n.state)
            n = n.parent
        path.append(n.state)
        path.reverse()
        return path


class RRT(object):
    """
    Rapidly-Exploring Random Tree Planner
    """

    def __init__(
        self,
        num_samples,
        num_dimensions=2,
        step_length=1,
        lims=None,
        connect_prob=0.05,
        collision_func=None,
    ):
        """
        Initialize an RRT planning instance
        """
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
                self.limits.append([0, 100])
            self.limits = np.array(self.limits)

        self.ranges = self.limits[:, 1] - self.limits[:, 0]
        self.found_path = False

    def plan(self, init, goal):
        """
        Plan a path from init to goal
        """
        plan = self.build_rrt(init, goal)
        return plan

    def build_rrt(self, init, goal):
        """
        Build the rrt from init to goal
        Returns path to goal or None
        """
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build tree and search
        self.T = RRTSearchTree(init)

        # Sample and extend
        for _ in range(self.K):
            x_rand = self.sample()
            status, new_node = self.extend(self.T, x_rand)

            if status == _REACHED:
                return self.T.get_back_path(new_node)

        return None

    def build_rrt_connect(self, init, goal):
        """
        Build the rrt connect from init to goal
        Returns path to goal or None
        """
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build tree and search
        self.T = RRTSearchTree(init)

        # Sample and extend
        for i in range(self.K):
            x_rand = self.sample()
            while True:
                status, new_node = self.extend(self.T, x_rand)

                if status in [_TRAPPED, _REACHED]:
                    break

            if status == _REACHED:
                return self.T.get_back_path(new_node)

        return None

    def build_bidirectional_rrt_connect(self, init, goal):
        """
        Build two rrt connect trees from init and goal
        Growing towards each oter
        Returns path to goal or None
        """
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build trees and search
        self.T_init = RRTSearchTree(init)
        self.T_goal = RRTSearchTree(goal)

        T_a, T_b = self.T_init, self.T_goal

        for iter in range(self.K):
            goal_start = True if (T_a.root.state == self.goal).all() else False

            q_rand = self.sample(goal_start)
            status_a, node_a = self.extend(T_a, q_rand, goal_start)

            if status_a != _TRAPPED:
                while True:
                    status_b, node_b = self.extend(T_b, node_a.state, not goal_start)
                    if status_b == _TRAPPED:
                        break

                    if (
                        status_b == _REACHED
                        or np.linalg.norm(node_b.state - node_a.state) < self.epsilon
                    ):
                        self.T = RRTSearchTree(self.init)
                        self.T.nodes = self.T_init.nodes + self.T_goal.nodes
                        self.T.edges = (
                            self.T_init.edges
                            + self.T_goal.edges
                            + [(node_b.state, node_a.state)]
                        )

                        path_from_init = self.T_init.get_back_path(node_a)
                        path_from_goal = self.T_goal.get_back_path(node_b)
                        path_from_goal.reverse()

                        path = path_from_init + path_from_goal

                        if not (path[0] == self.init).all():
                            path.reverse()

                        return path
            if len(T_a.nodes) > len(T_b.nodes):
                T_a, T_b = T_b, T_a

        return None

    def sample(self, goal_start=False):
        """
        Sample a new configuration
        Returns a configuration of size self.n bounded in self.limits
        """
        # Return goal with connect_prob probability
        if np.random.rand() < self.connect_prob:
            return self.init if goal_start else self.goal
        else:
            return np.random.uniform(self.limits[:, 0], self.limits[:, 1])

    def extend(self, T, q, goal_start=False):
        """
        Perform rrt extend operation.
        q - new configuration to extend towards
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED
        """
        goal = self.init if goal_start else self.goal

        x_near, distance = T.find_nearest(q)

        if distance < self.epsilon:
            return _TRAPPED, None

        x_new = x_near.state + (q - x_near.state) * (self.epsilon / distance)

        if not self.in_collision(x_new):
            new_node = TreeNode(x_new, x_near)
            T.add_node(new_node, x_near)

            if np.linalg.norm(x_new - goal) < self.epsilon:
                if (x_new != goal).all():
                    goal_node = TreeNode(goal, new_node)
                    T.add_node(goal_node, new_node)
                else:
                    goal_node = new_node

                return _REACHED, goal_node

            return _ADVANCED, new_node

        return _TRAPPED, None

    def fake_in_collision(self, q):
        """
        We never collide with this function!
        """
        return False
