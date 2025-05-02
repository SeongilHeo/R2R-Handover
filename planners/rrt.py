#!/usr/bin/env python
"""Rapidly-exploring random tree planners."""

import numpy as np
from time import time

_TRAPPED = "trapped"
_ADVANCED = "advanced"
_REACHED = "reached"

_COLLISION = 0
_NEAR = 1
_NEIGH = 2
_TOTAL = 3

def compute_dist(A, B):
    """
    Compute the Euclidean distance between two states.
    """
    cost = np.linalg.norm(np.asarray(A) - np.asarray(B))
    return cost

def compute_smooth(path):
    """
    Compute a simple smoothness cost from heading changes along a path.
    """
    if path is None or len(path) < 3:
        return 0.0

    smoothness = 0.0
    for i in range(1, len(path) - 1):
        prev_step = np.asarray(path[i]) - np.asarray(path[i - 1])
        next_step = np.asarray(path[i + 1]) - np.asarray(path[i])
        prev_norm = np.linalg.norm(prev_step)
        next_norm = np.linalg.norm(next_step)
        if prev_norm == 0 or next_norm == 0:
            continue
        cosine = np.dot(prev_step, next_step) / (prev_norm * next_norm)
        smoothness += np.arccos(np.clip(cosine, -1.0, 1.0))
    return smoothness


def compute_cost(path):
    """
    Compute the total length of a path.
    """
    if path is None or len(path) < 2:
        return 0.0

    return sum(compute_dist(path[i], path[i + 1]) for i in range(len(path) - 1))

class TreeNode:
    """
    Class to hold node state and connectivity for building an RRT
    """

    def __init__(self, state, parent=None, cost=0.0):
        self.state = state
        self.children = []
        self.parent = parent
        self.cost = cost  # added

    def add_child(self, child):
        """
        Add a child node
        """
        self.children.append(child)

    def remove_child(self, child): 
        """
        Remove a child node
        """
        self.children.remove(child)

class RRTSearchTree:
    """
    Search tree used for building an RRT
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
        Find node in tree closest to s_query
        returns - (nearest node, dist to nearest node)
        """
        min_d = 1000000
        nn = self.root
        for n_i in self.nodes:
            d = compute_dist(s_query, n_i.state)
            if d < min_d:
                nn = n_i
                min_d = d
        return (nn, min_d)
    
    def find_neighbors(self, s_query, radius):
        """
        Find nodes in tree within radius of s_query
        returns - list of nodes within radius
        """
        neighbors = []
        for n_i in self.nodes:
            d = compute_dist(s_query, n_i.state)
            if d < radius:
                neighbors.append(n_i)
        return neighbors

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

    def add_edge(self, parent, child):
        """
        Add an edge to the tree
        parent - parent node
        child - child node
        """
        parent.add_child(child)
        child.parent = parent
        self.edges.append((parent.state, child.state))

    def remove_edge(self, parent, child):
        """
        Remove an edge from the tree
        parent - parent node
        child - child node
        """
        parent.remove_child(child)
        child.parent = None
        self.edges.remove((parent.state, child.state))

    def get_states_and_edges(self):
        """
        Return a list of states and edges in the tree
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
        radius=0.1,
        lims=None,
        connect_prob=0.05,
        collision_func=None,
        name=None
    ):
        """
        Initialize an RRT planning instance
        """
        self.name = name
        self.K = num_samples
        self.n = num_dimensions
        self.epsilon = step_length
        self.connect_prob = connect_prob
        self.radius = radius # for rrt*
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

        self.time_table = {
            _COLLISION: 0,
            _NEAR: 0,
            _NEIGH: 0,
            _TOTAL: 0
        }

        self.records = []

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
        start_time = time()
        # Sample and extend
        for _ in range(self.K):
            x_rand = self.sample()
            status, new_node = self.extend(self.T, x_rand)

            if status == _REACHED:
                self.time_table[_TOTAL] = (time() - start_time)
                return self.T.get_back_path(new_node)
        self.time_table[_TOTAL] = (time() - start_time)
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
        
        start_time = time()
        # Sample and extend
        for i in range(self.K):
            x_rand = self.sample()
            while True:
                status, new_node = self.extend(self.T, x_rand)

                if status in [_TRAPPED, _REACHED]:
                    break

            if status == _REACHED:
                self.time_table[_TOTAL] = (time() - start_time)
                return self.T.get_back_path(new_node)
        
        self.time_table[_TOTAL] = (time() - start_time)
        return None

    def build_bidirectional_rrt_connect(self, init, goal):
        """
        Build two rrt connect trees from init and goal
        Growing towards each other
        Returns path to goal or None
        """
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build trees and search
        self.T_init = RRTSearchTree(init)
        self.T_goal = RRTSearchTree(goal)

        T_a, T_b = self.T_init, self.T_goal
        start_time = time()
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
                        or compute_dist(node_b.state, node_a.state) < self.epsilon
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

                        self.time_table[_TOTAL] = (time() - start_time)
                        return path
            if len(T_a.nodes) > len(T_b.nodes):
                T_a, T_b = T_b, T_a

        self.time_table[_TOTAL] = (time() - start_time)
        return None

    def build_rrt_star(self, init, goal):
        """
        RRT* implementation: samples, steers, chooses best parent among neighbors,
        then rewires nearby nodes for lower-cost connections.
        Returns: the (near-)optimal path as a list of states, or None.
        """
        self.goal = np.array(goal)
        self.init = np.array(init)
        
        # Build tree and search
        self.T = RRTSearchTree(init)
        start_time = time()
        goal_node = None
        for i in range(self.K):
            x_rand = self.sample()
            status, new_node = self.extend_rewire(self.T, x_rand)
            if status == _REACHED:
                goal_node = new_node
                self.record_path(i, self.T.get_back_path(new_node))

        if goal_node is not None:
            self.time_table[_TOTAL] = (time() - start_time)
            return self.T.get_back_path(goal_node)
        
        self.time_table[_TOTAL] = (time() - start_time)
        return None

    def build_rrt_star_trace(self, init, goal):
        """
        Run RRT* and record cumulative timing and best path length each iteration.
        """
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        self.T = RRTSearchTree(init)
        self.records = []
        best_node = None
        start_time = time()

        for iter_idx in range(1, self.K + 1):
            x_rand = self.sample()
            status, new_node = self.extend_rewire(self.T, x_rand)

            if status == _REACHED:
                best_node = new_node

            best_path = self.T.get_back_path(best_node) if best_node is not None else None
            total_time = time() - start_time
            self.time_table[_TOTAL] = total_time
            self.records.append(
                {
                    "iter": iter_idx,
                    "collision": self.time_table[_COLLISION],
                    "near": self.time_table[_NEAR],
                    "neighbors": self.time_table[_NEIGH],
                    "total": total_time,
                    "path": compute_cost(best_path),
                }
            )

        self.time_table[_TOTAL] = time() - start_time
        return (self.T.get_back_path(best_node) if best_node is not None else None), self.records

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
        # Collision time (start)
        ne_start_time = time()
        x_near, distance = T.find_nearest(q)
        self.time_table[_NEAR] += (time() - ne_start_time)

        if distance < self.epsilon:
            return _TRAPPED, None
        x_new = x_near.state + (q - x_near.state) * (self.epsilon / distance)
        # Collision time (start)
        c_start_time = time()
        if not self.in_collision(x_new, name=self.name):
            # Collision time (end)
            self.time_table[_COLLISION] += (time() - c_start_time)

            new_node = TreeNode(x_new, x_near)
            T.add_node(new_node, x_near)

            if compute_dist(x_new, goal) < self.epsilon:
                if not np.allclose(x_new, goal):
                    goal_node = TreeNode(goal, new_node)
                    T.add_node(goal_node, new_node)
                else:
                    goal_node = new_node

                return _REACHED, goal_node

            return _ADVANCED, new_node

        return _TRAPPED, None

    def extend_rewire(self, T, q):
        """
        Perform rrt* extend operation.
        q - new configuration to extend towards
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED
        """
        # 1. Find the nearest node in T and the distance to q
        ne_start_time = time()
        x_nearest, distance = T.find_nearest(q)
        self.time_table[_NEAR] += (time() - ne_start_time)

        # Cannot extend if q is too close to x_nearest
        if distance < self.epsilon:
            return _TRAPPED, None

        # 2. Steer: create new point x_new at distance epsilon from x_nearest toward q
        x_new = x_nearest.state + (q - x_nearest.state) * (self.epsilon / distance)
        # cost to reach x_new via x_nearest
        c_new = x_nearest.cost + compute_dist(x_new, x_nearest.state)

        # 3. Collision check for the new edge
        # Collision time (start)
        t_start_time = time()
        if not self.in_collision(x_new, name=self.name):
            self.time_table[_COLLISION] += (time() - t_start_time)
            # 4. Find all existing nodes within rewiring radius
            n_start_time = time()
            neighbor_nodes = T.find_neighbors(x_new, self.radius)
            self.time_table[_NEIGH] += (time() - n_start_time)

            best_parent = x_nearest
            best_cost = c_new
            safe_neighbor_nodes = []

            # 5. Among neighbors, pick a collision-free parent with minimal cost
            for near_node in neighbor_nodes:
                c_start_time = time()
                if not self.in_collision(near_node.state, name=self.name):
                    self.time_table[_COLLISION] += (time() - c_start_time)

                    cost = near_node.cost + compute_dist(near_node.state, x_new)
                    if cost < best_cost:
                        best_parent = near_node
                        best_cost = cost
                    safe_neighbor_nodes.append(near_node)

            # 6. Create the new node with the best parent and cost, then add to T
            new_node = TreeNode(x_new, best_parent, best_cost)
            T.add_node(new_node, best_parent)

            # 7. Rewire: try to connect safe neighbors through new_node if it lowers their cost
            for near_node in safe_neighbor_nodes:
                new_cost = new_node.cost + compute_dist(near_node.state, new_node.state)
                if new_cost < near_node.cost:
                    # remove old edge and add the new lower-cost edge
                    T.remove_edge(near_node.parent, near_node)
                    near_node.cost = new_cost
                    T.add_edge(new_node, near_node)

            # 8. Check if the newly added node reached the goal
            cost_to_goal = compute_dist(x_new, self.goal)
            if cost_to_goal < self.epsilon:
                # if x_new is not exactly the goal, add a goal node
                if not np.allclose(x_new, self.goal):
                    goal_node = TreeNode(self.goal, new_node, new_node.cost + cost_to_goal)
                    T.add_node(goal_node, new_node)
                else:
                    goal_node = new_node
                return _REACHED, goal_node

            # Successfully added node without reaching goal
            return _ADVANCED, new_node

        # Collision detected; extension is trapped
        return _TRAPPED, None
        
    def plan(self, start, goal):
        """
        Provide a local-planner interface for PRM.
        """
        return self.build_rrt_connect(start, goal)

    def fake_in_collision(self, q, name=None):
        """
        We never collide with this function!
        """
        return False

    def record_path(self, iter, path):
        """
        Record the path and time taken
        """
        self.records.append([iter, path, dict(self.time_table)])
