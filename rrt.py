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
    
    def find_neighbors(self, s_query, radius):
        """
        Find nodes in tree within radius of s_query
        returns - list of nodes within radius
        """
        neighbors = []
        for n_i in self.nodes:
            d = np.linalg.norm(s_query - n_i.state)
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

    # Added
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

        for i in range(self.K):
            x_rand = self.sample()
            status, new_node = self.extend_rewire(self.T, x_rand)
            # if status == _REACHED:
            #     self.record.append((i, self.T.get_back_path(new_node), new))

        if status == _REACHED:
            return self.T.get_back_path(new_node)

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

        if not self.in_collision(x_new, name=self.name):
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

    def extend_rewire(self, T, q):
        """
        Perform rrt* extend operation.
        q - new configuration to extend towards
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED
        """
        # 1. Find the nearest node in T and the distance to q
        x_nearest, distance = T.find_nearest(q)

        # Cannot extend if q is too close to x_nearest
        if distance < self.epsilon:
            return _TRAPPED, None

        # 2. Steer: create new point x_new at distance epsilon from x_nearest toward q
        x_new = x_nearest.state + (q - x_nearest.state) * (self.epsilon / distance)
        # cost to reach x_new via x_nearest
        c_new = x_nearest.cost + np.linalg.norm(x_new - x_nearest.state)

        # 3. Collision check for the new edge
        if not self.in_collision(x_new, name=self.name):
            # 4. Find all existing nodes within rewiring radius
            neighbor_nodes = T.find_neighbors(x_new, self.radius)

            best_parent = x_nearest
            best_cost = c_new
            safe_neighbor_nodes = []

            # 5. Among neighbors, pick a collision-free parent with minimal cost
            for near_node in neighbor_nodes:
                if not self.in_collision(near_node.state, name=self.name):
                    cost = near_node.cost + np.linalg.norm(near_node.state - x_new)
                    if cost < best_cost:
                        best_parent = near_node
                        best_cost = cost
                    safe_neighbor_nodes.append(near_node)

            # 6. Create the new node with the best parent and cost, then add to T
            new_node = TreeNode(x_new, best_parent, best_cost)
            T.add_node(new_node, best_parent)

            # 7. Rewire: try to connect safe neighbors through new_node if it lowers their cost
            for near_node in safe_neighbor_nodes:
                new_cost = new_node.cost + np.linalg.norm(near_node.state - new_node.state)
                if new_cost < near_node.cost:
                    # remove old edge and add the new lower-cost edge
                    T.remove_edge(near_node.parent, near_node)
                    near_node.cost = new_cost
                    T.add_edge(new_node, near_node)

            # 8. Check if the newly added node reached the goal
            cost_to_goal = np.linalg.norm(x_new - self.goal)
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
        
    def fake_in_collision(self, q):
        """
        We never collide with this function!
        """
        return False
