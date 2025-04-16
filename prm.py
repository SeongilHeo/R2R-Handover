#!/usr/bin/env python
"""
Package providing implementation of a probabilistic roadmap algorithm
"""

import numpy as np
import heapq


def fake_in_collision(q):
    """
    We never collide with this function!
    """
    return False


def euclidean_heuristic(self, s, goal):
    """
    Euclidean heuristic function

    s - configuration vector
    goal - goal vector

    returns - floating point estimate of the cost to the goal from state s
    """
    return np.linalg.norm(s - goal)


def get_distance(path):
    distance = 0
    for i in range(len(path) - 1):
        distance += np.linalg.norm(path[i] - path[i + 1])
    return distance


class PriorityQ:
    """
    Priority queue implementation with quick access for membership testing
    Setup currently to only with the SearchNode class
    """

    def __init__(self):
        """
        Initialize an empty priority queue
        """
        self.l = []  # list storing the priority q
        self.s = set()  # set for fast membership testing

    def __contains__(self, x):
        """
        Test if x is in the queue
        """
        return x in self.s

    def push(self, x, cost):
        """
        Adds an element to the priority queue.
        If the state already exists, we update the cost
        """
        if tuple(x.state.tolist()) in self.s:
            return self.replace(x, cost)
        heapq.heappush(self.l, (cost, x))
        self.s.add(tuple(x.state.tolist()))

    def pop(self):
        """
        Get the value and remove the lowest cost element from the queue
        """
        x = heapq.heappop(self.l)
        self.s.remove(tuple(x[1].state.tolist()))
        return x[1]

    def peak(self):
        """
        Get the value of the lowest cost element in the priority queue
        """
        x = self.l[0]
        return x[1]

    def __len__(self):
        """
        Return the number of elements in the queue
        """
        return len(self.l)

    def replace(self, x, new_cost):
        """
        Removes element x from the q and replaces it with x with the new_cost
        """
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
    """
    Function to determine the path that lead to the specified search node

    node - the SearchNode that is the end of the path

    returns - a tuple containing (path, action_path) which are lists respectively of the states
    visited from init to goal (inclusive) and the actions taken to make those transitions.
    """
    path = []
    while node.parent is not None:
        path.append(node.state)
        node = node.parent
    path.append(node.state)
    path.reverse()
    return path


def backpath_global(node, tree_nodes):
    """
    Function to determine the global path that lead to the specified search node

    node - the SearchNode that is the end of the path

    returns - a tuple containing (path, action_path) which are lists respectively of the states
    visited from init to goal (inclusive) and the actions taken to make those transitions.
    """
    path = []
    while node.parent is not None:
        found = False
        for t_node in tree_nodes:
            if node == t_node:
                found = True
        if found:
            path.append(node.state)
        node = node.parent
    path.append(node.state)
    path.reverse()
    return path


class StraightLinePlanner:
    def __init__(self, step_size, collision_func=None):
        self.in_collision = collision_func
        self.epsilon = step_size
        if collision_func is None:
            self.in_collision = fake_in_collision

    def plan(self, start, goal):
        """
        Check if edge is collision free, taking epsilon steps towards the goal
        Returns: None / False if edge in collsion
                 Plan / True if edge if free
        """
        start = np.array(start)
        goal = np.array(goal)

        direction = goal - start
        distance = np.linalg.norm(direction)

        if distance == 0:
            return True

        direction = direction / distance
        for i in range(int(distance / self.epsilon) + 1):
            if self.in_collision(start + i * self.epsilon * direction):
                return False

        return True


class RoadMapNode:
    """
    Nodes to be used in a built RoadMap class
    """

    def __init__(self, state, cost=0, parent=None):
        self.state = np.array(state)
        self.neighbors = []
        self.cost = 0
        self.parent = parent

    def add_neighbor(self, n_new):
        """
        n_new - new neighbor
        """
        self.neighbors.append(n_new)

    def is_neighbor(self, n_test):
        """
        Test if n_test is already our neighbor
        """
        for n in self.neighbors:
            if np.linalg.norm(n.state - n_test.state) == 0.0:
                return True
        return False

    def __eq__(self, other):
        return np.linalg.norm(self.state - other.state) == 0.0


class RoadMap:
    """
    Class to store a built roadmap for searching in our multi-query PRM
    """

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.cuts = {}

    def add_node(self, node, neighbors):
        """
        Add a node to the roadmap. Connect it to its neighbors
        """
        # Avoid adding duplicates
        self.nodes.append(node)
        for n in neighbors:
            node.add_neighbor(n)
            if not n.is_neighbor(node):
                n.add_neighbor(node)
                self.edges.append((n.state, node.state))

    def add_cut(self, local_plan, step):
        distance = get_distance(local_plan)

        key = (tuple(local_plan[0].tolist()), tuple(local_plan[-1].tolist()))
        value = local_plan[1:-1]

        if key not in self.cuts or distance < self.cuts[key][1]:
            self.cuts[key] = (value, distance)

        return True

    def get_cut(self, start_state, goal_state):
        key = tuple(start_state.tolist()), tuple(goal_state.tolist())
        rkey = tuple(goal_state.tolist()), tuple(start_state.tolist())
        if self.cuts.get(key):
            return self.cuts[key]
        elif self.cuts.get(rkey):
            path, cost = self.cuts[rkey]
            return reversed(path), cost
        else:
            return None

    def get_states_and_edges(self):
        states = np.array([n.state for n in self.nodes])
        return (states, self.edges)


class PRM:
    def __init__(
        self,
        num_samples,
        local_planner,
        num_dimensions,
        lims=None,
        collision_func=None,
        radius=2.0,
        epsilon=0.1,
    ):
        self.local_planner = local_planner
        self.r = radius
        self.N = num_samples
        self.n = num_dimensions
        self.epsilon = epsilon  # step_lenght
        self.graph_search = self.uniform_cost_search

        self.in_collision = collision_func
        if collision_func is None:
            self.in_collision = fake_in_collision

        # Setup range limits
        self.limits = lims
        if self.limits is None:
            self.limits = []
            for n in range(num_dimensions):
                self.limits.append([0, 100])
            self.limits = np.array(self.limits)

        self.ranges = self.limits[:, 1] - self.limits[:, 0]

        # Build the roadmap instance
        self.T = RoadMap()

    def build_prm(self, reset=False):
        """
        reset - empty the current roadmap if requested
        """
        if reset:
            self.T = RoadMap()

        for _ in range(self.N):
            q = self.sample()
            if not self.in_collision(q):
                new_node = RoadMapNode(q)
                neighbors = self.find_valid_neighbors(new_node, self.T.nodes, self.r)
                self.T.add_node(new_node, neighbors)

    def find_valid_neighbors(self, n_query, samples, r):
        """
        Find the nodes that are close to n_query and can be attached by the local planner
        returns - list of neighbors reached by the local planner
        """
        valid_neighbors = []

        for node in samples:
            if np.linalg.norm(n_query.state - node.state) < r:
                local_plan = self.local_planner.plan(n_query.state, node.state)
                if local_plan:
                    if isinstance(local_plan, list) and len(local_plan) > 2:
                        if self.T.add_cut(local_plan, r):
                            valid_neighbors.append(node)
                    else:
                        valid_neighbors.append(node)

        return valid_neighbors

    def query(self, start, goal):
        """
        Generate a path from start to goal using the built roadmap
        returns - Path of configurations if in roadmap, None otherwise
        """
        start_node = RoadMapNode(start)
        goal_node = RoadMapNode(goal)

        start_neighbors = self.find_valid_neighbors(start_node, self.T.nodes, self.r)
        goal_neighbors = self.find_valid_neighbors(goal_node, self.T.nodes, self.r)
        self.T.add_node(start_node, start_neighbors)
        self.T.add_node(goal_node, goal_neighbors)

        def is_goal(x):
            """
            Test if a sample is at the goal
            """
            return np.linalg.norm(x - goal) < self.epsilon

        # Run search on the roadmap to find a plan
        start_node.parent = None
        local_plan, global_plan, visited = self.graph_search(start_node, is_goal)
        if local_plan and not (local_plan[-1] == goal).all():
            local_plan.append(goal)
            global_plan.append(goal)
        return local_plan, global_plan, visited

    def uniform_cost_search(self, init_node, is_goal):
        """
        Perform graph search on the roadmap
        """
        cost = 0
        frontier = PriorityQ()
        frontier.push(init_node, cost)
        visited = dict()
        # You need to modify your graph search from HW1 to expand neighbors instead of actions
        while frontier:
            current = frontier.pop()
            if is_goal(current.state):
                return (
                    backpath(current),
                    backpath_global(current, self.T.nodes),
                    visited,
                )

            key = tuple(current.state.tolist())
            if key not in visited or current.cost < visited[key]:
                visited[key] = current.cost

            for neighbor in current.neighbors:
                cut = self.T.get_cut(current.state, neighbor.state)
                if cut:
                    new_cost = current.cost + cut[1]
                    new_key = tuple(neighbor.state.tolist())

                    if new_key not in visited or new_cost < visited[new_key]:
                        prev_node = current
                        for point in cut[0]:
                            point_node = RoadMapNode(point)
                            point_node.parent = prev_node
                            prev_node = point_node

                        neighbor.parent = prev_node
                        neighbor.cost = new_cost

                        visited[new_key] = new_cost
                        frontier.push(neighbor, new_cost)

                else:
                    new_cost = current.cost + np.linalg.norm(
                        neighbor.state - current.state
                    )
                    new_key = tuple(neighbor.state.tolist())

                    if new_key not in visited or new_cost < visited[new_key]:
                        neighbor.parent = current
                        neighbor.cost = new_cost

                        visited[new_key] = new_cost
                        frontier.push(neighbor, new_cost)

        return None, None, visited

    def sample(self):
        """
        Sample a new configuration
        Returns a configuration of size self.n bounded in self.limits
        """
        return np.random.uniform(self.limits[:, 0], self.limits[:, 1])
