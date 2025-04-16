#!/usr/bin/env python
"""
Package providing helper classes and functions for performing graph search operations for planning.
"""
import numpy as np
import matplotlib.pyplot as plotter
from math import cos, sin, pi

_DEBUG = False
_BOUNDS = "Bounds"
_GOAL = "Goal"
_OBSTACLE = "Obstacle"
_START = "Start"
_ROBOT = "RobotLinks"
_ROBOT_LOC = "RobotBase"


class RevoluteRobotChain:
    def __init__(self, link_lengths, root=None):
        """
        Build a robot comprised of revolute joints with links of length provided and the
        """
        self.link_lengths = link_lengths
        if root is None:
            self.root = np.zeros(2)
        else:
            self.root = root
        self.lims = np.array([[-pi, pi], [-pi, pi], [-pi, pi]])

    def fk(self, q):
        """
        Compute forward kinematics for the robot with configuration q
        Returns a tuple of form (Pts, Thetas)
        Pts - an array of (x,y) coordinates of the links ends
        Thetas - list of link end angles
        """
        beta = 0.0
        X = [self.root[0]]
        Y = [self.root[1]]
        Thetas = [0.0]
        for i in range(len(self.link_lengths)):
            beta += q[i]
            x_i = self.link_lengths[i] * cos(beta)
            y_i = self.link_lengths[i] * sin(beta)
            X.append(X[-1] + x_i)
            Y.append(Y[-1] + y_i)
            Thetas.append(beta)
        pts = []
        for x, y in zip(X, Y):
            pts.append((x, y))
        pts = np.array(pts)

        return pts
    
    def ik(self, target, q_init=None, max_iter=100, tol=1e-3, alpha=0.1):
        """
        Compute Inverse Kinematics using Jacobian Transpose method
        """
        target = target - self.root
        n = len(self.link_lengths)
        q = np.zeros(n) if q_init is None else np.array(q_init)

        for _ in range(max_iter):
            pts = self.fk(q)
            end_effector = pts[-1]
            error = np.array(target) - end_effector

            if np.linalg.norm(error) < tol:
                break

            J = self.compute_jacobian(q)
            dq = alpha * J.T @ error
            q += dq

            # 관절 제한 적용
            q = np.clip(q, self.lims[:, 0], self.lims[:, 1])

        return q


    def compute_jacobian(self, q):
        """
        Compute Jacobian matrix for planar robot
        :param q
        :return: Jacobian (2 x n)
        """
        n = len(q)
        J = np.zeros((2, n))
        pts = self.fk(q)
        end_effector = pts[-1]

        theta = 0.0
        for i in range(n):
            theta += q[i]
            r = end_effector - pts[i]
            J[0, i] = -self.link_lengths[i] * sin(theta)
            J[1, i] =  self.link_lengths[i] * cos(theta)

        return J



    def draw(self, q=None, color="b", show=False, base_color="g"):
        """
        Draw the robot with the provided configuration
        """
        pts = self.fk(q)
        for i, p in enumerate(pts):
            if i == 0:
                style = base_color + "o"
            else:
                style = color + "o"
            plotter.plot(p[0], p[1], style)
            if i > 0:
                plotter.plot([prev_p[0], p[0]], [prev_p[1], p[1]], color)
            prev_p = p[:]
        if show:
            plotter.show(block=True)


class PolygonEnvironment:
    """
    A simple class to store polygon obstacle environments
    """

    def __init__(self):
        """
        Create the storage types needed for the class
        robot - an instance of RevoluteRobotChain class or 2DPointRobot class
        """
        self.polygons = []
        self.robot_1 = None
        self.robot_base_1 = np.zeros(2)
        self.robot_2 = None
        self.robot_base_2 = np.zeros(2)
        self.goal = None  # In configuration space
        self.start = None  # In configuration space
        self.line_parser = {
            _BOUNDS: self.parse_bounds,
            _GOAL: self.parse_goal,
            _OBSTACLE: self.parse_obstacle,
            _START: self.parse_start,
            _ROBOT: self.parse_robot_links,
            _ROBOT_LOC: self.parse_robot_base,
        }

    def read_env(self, env_file_path):
        """
        Read in a map from file that has the form.
        It can read in lines of the four types listed which start which are of the form
        <typename>: vals ...
        The for options are:
        Bounds: x_min x_max y_min y_max
        Goal: goal_q1 goal_q2 ...
        Start: start_q_1 start_q_2 ...
        Obstacle: x1 y1 x2 y2 x3 y3 [x4 y5...xn yn]
        """
        env_file = open(env_file_path, "r")
        file_infos = env_file.readlines()
        for l in file_infos:
            line_info = l.strip().split()
            if line_info[0].startswith("#"):
                continue
            if "_" in line_info[0]:
                func, idx = line_info[0].split("_")
                self.line_parser[func](line_info[1:], int(idx[0]))
            else:
                self.line_parser[line_info[0][:-1]](line_info[1:])

    def parse_bounds(self, line_data):
        """
        Parse map boundaries
        """
        self.x_min = float(line_data[0])
        self.x_max = float(line_data[1])
        self.y_min = float(line_data[2])
        self.y_max = float(line_data[3])
        self.lims = np.array([[self.x_min, self.x_max], [self.y_min, self.y_max]])

    def parse_obstacle(self, line_data):
        """
        Parse a polygon obstacle line
        """
        vals = [float(x) for x in line_data]
        pts = []
        # Parse pair of values into points for obstacle vertices
        while len(vals) > 0:
            pts.append(np.array(vals[:2]))
            vals = vals[2:]

        if len(pts) < 3:
            print("Need at least 3 points to define an obstacle")
            return
        obstacle = np.array(pts)
        self.polygons.append(obstacle)

    def parse_goal(self, line_data):
        """
        Parse a goal location
        """
        self.goal = np.array([float(l) for l in line_data])

    def parse_start(self, line_data):
        """
        Parse a start location
        """
        self.start = np.array([float(l) for l in line_data])

    def parse_robot_links(self, link_data, idx):
        if idx == 1:
            self.robot_1 = RevoluteRobotChain([float(l) for l in link_data], self.robot_base_1)
            self.lims = self.robot_1.lims
        elif idx == 2:
            self.robot_2 = RevoluteRobotChain([float(l) for l in link_data], self.robot_base_2)
            self.lims = self.robot_2.lims


    def parse_robot_base(self, base_data, idx=None):
        if idx == 1:
            self.robot_base_1 = np.array([float(p) for p in base_data])
            if self.robot_1 is not None:
                self.robot_1.root = self.robot_base_1[:]
        elif idx == 2:
            self.robot_base_2 = np.array([float(p) for p in base_data])
            if self.robot_2 is not None:
                self.robot_2.root = self.robot_base_2[:]


    def test_collisions(self, q1, q2):
        """
        Test collision for a specified robot configuration q and the environment env
        """
        # Get robot links from current 
        robot_pts_1 = self.robot_1.fk(q1)
        robot_pts_2 = self.robot_2.fk(q2)
        
        # Set each robot links 
        # 1
        robot_links_1 = []
        robot_pts = robot_pts_1[0]
        prev_pt = robot_pts_1[0]

        for pt in robot_pts[1:]:
            robot_links_1.append((prev_pt, pt))
            prev_pt = pt[:]
        # 2
        robot_links_2 = []
        robot_pts = robot_pts_2[0]
        prev_pt = robot_pts_2[0]
        for pt in robot_pts[1:]:
            robot_links_2.append((prev_pt, pt))
            prev_pt = pt[:]

        # Check collision between two robots
        for link_num_1, link_1 in enumerate(robot_links_1):
            for link_num_2, link_2 in enumerate(robot_links_2):
                if self.line_line_collision(link_1, link_2):
                    if _DEBUG:
                        print(f"Collision between robot#1's {link_num_1}-th link ({link_1}), robot#2's {link_num_2}-th link ({link_2})")
                    return True
                

        # # Test collision with all polygons
        # for poly_num, polygon in enumerate(self.polygons):
        #     if _DEBUG:
        #         print("polygon", polygon)
        #     for link_num, link in enumerate(robot_links):
        #         if self.point_in_polygon(link[1], polygon):
        #             return True
        #         if _DEBUG:
        #             print("Testing link", link_num)
        #         for i in range(len(polygon)):
        #             if _DEBUG:
        #                 print("\nTestint pt", i, "on polygon", poly_num)
        #             prev_pt = polygon[i - 1]
        #             pt = polygon[i]
        #             if self.line_line_collision(link, (prev_pt, pt)):
        #                 if _DEBUG:
        #                     print("Collision between", link, (prev_pt, pt))
        #                 return True
    
        return False

    def line_line_collision(self, l1, l2, eps=0.0001):
        """
        Test collision between two line segments l1 and l2
        """
        if _DEBUG:
            print("l1", l1)
            print("l2", l2)

        a1 = l1[0]
        a2 = l1[1]
        b1 = l2[0]
        b2 = l2[1]

        denom = (a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0])
        if denom == 0:  # parallel lines
            if _DEBUG:
                print("Parallel lines cant intersect")
            return False

        # Get intersection point
        x_i = (
            (a1[0] * a2[1] - a1[1] * a2[0]) * (b1[0] - b2[0])
            - (a1[0] - a2[0]) * (b1[0] * b2[1] - b1[1] * b2[0])
        ) / denom
        y_i = (
            (a1[0] * a2[1] - a1[1] * a2[0]) * (b1[1] - b2[1])
            - (a1[1] - a2[1]) * (b1[0] * b2[1] - b1[1] * b2[0])
        ) / denom
        if _DEBUG:
            print("a1", a1)
            print("a2", a2)
            print("b1", b1)
            print("b2", b2)
            print("(x_i, y_i) = (", x_i, ",", y_i, ")")

        # Test if intersection point between bounds
        if x_i < min(a1[0], a2[0]) - eps or x_i > max(a1[0], a2[0]) + eps:
            if _DEBUG:
                print("x not in a bounds")
            return False
        if x_i < min(b1[0], b2[0]) - eps or x_i > max(b1[0], b2[0]) + eps:
            if _DEBUG:
                print("x not in b bounds")
            return False
        if y_i < min(a1[1], a2[1]) - eps or y_i > max(a1[1], a2[1]) + eps:
            if _DEBUG:
                print("y not in a bounds")
            return False
        if y_i < min(b1[1], b2[1]) - eps or y_i > max(b1[1], b2[1]) + eps:
            if _DEBUG:
                print("y not in b bounds")
            return False
        return True

    def point_in_polygon(self, pt, poly):
        """
        Determine if a point lies within a polygon
        """
        n = len(poly)
        inside = False

        p1 = poly[0]
        for i in range(n + 1):
            p2 = poly[i % n]
            try:
                if (
                    pt[1] > min(p1[1], p2[1])
                    and pt[1] <= max(p1[1], p2[1])
                    and pt[0] <= max(p1[0], p2[0])
                ):
                    if p1[1] != p2[1]:
                        x_cross = (pt[1] - p1[1]) * (p2[0] - p1[0]) / (
                            p2[1] - p1[1]
                        ) + p1[0]
                    if p1[0] == p2[0] or pt[0] <= x_cross:
                        inside = not inside
            except ValueError:
                print("pt", pt)
                print("poly", poly)
            p1 = p2

        return inside

    def draw_env(self, p1=None, p2=None, q1=None, q2=None, show=True):
        """
        Draw the environment obstacle map
        """
        plotter.figure(figsize=(10, 5))
        plotter.axis([self.x_min-10, self.x_max+10, self.y_min-10, self.y_max+10])

        # # Draw all obstacles
        # for p in self.polygons:
        #     prev_pt = p[-1]
        #     for pt in p:
        #         plotter.plot([prev_pt[0], pt[0]], [prev_pt[1], pt[1]], "r")
        #         prev_pt = pt[:]

        # Draw Table
        x = [-150,150]
        y_start, y_end = 0,-10

        plotter.fill_between(x, y_start, y_end, color='lightgray')

        # Draw start and goal
        if (self.start.shape[0] == 3):              # option#1: configuration
            start_fk = self.robot_1.fk(self.start)
            start_x = start_fk[-1]

            goal_fk = self.robot_2.fk(self.goal)
            goal_x = goal_fk[-1]
        
        else:                                       # option#2: coordinate
            start_x = self.start
            goal_x = self.goal

        # Draw robots
        if p1:
            q1 = self.robot_1.ik(p1)[0]
        if p2: 
            q2 = self.robot_2.ik(p2)[0]

        if q1 is not None:
            self.robot_1.draw(q1)

        if q2 is not None:
            self.robot_2.draw(q2)


        plotter.plot(start_x[0], start_x[1], "ro", markersize=8)
        plotter.plot(goal_x[0], goal_x[1], "ro", markersize=8)

        if show:
            plotter.show()

    def draw_plan(
        self,
        plan,
        planner,
        dynamic_tree=False,
        dynamic_plan=True,
        show=True,
    ):
        """
        Draw the environment with an overlaid plan.
        plan - sequence of configurations to be drawn as plan (not drawn if pass in None)
        planner - a planner which has a function of the form
                  vertices, edges = planner.T.get_states_and_edges()
                  if None the search graph is not drawn
        """
        self.draw_env(q=self.start, show=False)

        plotter.ion()
        if show:
            plotter.show()

        ws_goal = self.robot.fk(self.goal)
        ws_init = self.robot.fk(self.start)

        plotter.plot(ws_goal[-1][0], ws_goal[-1][1], "go")
        plotter.plot(ws_goal[-1][0], ws_goal[-1][1], "g.")
        plotter.plot(ws_init[-1][0], ws_init[-1][1], "ro")
        plotter.plot(ws_init[-1][0], ws_init[-1][1], "r.")
        plotter.pause(0.1)

        if planner is not None:
            Qs, edges = planner.T.get_states_and_edges()
            # Draw tree for each of the robot links
            for i, e in enumerate(edges):
                X0 = self.robot.fk(e[0])
                X1 = self.robot.fk(e[1])
                e0 = X0[-1]
                e1 = X1[-1]
                plotter.plot([e0[0], e1[0]], [e0[1], e1[1]], "b")
                plotter.plot([e0[0], e1[0]], [e0[1], e1[1]], "b.")
                if dynamic_tree:
                    plotter.pause(0.001)

        # Draw goal
        goal_fk = self.robot.fk(self.goal)
        goal_x = goal_fk[-1]
        plotter.plot(goal_x[0], goal_x[1], "go")
        plotter.plot(goal_x[0], goal_x[1], "g.")
        # Draw start
        start_fk = self.robot.fk(self.start)
        start_x = start_fk[-1]
        plotter.plot(start_x[0], start_x[1], "ro")
        plotter.plot(start_x[0], start_x[1], "r.")

        if plan is not None:
            self.robot.draw(plan[0], color="g")
            for i in range(len(plan)):
                Qp = plan[i - 1]
                Qr = plan[i]
                Ps = self.robot.fk(Qp)
                Rs = self.robot.fk(Qr)
                r_prev = None
                for p, r in zip(Ps, Rs):
                    plotter.plot(r[0], r[1], "g.")
                    if i != 0:
                        plotter.plot(p[0], p[1], "g.")
                        plotter.plot([p[0], r[0]], [p[1], r[1]], "c")
                        if r_prev is not None:
                            plotter.plot([r_prev[0], r[0]], [r_prev[1], r[1]], "g")
                    r_prev = r[:]
                if dynamic_plan:
                    plotter.pause(0.01)
            self.robot.draw(plan[-1], color="r", show=True)
