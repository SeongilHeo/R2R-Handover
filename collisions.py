#!/usr/bin/env python
"""
Package providing helper classes and functions for performing graph search operations for planning.
"""
import numpy as np
import matplotlib.pyplot as plotter
from math import cos, sin, pi
from scipy.optimize import minimize
from time import time

_DEBUG = False
_BOUNDS = "Bounds"
_GOAL = "Goal"
_OBSTACLE = "Obstacle"
_TABLE = "Table"
_START = "Start"
_ROBOT = "RobotLinks"
_ROBOT_LOC = "RobotBase"

class Table:
    def __init__(self, x=None, y=None):
        self.x = x if x else [-150,150] 
        self.y = y if y else [-10,0]

    def draw(self, color="lightgray", show=False):
        plotter.fill_between(self.x, self.y[0], self.y[1], color=color)
        if show:
            plotter.show(block=True)


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
        self.start = None # np.array([1,57, 0 ,0])
        self.goal = None # np.zeros(3)
        self.lims = np.array([[0, pi], [-pi, pi], [-pi, pi]])

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
    
    def cost(self, q, target):
        pts = self.fk(q)
        pos_error = np.linalg.norm(pts[-1] - target)

        return pos_error
    
    def table_constraint(self, q):
        pts = self.fk(q)
        y_coords = pts[:, 1]
        return y_coords  # This means: each y >= 0
    
    
    def ik(self, target, q_init=None):
        if q_init is None:
            q_init = [1.57,0,0]     

        constraints = [{
            'type': 'ineq',
            'fun': lambda q: self.table_constraint(q)  # each y >= 0
        }]

        result = minimize(
            fun=lambda q: self.cost(q, target),
            x0=np.array(q_init),
            bounds=self.lims,
            constraints=constraints,
            method='trust-constr'
        )

        return result.x

    
    # def compute_jacobian(self, q):
    #     """
    #     Compute Jacobian matrix for planar robot
    #     :param q: Joint angles [θ₁, θ₂, θ₃]
    #     :return: Jacobian (2 x n)
    #     """
    #     n = len(self.link_lengths)
    #     J = np.zeros((2, n))
        
    #     # Sum of angles up to each joint
    #     sum_angles = np.cumsum(q)
        
    #     # Total sum of all angles (for the end effector)
    #     total_angle = sum_angles[-1] if n > 0 else 0.0
        
    #     # Compute Jacobian columns
    #     for i in range(n):
    #         # Sum of angles up to joint i (θ₁ + θ₂ + ... + θᵢ)
    #         sum_up_to_i = sum_angles[i]
            
    #         # Contribution of joint i to the end-effector position
    #         dx = 0.0
    #         dy = 0.0
            
    #         # For each link after joint i (including link i)
    #         for j in range(i, n):
    #             # Sum of angles up to link j (θ₁ + θ₂ + ... + θⱼ)
    #             sum_up_to_j = sum_angles[j]
                
    #             dx -= self.link_lengths[j] * sin(sum_up_to_j)
    #             dy += self.link_lengths[j] * cos(sum_up_to_j)
            
    #         J[0, i] = dx
    #         J[1, i] = dy
        
    #     return J



    def draw(self, q=None, color="b", show=False, base_color="g"):
        """
        Draw the robot with the provided configuration
        """
        # q = self.start
        pts = self.fk(q)
        for i, p in enumerate(pts):
            if i == 0:
                plotter.plot(p[0], p[1], color=base_color, marker='s')
            elif i == len(pts) - 1:
                gx,gy=self.get_gripper_pose(pts)
                plotter.plot(gx, gy, color=color, linewidth=3)
            else:
                plotter.plot(p[0], p[1], color=color, marker='o')
            if i > 0:
                plotter.plot([prev_p[0], p[0]], [prev_p[1], p[1]], color)
            prev_p = p[:]
        if show:
            plotter.show(block=True)

    def get_gripper_pose(self, pts):
        prev_p = pts[-2]
        end_p = pts[-1]
        dx = end_p[0] - prev_p[0]
        dy = end_p[1] - prev_p[1]
        angle = np.arctan2(dy, dx)
        length = 5  # length of the gripper
        gx = np.array([end_p[0] - length * np.sin(angle), end_p[0] + length * np.sin(angle)])
        gy = np.array([end_p[1] + length * np.cos(angle), end_p[1] - length * np.cos(angle)])

        return gx, gy
    
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
        self.robot1 = None
        self.robot_base_1 = np.zeros(2)
        self.robot2 = None
        self.robot_base_2 = np.zeros(2)
        self.table = Table()
        self.goal = None  # In configuration space
        self.handover = None  # In configuration space
        self.start = None  # In configuration space
        self.line_parser = {
            _BOUNDS: self.parse_bounds,
            _GOAL: self.parse_goal,
            _OBSTACLE: self.parse_obstacle,
            _TABLE: self.parse_table,
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

    def parse_table(self, line_data):
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
        
        # Add to obstacles
        table = np.array(pts)
        self.polygons.append(table)
        # Set table configuration
        self.table.x = [np.min(table[:, 0]), np.max(table[:, 0])]
        self.table.y = [np.min(table[:, 1]), np.max(table[:, 1])]

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
            self.robot1 = RevoluteRobotChain([float(l) for l in link_data], self.robot_base_1)
            self.lims = self.robot1.lims
        elif idx == 2:
            self.robot2 = RevoluteRobotChain([float(l) for l in link_data], self.robot_base_2)
            self.lims = self.robot2.lims


    def parse_robot_base(self, base_data, idx=None):
        if idx == 1:
            self.robot_base_1 = np.array([float(p) for p in base_data])
            if self.robot1 is not None:
                self.robot1.root = self.robot_base_1[:]
        elif idx == 2:
            self.robot_base_2 = np.array([float(p) for p in base_data])
            if self.robot2 is not None:
                self.robot2.root = self.robot_base_2[:]


    def test_collisions(self, q, name): #q1=None, q2=None):
        """
        Test collision for a specified robot configuration q and the environment env
        """
        if name == "robot1":
            robot_pts = self.robot1.fk(q)
        elif name == "robot2":
            robot_pts = self.robot2.fk(q)
    
        robot_links = []
        prev_pt = robot_pts[0]
        for pt in robot_pts[1:]:
            robot_links.append((prev_pt, pt))
            prev_pt = pt[:]

        for poly_num, polygon in enumerate(self.polygons):
            if _DEBUG:
                print("polygon", polygon)
            for link_num, link in enumerate(robot_links):
                if self.point_in_polygon(link[1], polygon):
                    print("polygon in point")
                    return True
                if link[1][1] < 0:
                    print("check",link)
                
                if _DEBUG:
                    print("Testing link", link_num)
                for i in range(len(polygon)):
                    if _DEBUG:
                        print("\nTestint pt", i, "on polygon", poly_num)
                    prev_pt = polygon[i - 1]
                    pt = polygon[i]
                    if self.line_line_collision(link, (prev_pt, pt)):
                        if _DEBUG:
                            print("Collision between", link, (prev_pt, pt))
                        return True
        return False

    def line_line_collision(self, l1, l2, eps=0.0001):
        '''
        Test collision between two line segments l1 and l2
        '''
        if _DEBUG:
            print('l1', l1)
            print('l2', l2)

        a1 = l1[0]
        a2 = l1[1]
        b1 = l2[0]
        b2 = l2[1]

        denom = (a1[0] - a2[0])*(b1[1]-b2[1]) - (a1[1]-a2[1])*(b1[0]-b2[0])
        if denom == 0: # parallel lines
            if _DEBUG:
                print('Parallel lines cant intersect')
            return False

        # Get intersection point
        x_i = ((a1[0]*a2[1] - a1[1]*a2[0])*(b1[0]-b2[0]) -
               (a1[0] - a2[0])*(b1[0]*b2[1] - b1[1]*b2[0]))/denom
        y_i = ((a1[0]*a2[1] - a1[1]*a2[0])*(b1[1]-b2[1]) -
               (a1[1] - a2[1])*(b1[0]*b2[1] - b1[1]*b2[0]))/denom
        if _DEBUG:
            print('a1', a1)
            print('a2', a2)
            print('b1', b1)
            print('b2', b2)
            print('(x_i, y_i) = (', x_i, ',', y_i,')')

        # Test if intersection point between bounds
        if x_i < min(a1[0], a2[0])-eps or x_i > max(a1[0], a2[0])+eps:
            if _DEBUG:
                print('x not in a bounds')
            return False
        if x_i < min(b1[0], b2[0])-eps or x_i > max(b1[0], b2[0])+eps:
            if _DEBUG:
                print('x not in b bounds')
            return False
        if y_i < min(a1[1], a2[1])-eps or y_i > max(a1[1], a2[1])+eps:
            if _DEBUG:
                print('y not in a bounds')
            return False
        if y_i < min(b1[1], b2[1])-eps or y_i > max(b1[1], b2[1])+eps:
            if _DEBUG:
                print('y not in b bounds')
            return False
        return True

    def point_in_polygon(self, pt, poly):
        """
        Determine if a point lies within a polygon
        """
        n = len(poly)
        inside = False

        p1 = poly[0]
        for i in range(n+1):
            p2 = poly[i % n]
            try:
                if (pt[1] > min(p1[1],p2[1]) and pt[1] <= max(p1[1],p2[1]) and
                    pt[0] <= max(p1[0],p2[0])):
                    if p1[1] != p2[1]:
                        x_cross = (pt[1] - p1[1])*(p2[0] - p1[0])/(p2[1]-p1[1])+p1[0]
                    if p1[0] == p2[0] or pt[0] <= x_cross:
                        inside = not inside
            except ValueError:
                print('pt', pt)
                print('poly', poly)
            p1 = p2

        return inside

    def draw_env(self, p1=None, p2=None, q1=None, q2=None, show=False):
        """
        Draw the environment obstacle map
        """

        plotter.figure(figsize=(10, 5))
        plotter.axis([self.x_min-10, self.x_max+10, self.y_min-50, self.y_max+50])

        # Draw Table
        self.table.draw()

        # Set start and goal
        if (self.start.shape[0] == 3):                                              # option#1: configuration
            start_fk = self.robot1.fk(self.start)
            start_x = start_fk[-1]

            goal_fk = self.robot2.fk(self.goal)
            goal_x = goal_fk[-1]
        else:                                                                       # option#2: coordinate
            start_x = self.start
            goal_x = self.goal

        # Set robots position   (x,y) -> (theta_1, theta_2, theta_3)
        if self.robot1.start is not None:          # robot 1
            q1 = self.robot1.start
        elif p1:
            q1 = self.robot1.ik(p1)

        if self.robot2.start is not None:          # robot 2
            q2 = self.robot2.start
        elif p2: 
            q2 = self.robot2.ik(p2)
        # Draw robots
        if q1 is not None:                          # robot 1                       #  drawing
            self.robot1.draw(q1)
        else:
            self.robot1.draw()

        if q2 is not None:                          # robot 2
            self.robot2.draw(q2)
        else:
            self.robot2.draw()

        # Draw start and goal
        plotter.plot(start_x[0], start_x[1], "ro", markersize=8)                    # start 
        plotter.plot(goal_x[0], goal_x[1], "mo", markersize=8)                      # goal
        if self.handover:
            plotter.plot(self.handover[0], self.handover[1], "yo", markersize=8)    # handover

        if show:
            plotter.show()

    def draw_plan(
        self,
        plan1,
        plan2,
        planner1,
        planner2,
        dynamic_tree=False,
        dynamic_plan=True,
        show=False,
        save=True
    ):
        """
        Draw the environment with an overlaid plan.
        plan - sequence of configurations to be drawn as plan (not drawn if pass in None)
        planner - a planner which has a function of the form
                  vertices, edges = planner.T.get_states_and_edges()
                  if None the search graph is not drawn
        """
        self.draw_env(show=False)
        if save:
            import os
            import imageio.v2 as imageio

            folder_path = "frames"
            os.makedirs(folder_path, exist_ok=True)
            frame_i=0

            plotter.savefig(f"frames/frame_{frame_i:04d}.png")
            frame_i+=1

        plotter.ion()
        if show:
            plotter.show()

        if planner1 is not None:
            Qs, edges = planner1.T.get_states_and_edges()
            # Draw tree for each of the robot links
            for i, e in enumerate(edges):
                X0 = self.robot1.fk(e[0])
                X1 = self.robot1.fk(e[1])
                e0 = X0[-1]
                e1 = X1[-1]
                plotter.plot([e0[0], e1[0]], [e0[1], e1[1]], "b")
                plotter.plot([e0[0], e1[0]], [e0[1], e1[1]], "b.")
                if dynamic_tree:
                    plotter.pause(0.001)
                if save:
                    plotter.savefig(f"frames/frame_{frame_i:04d}.png")
                    frame_i+=1

        if planner2 is not None:
            Qs, edges = planner2.T.get_states_and_edges()
            # Draw tree for each of the robot links
            for i, e in enumerate(edges):
                X0 = self.robot2.fk(e[0])
                X1 = self.robot2.fk(e[1])
                e0 = X0[-1]
                e1 = X1[-1]
                plotter.plot([e0[0], e1[0]], [e0[1], e1[1]], "b")
                plotter.plot([e0[0], e1[0]], [e0[1], e1[1]], "b.")
                if dynamic_tree:
                    plotter.pause(0.001)
                if save:
                    plotter.savefig(f"frames/frame_{frame_i:04d}.png")
                    frame_i+=1


        # # Draw goal
        # goal_fk = self.robot.fk(self.goal)
        # goal_x = goal_fk[-1]
        # plotter.plot(goal_x[0], goal_x[1], "go")
        # plotter.plot(goal_x[0], goal_x[1], "g.")
        # # Draw start
        # start_fk = self.robot.fk(self.start)
        # start_x = start_fk[-1]
        # plotter.plot(start_x[0], start_x[1], "ro")
        # plotter.plot(start_x[0], start_x[1], "r.")

        if plan1 is not None:
            self.robot1.draw(plan1[0], color="g")
            for i in range(len(plan1)):
                Qp = plan1[i - 1]
                Qr = plan1[i]
                Ps = self.robot1.fk(Qp)
                Rs = self.robot1.fk(Qr)
                r_prev = None
                for p, r in zip(Ps, Rs):
                    plotter.plot(r[0], r[1], "g.")
                    if i != 0:
                        plotter.plot(p[0], p[1], "g.")
                        plotter.plot([p[0], r[0]], [p[1], r[1]], "c")
                        if r_prev is not None:
                            plotter.plot([r_prev[0], r[0]], [r_prev[1], r[1]], "g")
                    r_prev = r[:]
                plotter.plot(Rs[-1][0],Rs[-1][1], "ro")
                if dynamic_plan:
                    plotter.pause(0.1)
                if save:
                    plotter.savefig(f"frames/frame_{frame_i:04d}.png")
                    frame_i+=1



        if plan2 is not None:
            self.robot2.draw(plan2[0], color="g")
            for i in range(len(plan2)):
                Qp = plan2[i - 1]
                Qr = plan2[i]
                Ps = self.robot2.fk(Qp)
                Rs = self.robot2.fk(Qr)
                r_prev = None
                for p, r in zip(Ps, Rs):
                    plotter.plot(r[0], r[1], "g.")
                    if i != 0:
                        plotter.plot(p[0], p[1], "g.")
                        plotter.plot([p[0], r[0]], [p[1], r[1]], "c")
                        if r_prev is not None:
                            plotter.plot([r_prev[0], r[0]], [r_prev[1], r[1]], "g")
                    r_prev = r[:]
                plotter.plot(Rs[-1][0],Rs[-1][1], "ro")
                if dynamic_plan:
                    plotter.pause(0.01)
                if save:
                    plotter.savefig(f"frames/frame_{frame_i:04d}.png")
                    frame_i+=1


            self.robot1.draw(plan1[-1], color="r")
            self.robot2.draw(plan2[-1], color="r")
        if save:
            plotter.savefig(f"frames/frame_{frame_i:04d}.png")

            # files = os.listdir(folder_path)
            # files = [os.path.join(folder_path, f) for f in files]
            # files.sort()
            with imageio.get_writer("robot_motion.gif", mode='I', duration=0.1) as writer:
                for idx in range(frame_i):
                    fname = f"frames/frame_{idx:04d}.png"
                    image = imageio.imread(fname)
                    writer.append_data(image)
                    os.remove(fname)
        
        # stay on ending frame
        if show:
            plotter.show(block=True)


    def find_handover_point(self, num_samples=100):
        self.handover=[0,50]
        return
        center = (self.start + self.goal) / 2
        max_radius = 100
        num_rings = 10
        samples_per_ring = num_samples // num_rings

        for i in range(1, num_rings + 1):
            radius = max_radius * (i / num_rings)
            for _ in range(samples_per_ring):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, radius)
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                if y < self.table.y[0]:  # skip points below the table
                    continue
                point = np.array([x, y])
                try:
                    q1 = self.robot1.ik(point)
                    q2 = self.robot2.ik(point)
                    if np.all(self.robot1.fk(q1)[:, 1] >= self.table.y[0]) and np.all(self.robot2.fk(q2)[:, 1] >= self.table.y[0]):
                        self.handover = point
                        return
                except:
                    continue

        raise ValueError("No common reachable handover point found!")

    def set_start_goal_config(self, ):
        
        self.find_handover_point()

        self.robot1.start = self.robot1.ik(self.start)
        self.robot1.goal = self.robot1.ik(self.handover)
        self.robot2.start = self.robot2.ik(self.handover)
        self.robot2.goal = self.robot2.ik(self.goal)


