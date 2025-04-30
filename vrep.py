"""
ref: https://manual.coppeliarobotics.com/index.html > Regular API reference

@author: Seongil Heo
"""

import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from scipy.optimize import minimize

# ROBOT = "LBRiiwa7R800"
ROBOT = "LBRiiwa14R820"
class Arm:
    def __init__(self, robot: int, start, goal, joint_names, joint_handles, link_names, link_handles, gripper, sim, simIK):
        self.sim = sim
        self.simIK = simIK
        self.start = start
        self.goal = goal
        self.joint_names = joint_names
        self.joint_handles = joint_handles
        self.link_names = link_names
        self.link_handles = link_handles
        self.gripper = gripper

        self.init_ik(robot)
        
        self.lims = [self.sim.getJointInterval(self.joint_handles[i])[1] for i in range(len(self.joint_handles))]
    
    def init_ik(self, robot: int):
        
        simBase = -1
        # Create IK environment
        self.ikEnv = self.simIK.createEnvironment()
        self.ikBase = self.simIK.createDummy(self.ikEnv)
        self.simTarget = self.sim.getObject("/Target")

        self.simIK.setObjectMatrix(self.ikEnv, self.ikBase, -1, self.sim.getObjectMatrix(self.sim.getObject(f"/{ROBOT}[{robot}]/joint1")))

        parent = self.ikBase
        
        # Duplicate IK Joints
        self.ikJoints = []
        for i in range(len(self.joint_handles)):
            self.ikJoints.append(self.simIK.createJoint(self.ikEnv, self.simIK.jointtype_revolute))
            self.simIK.setJointMode(self.ikEnv, self.ikJoints[i], self.simIK.jointmode_ik)
            cyclic, interv = self.sim.getJointInterval(self.joint_handles[i])
            self.simIK.setJointInterval(self.ikEnv, self.ikJoints[i], cyclic, interv)
            self.simIK.setJointPosition(self.ikEnv, self.ikJoints[i], self.sim.getJointPosition(self.joint_handles[i]))
            self.simIK.setObjectMatrix(self.ikEnv, self.ikJoints[i], simBase, self.sim.getObjectMatrix(self.joint_handles[i]))
            self.simIK.setObjectParent(self.ikEnv, self.ikJoints[i], parent, True) 
            parent = self.ikJoints[i]

        # Duplicate IK Tip
        self.ikTip = self.simIK.createDummy(self.ikEnv)
        self.simIK.setObjectMatrix(self.ikEnv, self.ikTip, simBase, self.sim.getObjectMatrix(self.gripper))
        self.simIK.setObjectParent(self.ikEnv, self.ikTip, parent, True)
        
        # Duplicate IK Target
        self.ikTarget = self.simIK.createDummy(self.ikEnv)
        self.simIK.setObjectMatrix(self.ikEnv, self.ikTarget, simBase, self.sim.getObjectMatrix(self.simTarget))
        self.simIK.setTargetDummy(self.ikEnv, self.ikTip, self.ikTarget)
        
        # Undamped IK Group
        self.ikGroup_undamped = self.simIK.createGroup(self.ikEnv)
        self.simIK.setGroupCalculation(self.ikEnv, self.ikGroup_undamped, self.simIK.method_pseudo_inverse, 0, 6)
        ikElementHandle = self.simIK.addElement(self.ikEnv, self.ikGroup_undamped, self.ikTip)
        self.simIK.setElementBase(self.ikEnv, self.ikGroup_undamped, ikElementHandle, self.ikBase)
        self.simIK.setElementConstraints(self.ikEnv, self.ikGroup_undamped, ikElementHandle, self.simIK.constraint_pose)
        
        # Damped IK Group
        self.ikGroup_damped = self.simIK.createGroup(self.ikEnv)
        self.simIK.setGroupCalculation(self.ikEnv, self.ikGroup_damped, self.simIK.method_damped_least_squares, 1, 99)
        ikElementHandle = self.simIK.addElement(self.ikEnv, self.ikGroup_damped, self.ikTip)
        self.simIK.setElementBase(self.ikEnv, self.ikGroup_damped, ikElementHandle, self.ikBase)
        self.simIK.setElementConstraints(self.ikEnv, self.ikGroup_damped, ikElementHandle, self.simIK.constraint_pose)

            
    def ik_vrep(self, target):
        self.sim.setObjectPosition(self.simTarget, target.tolist())

        # Compute IK
        res, *_ = self.simIK.handleGroup(self.ikEnv, self.ikGroup_undamped)
        if res != self.simIK.result_success:
            res, *_ = self.simIK.handleGroup(self.ikEnv, self.ikGroup_damped)
            if res != self.simIK.result_success:
                print("Both solvers failed")
            else:
                print("Undamped solver failed, damped solver success")

        # Load IK Joints
        joint_positions = []
        for i in range(len(self.ikJoints)):
            joint_positions.append(self.simIK.getJointPosition(self.ikEnv, self.ikJoints[i]))

        return np.array(joint_positions)
    
    def cost(self, q, target):

        for joint, angle in zip(self.joint_handles, q):
            self.sim.setJointPosition(joint, angle)
        
        pos_error = np.linalg.norm(self.sim.getObjectPosition(self.gripper) - target)

        return pos_error
    
    def ik(self, target, q_init=None):
        if q_init is None:
            q_init = [1.57,0,0,0,0,0,0]   

        result = minimize(
            fun=lambda q: self.cost(q, target),
            x0=np.array(q_init),
            bounds=self.lims,
            method='trust-constr'
        )

        return result.x

class VrepWrapper:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simIK = self.client.require('simIK')

        self.removedummy()

        # self.start = np.array([-1.5,0,0.5])
        self.start = np.array([0,0.3,0.7])
        # self.goal = np.array([1.5,0,0.5])
        self.goal = np.array([0,-0.3,0.3])
        self.handover = np.zeros(3)
        
        self.dim = 3
        self.num_joints = 7
        self.num_links = 8        

        self.lims = np.array([[-np.pi, np.pi] for _ in range(self.num_joints)])

        robot1_joint_names = [f"/{ROBOT}[0]/joint{i}" for i in range(1, 1+self.num_joints)]#+[f"/{ROBOT}[0]/connection"]
        robot1_link_names = [f"/{ROBOT}[0]/link{i}" for i in range(1, 1 + self.num_links)]
        robot1_joint_handles = [self.sim.getObject(joint_name) for joint_name in robot1_joint_names]
        robot1_link_handles = [self.sim.getObject(joint_name) for joint_name in robot1_link_names]
        robot1_gripper = self.sim.getObject(f"/{ROBOT}[0]/connection")
        for joint in robot1_joint_handles:
            self.sim.setJointPosition(joint, 0)

        # robot 2
        robot2_joint_names = [f"/{ROBOT}[1]/joint{i}" for i in range(1, 1+self.num_joints)]#+[f"/{ROBOT}[1]/connection"]
        robot2_link_names = [f"/{ROBOT}[1]/link{i}" for i in range(1, 1 + self.num_links)]
        robot2_joint_handles = [self.sim.getObject(joint_name) for joint_name in robot2_joint_names]
        robot2_link_handles = [self.sim.getObject(joint_name) for joint_name in robot2_link_names]
        robot2_gripper = self.sim.getObject(f"/{ROBOT}[1]/connection")
        for joint in robot2_joint_handles:
            self.sim.setJointPosition(joint, 0)
        
        # robot 1
        self.robot1 = Arm(
            robot = 0,
            sim = self.sim,
            simIK = self.simIK,
            start = self.start,
            goal = self.handover,
            joint_names = robot1_joint_names,
            joint_handles = robot1_joint_handles,
            link_names= robot1_link_names,
            link_handles= robot1_link_handles,
            gripper = robot1_gripper,
        )
        # robot 2
        self.robot2 = Arm(
            robot = 1,
            sim = self.sim,
            simIK = self.simIK,
            start = self.handover,
            goal = self.goal,
            joint_names = robot2_joint_names,
            joint_handles = robot2_joint_handles,
            link_names= robot2_link_names,
            link_handles= robot2_link_handles,
            gripper = robot2_gripper,
        )
        
        # color RGB
        red = [1,0,0]
        green = [0,1,0]
        blue = [0,0,1]

        # floor
        floor_name = "/Floor"
        self.floor = self.sim.getObject(floor_name)
        # table
        self.table = None
        # start
        self.start_handle = self.sim.getObject("/Start")
        # handover
        self.handover_handle = self.sim.getObject("/Handover")
        # end
        self.end_handle = self.sim.getObject("/End")

        # Drawing
        point_obj = self.sim.drawing_points     # object type
        line_obj = self.sim.drawing_lines
        
        self.points = self.sim.addDrawingObject( # drawing objects 
            point_obj,                  # objectType
            5,                          # size
            0,                          # duplicateTolerance
            -1,                         # parentObjectHandle
            1000000,                    # maxItemCount
            blue                        # color
        )
        self.importantPoints = self.sim.addDrawingObject(
            point_obj, 10, 0, -1, 1000000, red
        )
        self.edges = self.sim.addDrawingObject(
            line_obj, 1, 0, -1, 1000000, blue
        )
        self.plan = self.sim.addDrawingObject( 
            line_obj, 8, 0, -1, 1000000, green
        )

        self.test_collisions = self.testCollision
        # self.test_collisions = self.fake_in_collision

    def testCollision(self, state, name):
        formatted = np.reshape(np.array(state),(-1,self.num_joints))
        collides,fk = self.checkCollision(formatted, name)
        return np.sum(collides)>0

    def runTrajectory(self, robot_name, plan):
        # Select robot
        if robot_name == "robot1":
            joint_handles = self.robot1.joint_handles
        elif robot_name == "robot2":
            joint_handles = self.robot2.joint_handles
        # Move along plan
        for angles in plan:
            for i in range(self.num_joints): 
                self.sim.setJointPosition(joint_handles[i], angles[i])
            time.sleep(0.01)

    def checkCollision(self, states, name=None):
        num_states = len(states)
        single_dim_states = np.reshape(states,-1)

        # Select robot
        if name == "robot1":
            joint_handles = self.robot1.joint_handles
            link_handles = self.robot1.link_handles 
            gripper = self.robot1.gripper
            other_handles = self.robot2.link_handles
        elif name == "robot2":
            joint_handles = self.robot2.joint_handles
            link_handles = self.robot2.link_handles
            gripper = self.robot2.gripper
            other_handles = self.robot1.link_handles

        # backup_position = [self.sim.getJointPosition(joint_handle) for joint_handle in joint_handles]

        fk = np.zeros((num_states, self.dim))
        collisions = np.zeros(num_states)

        # Set joint position (temp)
        for idx in range(num_states):
            for j in range(self.num_joints):
                self.sim.setJointPosition(joint_handles[j], single_dim_states[idx*7 + j])
            
            # Check collision with other
            for li in range(self.num_links):
                for lj in range(self.num_links):
                    collide, collidingObjectHandles = self.sim.checkCollision(link_handles[li], other_handles[lj])
                    if collide:
                        break
                if collide:
                    break
            
            # Check collision with floor
            for l in range(1, self.num_links):
                collide, collidingObjectHandles = self.sim.checkCollision(link_handles[l], self.floor) 
                if collide !=0:
                    break

            # Get end-effector's position
            pose = self.sim.getObjectPosition(gripper)
            fk[idx, :] =  pose
            collisions[idx] = collide

        # for l in range(self.num_joints):
            # self.sim.setJointPosition(self.joint_handles[l], backup_position[l])

        return collisions, fk
    
    def addPoint(self, point, isEnd = False):
        handle = self.importantPoints if isEnd else self.points
        self.sim.addDrawingObjectItem(handle, point.tolist())

    def addLine(self, line, isPlan = False):
        handle = self.plan if isPlan else self.edges
        self.sim.addDrawingObjectItem(handle,line.tolist()) # drawingObjectHandle, itemData

    def setJointPosition(self, robot, angles):
        for j in range(self.num_joints-1):
            self.sim.setJointPosition(robot.joint_handles[j], angles[j])
        fk = self.sim.getObjectPosition(robot.gripper)
        self.addPoint(np.array(fk))
        return fk
            
    def draw_plan(self, plan1, plan2, planner1, planner2, dynamic_tree=False, dynamic_plan=True, show=True):
        #dynamic_tree, dynamic_plan and show are all dummy values and do not function
        
        # Draw tree
        if planner1 is not None:
            print("Drawing robot1's tree")
            Qs, edges = planner1.T.get_states_and_edges()
            total_nodes = np.reshape(np.array(edges),(-1,7))
            for idx in range(len(total_nodes)):
                fk = self.setJointPosition(self.robot1, total_nodes[idx])

        if planner2 is not None:
            print("Drawing robot2's tree")
            Qs, edges = planner1.T.get_states_and_edges()
            total_nodes = np.reshape(np.array(edges),(-1,7))
            for idx in range(len(total_nodes)):
                fk = self.setJointPosition(self.robot2, total_nodes[idx])

        line = np.zeros((len(plan1),6))
        for idx in range(len(plan1)):                                   # robot 1
            fk = self.setJointPosition(self.robot1, plan1[idx])
            if idx:
                line[idx,3:] = fk
                if idx+1 < line.shape[0]:
                    line[idx+1,:3] = fk
                self.addLine(line[idx], isPlan=True)
            else:
                line[idx,:3] = fk


        line = np.zeros((len(plan2),6))
        for idx in range(len(plan2)):                                   # robot 2
            fk = self.setJointPosition(self.robot2, plan2[idx])
            if idx:
                line[idx,3:] = fk
                if idx+1 < line.shape[0]:
                    line[idx+1,:3] = fk
                self.addLine(line[idx], isPlan=True)
            else:
                line[idx,:3] = fk

        # self.runTrajectory("robot1" ,plan1)
        # self.runTrajectory("robot2", plan2)

    def find_handover_point(self, num_samples=100):
        self.handover=np.array([0,0,0.5])

    def set_start_goal_config(self):
        self.find_handover_point()
        
        # Draw start and goal
        self.sim.setObjectPosition(self.start_handle, self.start.tolist())
        self.sim.setObjectPosition(self.handover_handle, self.handover.tolist())
        self.sim.setObjectPosition(self.end_handle, self.goal.tolist())

        self.robot1.start = self.robot1.ik(self.start)
        self.robot1.goal = self.robot1.ik(self.handover)
        self.robot2.start = self.robot2.ik(self.handover)
        self.robot2.goal = self.robot2.ik(self.goal)

        for ps in [self.robot1.start, self.robot1.goal, self.robot2.start, self.robot2.goal]:
            print(tuple(f"{p:+2.09f}" for p in ps ))

    def vrepStop(self):
        self.sim.stopSimulation()

    def vrepReset(self):
        self.sim.stopSimulation()
        self.__init__()    

    def removedummy(self):
        i = 0
        while True:
            try: 
                self.sim.removeDrawingObject(i)
                i+=1
            except:
                break

    def fake_in_collision(self,q, name):
        """
        We never collide with this function!
        """
        return False
