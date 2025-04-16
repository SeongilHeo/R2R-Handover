from collisions_3D import RobotArm, pi, np
import matplotlib.pyplot as plt

ds = [10, 0, 0, 10, 0, 10, 10]
avs = [0, 10, 10, 0, 10, 0, 0]
alphas = [pi/2, 0, -pi/2, pi/2, 0, -pi/2, 0]

robot = RobotArm(ds, avs, alphas)

q = np.zeros(7)
robot.draw(q, show=True)