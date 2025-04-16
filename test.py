from collisions import PolygonEnvironment

problem ="env.txt"

environment = PolygonEnvironment()
environment.read_env(problem)
# environment.draw_env(q1=[-3.14, 0, 0],q2=[1,-1.57,0.3])
environment.draw_env(p1=[-100, 0],p2=[100,0])