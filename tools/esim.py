from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

H = sim.getObjectsInTree(sim.handle_scene, sim.handle_all)
for h in H:
    t = sim.getObjectType(h)
    a = sim.getObjectAlias(h)
    print(h,t,a)
    