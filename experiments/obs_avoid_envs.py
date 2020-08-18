import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import matplotlib.patches
import numpy as np
from draw_tools import get_colors
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Circle2D:
    def __init__(self, origin, radius=0.5):
        self.origin = origin
        self.radius = radius

    def isCollision(self, point2d):
        iscollision = False
        r = self.radius
        o = self.origin
        dist = np.sqrt(np.square(point2d[0] - o[0]) + np.square(point2d[1] - o[1]))
        if dist < r:
            iscollision = True

        return iscollision

    def plot(self, ax):
        r = self.radius
        o = self.origin
        theta = np.linspace(0, 2 * np.pi, 100)

        circle = matplotlib.patches.Circle((o[0], o[1]), r)
        circle.set_color(get_colors()['dark gray'])
        circle.set_edgecolor(get_colors()['black'])
        circle.set_hatch('////')
        ax.add_patch(circle)


class Polygon2D:
    def __init__(self, coords):
        self.coords = coords
        self.polygon = Polygon(coords)

    def isCollision(self, point2d):
        p = Point(point2d[0], point2d[1])
        return p.within(self.polygon)

    def plot(self, ax):
        # for i in range(len(self.coords) - 1):
        #     p0 = self.coords[i]
        #     p1 = self.coords[i+1]
        #     ax.plot([p0[0], p1[0]], [p0[1], p1[1]],style)

        polygon = matplotlib.patches.Polygon(np.array(self.coords))
        polygon.set_color(get_colors()['dark gray'])
        polygon.set_edgecolor(get_colors()['black'])
        polygon.set_hatch('////')
        ax.add_patch(polygon)



class ObsSet2D:
    def __init__(self):
        self.circle2d = []
        self.polygon2d = []
        self.poses = []

    def add_circle_obs(self, origin, radius):
        c = Circle2D(origin, radius)
        self.circle2d.append(c)

    def add_polygon_obs(self, coords):
        p = Polygon2D(coords)
        self.polygon2d.append(p)

    def add_openRec_obs(self, pos, r, width, length):
        wx = np.cos(np.pi/2 + r) * width
        wy = np.sin(np.pi/2 + r) * width
        lx = np.cos(r) * length
        ly = np.sin(r) * length
        p1 = (pos[0] + wx/2, pos[1] + wy/2)
        p2 = (pos[0] - wx/2, pos[1] - wy/2)

        p1 = (p1[0] - 2 * lx/5, p1[1] - 2 * ly/5)
        p2 = (p2[0] - 2 * lx/5, p2[1] - 2 * ly/5)

        p3 = (p2[0] + lx, p2[1] + ly)
        p4 = (p3[0] + wx/5, p3[1] + wy/5)
        p5 = (p4[0] - 4 * lx/5, p4[1] - 4 * ly/5)
        p6 = (p5[0] + 3 * wx/5, p5[1] + 3 * wy/5)
        p7 = (p6[0] + 4 * lx/5, p6[1] + 4 * ly/5)
        p8 = (p7[0] + wx/5, p7[1] + wy/5)

        self.poses.append(pos)
        self.add_polygon_obs([p1,p2,p3,p4,p5,p6,p7,p8, p1])

    def plot(self, ax):
        styles = ['ro', 'bo']
        for i in range(len(self.circle2d)):
            self.circle2d[i].plot(ax)

        for i in range(len(self.polygon2d)):
            if i <= len(self.poses) and i <= len(styles):
                ax.plot(self.poses[i][0], self.poses[i][1], styles[i])

            self.polygon2d[i].plot(ax)

    def isCollision(self, point2d):
        iscollision = False
        for i in range(len(self.circle2d)):
            iscollision = iscollision or self.circle2d[i].isCollision(point2d)

        for i in range(len(self.polygon2d)):
            iscollision = iscollision or self.polygon2d[i].isCollision(point2d)

        return iscollision


class ObsExp:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.create_exp()

    def get_envs(self, queries):
        if self.exp_name == 'goAroundObsR10':
            envs = []
            for i in range(np.shape(queries)[0]):
                env = ObsSet2D()
                query = queries[i,:]
                for j in range(10):
                    env.add_circle_obs(origin=[query[j*3], query[j*3+1]], radius=query[j*3+2])

                envs.append(env)

            return envs
        elif self.exp_name == 'goAroundObsR3V2':
            envs = []
            for i in range(np.shape(queries)[0]):
                env = ObsSet2D()
                query = queries[i,:]
                for j in range(3):
                    env.add_circle_obs(origin=[query[j*2], query[j*2+1]],
                                       radius=2.0)

                envs.append(env)

            return envs
        elif self.exp_name == 'Docking':
            envs = []
            for i in range(np.shape(queries)[0]):
                env = ObsSet2D()
                query = queries[i,:]
                env.add_openRec_obs(query[0:2], query[2], 3, 3.5)
                env.add_openRec_obs(query[3:5], query[5], 3, 3.5)
                envs.append(env)

            return envs

    def create_exp(self):
        if self.exp_name == 'goAroundObs' or self.exp_name == 'goAroundObsTest':
            obs = ObsSet2D()
            obs.add_circle_obs(origin=np.array([0, 0]), radius=0.5)
            # colors = [(0,0.43,0.73), (0.64, 0.08, 0.1), (0, 0.73, 0.43), (0.64, 0.1, 0.5)]
            colors = [(0,0.43,0.73),(0, 0.73, 0.43)]
            nmodel = 2
            start = np.array([0, -1])

            def gen_test(num_test):
                testx = np.linspace(-2, 2, num_test)
                testx = np.expand_dims(testx, axis=1)
                testy = np.ones(shape=(num_test,1))
                testgoals = np.concatenate([testx,testy],axis=1)
                return testgoals

            self.obs = obs
            self.nmodel = nmodel
            self.start = start
            self.colors = colors
            self.gen_test = gen_test
        elif self.exp_name == 'goAroundObsV2':
            obs = ObsSet2D()
            obs.add_circle_obs(origin=np.array([0,0]), radius=0.5)
            obs.add_circle_obs(origin=np.array([-1.5,-1.5]), radius=0.7)
            obs.add_circle_obs(origin=np.array([1.5,-1.5]), radius=0.7)
            colors = ['r', 'b', 'g', 'm']
            nmodel = 4
            start = np.array([0, -3])

            def gen_test(num_test):
                testx = np.linspace(-4, 4, num_test)
                testx = np.expand_dims(testx, axis=1)
                testy = np.ones(shape=(num_test,1))
                testgoals = np.concatenate([testx,testy],axis=1)
                return testgoals

            self.obs = obs
            self.nmodel = nmodel
            self.start = start
            self.colors = colors
            self.gen_test = gen_test
        elif self.exp_name == 'goAroundObsV3':
            obs = ObsSet2D()
            obs.add_circle_obs(origin=np.array([0,2]), radius=0.5)
            obs.add_circle_obs(origin=np.array([-1.2, -1.5]), radius=0.5)
            obs.add_circle_obs(origin=np.array([1.2, -1.5]), radius=0.5)
            obs.add_circle_obs(origin=np.array([-1.7, 0.5]), radius=0.5)
            obs.add_circle_obs(origin=np.array([1.7, 0.5]), radius=0.5)
            colors = ['r', 'b', 'g', 'm', 'c']
            nmodel = 5
            start = np.array([0, 0])

            def gen_test(num_test):
                tr = np.linspace(0, 2 * np.pi, num_test)
                tr = np.expand_dims(tr, axis=1)
                testx = 4 * np.cos(tr)
                testy = 4 * np.sin(tr)
                testgoals = np.concatenate([testx,testy],axis=1)
                return testgoals

            self.obs = obs
            self.nmodel = nmodel
            self.start = start
            self.colors = colors
            self.gen_test = gen_test
        elif self.exp_name == 'goAroundObsV4':
            obs = ObsSet2D()
            obs.add_circle_obs(origin=[1,1], radius=0.5)
            coords = [(-0.5, 4), (4, 4), (4, -0.5), (3, -0.5), (3, 3), (-0.5,3), (-0.5,4)]
            obs.add_polygon_obs(coords=coords)
            colors = ['r', 'b', 'g', 'm']
            nmodel = 4
            start = np.array([0, 0])

            def gen_test(num_test):
                testx = np.linspace(-1, 6, int(num_test/4))
                testy = 5 * np.ones(int(num_test/4))
                testx = np.expand_dims(testx, axis=1)
                testy = np.expand_dims(testy, axis=1)
                testgoals0 = np.concatenate([testx, testy], axis=1)
                testgoals1 = np.concatenate([testy, testx], axis=1)
                testx = np.linspace(-1, 2.5, int(num_test/4))
                testy = 2.5 * np.ones(int(num_test/4))
                testx = np.expand_dims(testx, axis=1)
                testy = np.expand_dims(testy, axis=1)
                testgoals2 = np.concatenate([testx, testy], axis=1)
                testgoals3 = np.concatenate([testy, testx], axis=1)
                testgoals = np.concatenate([testgoals0, testgoals1, testgoals2, testgoals3], axis=0)
                return testgoals

            self.obs = obs
            self.nmodel = nmodel
            self.start = start
            self.colors = colors
            self.gen_test = gen_test
        elif self.exp_name == 'goAroundObsV5':
            obs = ObsSet2D()
            coords = [(-3, -4), (3, -4), (3, 2), (-3, 2), (-3, 1), (2, 1), (2, -1), (-3, -1), (-3, -4)]
            obs.add_polygon_obs(coords=coords)
            coords = [(-3, 3), (3, 3), (3, 4), (-3, 4), (-3, 3)]
            obs.add_polygon_obs(coords=coords)
            colors = ['r', 'b']
            nmodel = 2
            start = np.array([0, 0])

            def gen_test(num_test):
                testx = np.random.uniform(3.5, 5.5, num_test)
                testy = np.random.uniform(-3, 2, num_test)
                testx = np.expand_dims(testx, axis=1)
                testy = np.expand_dims(testy, axis=1)
                testgoals = np.concatenate([testx, testy], axis=1)
                return testgoals

            self.obs = obs
            self.nmodel = nmodel
            self.start = start
            self.colors = colors
            self.gen_test = gen_test
        elif self.exp_name == 'goAroundObsR10':
            def gen_test(num_test):
                queries = []
                envs = []
                for i in range(num_test):
                    centers = np.random.uniform(-8, 8, size=(10,2))
                    radis = np.random.uniform(1,2, size=(10,1))
                    testpos = np.random.uniform(-10,10, size=(1,4))
                    env = ObsSet2D()

                    for j in range(10):
                        env.add_circle_obs(origin=[centers[j,0], centers[j,1]], radius=radis[j])

                    tq = np.concatenate([centers, radis], axis=1)
                    tq = tq.flatten()
                    tq = np.expand_dims(tq, axis=0)
                    tq = np.concatenate([tq, testpos], axis=1)
                    queries.append(tq)
                    envs.append(env)

                queries = np.concatenate(queries, axis=0)
                return queries, envs

            self.gen_test = gen_test
            self.start = np.array([0,0])
            self.qdim = 34
            self.envDim = 2
        elif self.exp_name == 'goAroundObsR3V2':
            def gen_test(num_test):
                queries = []
                envs = []
                for i in range(num_test):
                    centers = np.random.uniform(-5, 5, size=(3,2))
                    goals = np.random.uniform(-9,0, size=(1,2))
                    starts = np.random.uniform(0,9, size=(1,2))
                    env = ObsSet2D()

                    for j in range(3):
                        env.add_circle_obs(origin=[centers[j,0], centers[j,1]],
                                           radius=2.0)

                    tq = centers.flatten()
                    tq = np.expand_dims(tq, axis=0)
                    tq = np.concatenate([tq, goals, starts], axis=1)
                    queries.append(tq)
                    envs.append(env)

                queries = np.concatenate(queries, axis=0)
                return queries, envs

            self.gen_test = gen_test
            self.start = np.array([0,0])
            self.qdim = 10
            self.envDim = 2
        elif self.exp_name == 'Docking':
            def gen_test(num_test):
                queries = []
                envs = []
                for i in range(num_test):
                    r = np.random.uniform(0, 2*np.pi)
                    env = ObsSet2D()
                    env.add_openRec_obs([10,10], r, 5 / 3, 3)
                    queries.append(r)
                    envs.append(env)

                return queries, envs

            self.gen_test = gen_test
            self.start = np.array([1,1])
            self.qdim = 1
            self.envDim = 2

        else:
            raise Exception('The experiment cannot be set up')



