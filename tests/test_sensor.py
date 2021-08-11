import numpy as np
import random
import math
from mos3d.models.world.sensor_model import FrustumCamera
from mos3d import *
import mos3d.util as util


def test_point_in_plane():
    for i in range(1000):
        ref1 = np.array([random.randint(0, 100),
                        random.randint(0, 100),
                        random.randint(0, 100)])

        ref2 = np.array([random.randint(0, 100),
                         random.randint(0, 100),
                         random.randint(0, 100)])

        ref3 = np.array([random.randint(0, 100),
                         random.randint(0, 100),
                         random.randint(0, 100)])
        p = np.cross(util.vec(ref1, ref2),
                     util.vec(ref1, ref3))
        refs = [ref1, ref2, ref3]

        # point outside plane
        point = refs[random.randint(0,2)] + p*random.randint(1,100)
        assert np.dot(util.vec(ref1, point), p) >= 0
        # point outside plane
        point = refs[random.randint(0,2)] - p*random.randint(1,100)
        assert np.dot(util.vec(ref2, point), p) <= 0
    print("Passed.")

def test_within_cube():

    def visible(point, p, r):
        for i in range(6):
            if np.dot(util.vec(r[i], point), p[i]) >= 0:
                return False
        return True

    for i in range(100):
        size = random.randint(1, 100)

        ref1 = np.array([size, size, size])
        ref2 = np.array([0, 0, 0])

        p1A = np.array([size, size, size])
        p1B = np.array([size, size, 0])
        p1C = np.array([size, 0, size])
        n1 = np.cross(util.vec(p1A, p1C),
                      util.vec(p1A, p1B))

        p2A = p1A
        p2B = p1B
        p2C = np.array([0, size, size])
        n2 = np.cross(util.vec(p2A, p2B),
                      util.vec(p2A, p2C))

        p3A = p1A
        p3B = p2C
        p3C = p1C
        n3 = np.cross(util.vec(p3A, p3B),
                      util.vec(p3A, p3C))

        p4A = p1C
        p4B = np.array([0,0,size])
        p4C = np.array([size, 0, 0])
        n4 = np.cross(util.vec(p4A, p4B),
                      util.vec(p4A, p4C))

        p5A = p4C
        p5B = np.array([0, 0, 0])
        p5C = p1B
        n5 = np.cross(util.vec(p5A, p5B),
                      util.vec(p5A, p5C))

        p6A = p5B
        p6B = p4B
        p6C = p2C
        n6 = np.cross(util.vec(p6A, p6B),
                      util.vec(p6A, p6C))

        n = np.array([n1,n2,n3,n4,n5,n6])
        for i in range(6):  # normalize
            n[i] = n[i] / np.linalg.norm(n[i])
        r = np.array([ref1, ref1, ref1, ref2, ref2, ref2])

        for j in range(100):
            # point inside cube
            x1 = random.uniform(0, size)
            y1 = random.uniform(0, size)
            z1 = random.uniform(0, size)
            assert visible([x1, y1, z1], n, r)

            # point outside cube
            x2 = random.uniform(size, size*10)
            y2 = random.uniform(size, size*10)
            z2 = random.uniform(size, size*10)
            assert not visible([x2, y2, z2], n, r)


def test_within_frustum():

    def visible(point, p, r):
        for i in range(6):
            if np.dot(util.vec(r[i], point), p[i]) >= 0:
                return False
        return True

    for i in range(100):
        size1 = random.randint(1, 100)
        size2 = random.randint(size1, size1*2)
        dist = random.randint(1, 100)

        ref1 = np.array([size1/2, size1/2, dist])
        ref2 = np.array([-size2/2, -size2/2, 0])

        p1A = np.array([size1/2, size1/2, dist])
        p1B = np.array([-size1/2, size1/2, dist])
        p1C = np.array([size1/2, -size1/2, dist])
        n1 = np.cross(util.vec(p1A, p1B),
                      util.vec(p1A, p1C))

        p2A = p1A
        p2B = p1C
        p2C = np.array([size2/2, size2/2, 0])
        n2 = np.cross(util.vec(p2A, p2B),
                      util.vec(p2A, p2C))

        p3A = p1A
        p3B = p2C
        p3C = p1B
        n3 = np.cross(util.vec(p3A, p3B),
                      util.vec(p3A, p3C))

        p4A = np.array([-size2/2, -size2/2, 0])
        p4B = np.array([-size1/2, -size1/2, dist])
        p4C = np.array([-size2/2, size2/2, 0])
        n4 = np.cross(util.vec(p4A, p4B),
                      util.vec(p4A, p4C))

        p5A = p4B
        p5B = p4A
        p5C = p2B
        n5 = np.cross(util.vec(p5A, p5B),
                      util.vec(p5A, p5C))

        p6A = p4A
        p6B = p4C
        p6C = p2C
        n6 = np.cross(util.vec(p6A, p6B),
                      util.vec(p6A, p6C))

        n = np.array([n1,n2,n3,n4,n5,n6])
        for i in range(6):  # normalize
            n[i] = n[i] / np.linalg.norm(n[i])
        r = np.array([ref1, ref1, ref1, ref2, ref2, ref2])

        for j in range(100):
            # point outside cube
            x2 = random.uniform(size2/2, size2/2*10)
            y2 = random.uniform(size2/2, size2/2*10)
            z2 = random.uniform(dist, dist*10)
            assert not visible([x2, y2, z2], n, r)

            # point inside cube
            x1 = random.uniform(-size1/2, size1/2)
            y1 = random.uniform(-size1/2, size1/2)
            z1 = random.uniform(0, dist)
            assert visible([x1, y1, z1], n, r)


def test_frustum_camera(preset=None):

    def visible(point, p, r):
        for i in range(6):
            if np.dot(util.vec(r[i], point), p[i]) >= 0:
                print("Point: %s" % str(point))
                print("Point outside plane %i" % i)
                print("    Plane normal: %s" % str(p[i]))
                print("    Plane refs: %s" % str(r[i]))
                print("       Measure: %.3f" % np.dot(util.vec(r[i], point), p[i]))
                import pdb; pdb.set_trace()
                return False
        return True

    if preset is None:
        # w1 = 1.445
        # w2 = 1.454
        w1 = random.uniform(1, 10)
        h1 = w1
        w2 = random.uniform(w1, w1*2)
        h2 = w2
        near = random.uniform(1, 2)
        far = (h2/2)*near/(h1/2)
        print("  w1: %.3f" % w1)
        print("  w2: %.3f" % w2)
        print("near: %.3f" % near)
        print(" far: %.3f" % far)
        print(" FOV: %.3f" % (2*math.atan2(h1/2, near) / math.pi * 180))
        print(" FOV2: %.3f" % (2*math.atan2(h2/2, far) / math.pi * 180))

    else:
        fov = preset['fov']*math.pi / 180
        print(fov)
        near = preset['near']
        far = preset['far']
        h1 = near * math.tan(fov/2) * 2
        w1 = abs(h1 * preset['aspect_ratio'])
        h2 = far * math.tan(fov/2) * 2
        w2 = abs(h2 * preset['aspect_ratio'])
        print("%.2f, %.2f" % (w1, h1))
        print("%.2f, %.2f" % (w2, h2))

    ref1 = np.array([w1/2, h1/2, -near, 1])
    ref2 = np.array([-w2/2, -h2/2, -far, 1])

    p1A = np.array([w1/2, h1/2, -near])
    p1B = np.array([-w1/2, h1/2, -near])
    p1C = np.array([w1/2, -h1/2, -near])
    n1 = np.cross(util.vec(p1A, p1B),
                  util.vec(p1A, p1C))

    p2A = p1A
    p2B = p1C
    p2C = np.array([w2/2, h2/2, -far])
    n2 = np.cross(util.vec(p2A, p2B),util.vec(p2A, p2C))

    p3A = p1A
    p3B = p2C
    p3C = p1B
    n3 = np.cross(util.vec(p3A, p3B),
                  util.vec(p3A, p3C))

    p4A = np.array([-w2/2, -h2/2, -far])
    p4B = np.array([-w1/2, -h1/2, -near])
    p4C = np.array([-w2/2, h2/2, -far])
    n4 = np.cross(util.vec(p4A, p4B),
                  util.vec(p4A, p4C))

    p5A = p4B
    p5B = p4A
    p5C = p2B
    n5 = np.cross(util.vec(p5A, p5B),
                  util.vec(p5A, p5C))

    p6A = p4A
    p6B = p4C
    p6C = p2C
    n6 = np.cross(util.vec(p6A, p6B),
                  util.vec(p6A, p6C))

    p = np.array([n1,n2,n3,n4,n5,n6])
    for i in range(6):  # normalize
        p[i] = p[i] / np.linalg.norm(p[i])
    p = np.array([p[i].tolist() + [0] for i in range(6)])
    r = np.array([ref1, ref1, ref1, ref2, ref2, ref2])
    assert visible([0,0,-far-(-far+near)/2, 1], p, r)

    # Introduce random rotation & translation to the camera
    point_in = np.array([0,0,-far-(-far+near)/2 ,1])
    for i in range(10):
        x, y, z = random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)
        thx, thy, thz = util.to_radians(random.randint(-180, 180)),\
            util.to_radians(random.randint(-180, 180)),\
            util.to_radians(random.randint(-180, 180))
        r = np.transpose(np.matmul(util.R_z(thz),
                                   np.matmul(util.R_y(thy),
                                             np.matmul(util.R_x(thx),
                                                       np.matmul(util.T(x, y, z), np.transpose(r))))))
        p = np.transpose(np.matmul(util.R_z(thz),
                                   np.matmul(util.R_y(thy),
                                             np.matmul(util.R_x(thx), np.transpose(p)))))

        point_in = np.transpose(np.matmul(util.R_z(thz),
                                          np.matmul(util.R_y(thy),
                                                    np.matmul(util.R_x(thx),
                                                              np.matmul(util.T(x, y, z), np.transpose(point_in))))))
        assert visible(point_in, p, r)


world_basic =\
"1024\n1024\n1024\n\ncube 0 0 0\n---\nrobot 512 512 512 0 0 0 occlusion"


def test_fov_ratio():
    for i in range(4, 30):
        camera_str = " 60 1.0 0.1 " + str(i)
        worldstr = world_basic + camera_str
        gridworld, init_state = parse_worldstr(worldstr)
        volume = gridworld.get_frustum_poses(init_state.robot_pose)
        print("When depth=%d, field of view volume contains %d voxels"
              % (i, len(volume)))


    for m in [4, 8, 16, 32, 64, 128]:
        print("\nWorld %dx%dx%d:" % (m,m,m))
        for i in range(4, 30):
            camera_str = " 45 1.0 0.1 " + str(i)
            worldstr = world_basic + camera_str
            gridworld, init_state = parse_worldstr(worldstr)
            volume = gridworld.get_frustum_poses(init_state.robot_pose)
            ratio = len(volume) / (m**3)
            print("    In a world of dimensions %dx%dx%d, d=%d takes up %.3f" % (m,m,m,i,ratio))
            if abs(ratio - 0.02) < 1e-2:
                print("  **** recommended setting (2\%%) for %dx%dx%d: %d ****" % (m, m, m, i))
                p2r = True
            if abs(ratio - 0.05) < 1e-2:
                print("  **** recommended setting (5\%%) for %dx%dx%d: %d ****" % (m, m, m, i))
                p5r = True



def main():
    test_point_in_plane()
    test_within_cube()
    test_within_frustum()

    preset = {
        'fov': 90,
        'aspect_ratio': 1.0,
        'near': 1,
        'far': 10.0
    }
    test_frustum_camera(preset=preset)
    test_fov_ratio()
    print("Done.")

if __name__ == '__main__':
    main()
