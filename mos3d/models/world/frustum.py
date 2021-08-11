import math
import numpy as np
import random
from scipy.spatial.transform import Rotation as scipyR

def R_quat(x, y, z, w, affine=False):
    R = scipyR.from_quat([x,y,z,w])
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_dcm()
        aR[3,3] = 1
        R = aR
    return R

def R_euler(thx, thy, thz, affine=False):
    """
    Obtain the rotation matrix of Rz(thx) * Ry(thy) * Rx(thz); euler angles
    """
    R = scipyR.from_euler("xyz", [thx, thy, thz], degrees=True)
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_dcm()
        aR[3,3] = 1
        R = aR
    return R

def T(dx, dy, dz):
    return np.array([
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1
    ]).reshape(4,4)

# Math
def vec(p1, p2):
    """ vector from p1 to p2 """
    if type(p1) != np.ndarray:
        p1 = np.array(p1)
    if type(p2) != np.ndarray:
        p2 = np.array(p2)
    return p2 - p1

class FrustumCamera:

    def __init__(self, fov=90, aspect_ratio=1, near=1, far=5):
        """
        fov: angle (degree), how wide the viewing angle is.
        near: near-plane's distance to the camera
        far: far-plane's distance to the camera
        """
        # Initially, the camera is always at (0,0,0), looking at direction (0,0,-1)
        # This can be changed by calling `transform_camera()`
        #
        # 6 planes:
        #     3
        #  0 2 4 5
        #     1

        # sizes of near and far planes
        fov = fov*math.pi / 180
        h1 = near * math.tan(fov/2) * 2
        w1 = abs(h1 * aspect_ratio)
        h2 = far * math.tan(fov/2) * 2
        w2 = abs(h2 * aspect_ratio)
        self._dim = (w1, h1, w2, h2)
        self._params = (fov, aspect_ratio, near, far)

        ref1 = np.array([w1/2, h1/2, -near, 1])
        ref2 = np.array([-w2/2, -h2/2, -far, 1])

        p1A = np.array([w1/2, h1/2, -near])
        p1B = np.array([-w1/2, h1/2, -near])
        p1C = np.array([w1/2, -h1/2, -near])        
        n1 = np.cross(vec(p1A, p1B),
                      vec(p1A, p1C))

        p2A = p1A
        p2B = p1C
        p2C = np.array([w2/2, h2/2, -far])
        n2 = np.cross(vec(p2A, p2B),vec(p2A, p2C))        

        p3A = p1A
        p3B = p2C
        p3C = p1B
        n3 = np.cross(vec(p3A, p3B),
                      vec(p3A, p3C))

        p4A = np.array([-w2/2, -h2/2, -far])
        p4B = np.array([-w1/2, -h1/2, -near])
        p4C = np.array([-w2/2, h2/2, -far])
        n4 = np.cross(vec(p4A, p4B),
                      vec(p4A, p4C))

        p5A = p4B
        p5B = p4A
        p5C = p2B
        n5 = np.cross(vec(p5A, p5B),
                      vec(p5A, p5C))

        p6A = p4A
        p6B = p4C
        p6C = p2C
        n6 = np.cross(vec(p6A, p6B),
                      vec(p6A, p6C))

        p = np.array([n1,n2,n3,n4,n5,n6])
        for i in range(6):  # normalize
            p[i] = p[i] / np.linalg.norm(p[i])
        p = np.array([p[i].tolist() + [0] for i in range(6)])        
        r = np.array([ref1, ref1, ref1, ref2, ref2, ref2])
        assert self.within_range((p, r), [0,0,-far-(-far+near)/2, 1])
        self._p = p
        self._r = r

        # compute the volume inside the frustum
        volume = []
        count = 0
        for z in range(-int(round(far)), -int(round(near))):
            for y in range(-int(round(h2/2))-1, int(round(h2/2))+1):
                for x in range(-int(round(w2/2))-1, int(round(w2/2))+1):
                    if self.within_range((self._p, self._r), (x,y,z,1)):
                        volume.append([x,y,z,1])
        self._volume = np.array(volume, dtype=int)

    @property
    def near(self):
        return self._params[-2]
    
    @property
    def far(self):
        return self._params[-1]

    @property
    def fov(self):
        """returns fov in degrees"""
        return self._params[0] / math.pi * 180
    
    @property
    def aspect_ratio(self):
        return self._params[1]

    @property
    def volume(self):
        return self._volume

    def print_info(self):
        print("         FOV: " + str(self.fov))
        print("aspect_ratio: " + str(self.aspect_ratio))
        print("        near: " + str(self.near))
        print("         far: " + str(self.far))
        print(" volume size: " + str(len(self.volume)))

    def within_range(self, config, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion"""
        p, r = config
        for i in range(6): 
            if np.dot(vec(r[i], point), p[i]) >= 0:
                return False
        return True

    def transform_camera(self, pose, permanent=False):
        """Given a pose, transform the camera to that pose,
        and compute the new configuration (p, r).
        Returns the configuration after the transform is applied.
        If `permanent` is true, then the camera's own configuration is updated.
        In other words, this is saying `set up the camera at the given pose`
        """
        if len(pose) == 7:
            x, y, z, qx, qy, qz, qw = pose
            R = R_quat(qx, qy, qz, qw, affine=True)
        elif len(pose) == 6:
            x, y, z, thx, thy, thz = pose            
            R = R_euler(thx, thy, thz, affine=True)
        r_moved = np.transpose(np.matmul(T(x, y, z),
                                         np.matmul(R, np.transpose(self._r))))
        p_moved =  np.transpose(np.matmul(R, np.transpose(self._p)))
        if permanent:
            self._p = p_moved
            self._r = r_moved
            self._volume = np.transpose(np.matmul(T(x, y, z),
                                                  np.matmul(R, np.transpose(self._volume))))            
        return p_moved, r_moved
    

    def field_of_view_size(self):
        return len(self._volume)


def unittest():
    camera = FrustumCamera(fov=90, aspect_ratio=1, near=0.1, far=5)

    # The camera should now be placed at (0,2,0), looking at the negative z direction
    p, r = camera.transform_camera((0, 2, 0, 0, 0, 0))

    # Pass in (x,y,z,1) affine point
    print(camera.within_range((p,r), (0, 0, 0, 1)))
    print(camera.within_range((p,r), (0, 2, -1, 1)))
    print(camera.within_range((p,r), (0, 2, -2, 1)))
    print(camera.within_range((p,r), (0, 2, -4.9, 1)))        
    print(camera.within_range((p,r), (0, 2, -5, 1)))    
    
if __name__ == "__main__":
    unittest()
