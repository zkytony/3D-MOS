import numpy as np
import random
import math
from scipy.spatial.transform import Rotation as scipyR

# Math
def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def vec(p1, p2):
    """ vector from p1 to p2 """
    if type(p1) != np.ndarray:
        p1 = np.array(p1)
    if type(p2) != np.ndarray:
        p2 = np.array(p2)
    return p2 - p1

def proj(vec1, vec2, scalar=False):
    # Project vec1 onto vec2. Returns a vector in the direction of vec2.
    scale = np.dot(vec1, vec2) / np.linalg.norm(vec2)
    if scalar:
        return scale
    else:
        return vec2 * scale

def R_x(th):
    return np.array([
        1, 0, 0, 0,
        0, np.cos(th), -np.sin(th), 0,
        0, np.sin(th), np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_y(th):
    return np.array([
        np.cos(th), 0, np.sin(th), 0,
        0, 1, 0, 0,
        -np.sin(th), 0, np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_z(th):
    return np.array([
        np.cos(th), -np.sin(th), 0, 0,
        np.sin(th), np.cos(th), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def T(dx, dy, dz):
    return np.array([
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1
    ]).reshape(4,4)

def to_radians(th):
    return th*np.pi / 180

def to_degrees(th):
    return th*180 / np.pi

def R_between(v1, v2):
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Only applicable to 3D vectors!")
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    I = np.identity(3)

    vX = np.array([
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0
    ]).reshape(3,3)
    R = I + vX + np.matmul(vX,vX) * ((1-c)/(s**2))
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

def R_quat(x, y, z, w, affine=False):
    R = scipyR.from_quat([x,y,z,w])
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_dcm()
        aR[3,3] = 1
        R = aR
    return R    
    
def R_to_euler(R):
    """
    Obtain the thx,thy,thz angles that result in the rotation matrix Rz(thx) * Ry(thy) * Rx(thz)
    Reference: http://planning.cs.uiuc.edu/node103.html
    """
    return R.as_euler('xyz', degrees=True)
    # # To prevent numerical errors, avoid super small values.
    # epsilon = 1e-9
    # matrix[abs(matrix - 0.0) < epsilon] = 0.0
    # thz = to_degrees(math.atan2(matrix[1,0], matrix[0,0]))    
    # thy = to_degrees(math.atan2(-matrix[2,0], math.sqrt(matrix[2,1]**2 + matrix[2,2]**2)))
    # thx = to_degrees(math.atan2(matrix[2,1], matrix[2,2]))            
    # return thx, thy, thz

def R_to_quat(R):
    return R.as_quat()

def euler_to_quat(thx, thy, thz):
    return scipyR.from_euler("xyz", [thx, thy, thz], degrees=True).as_quat()

def quat_to_euler(x, y, z, w):
    return scipyR.from_quat([x,y,z,w]).as_euler("xyz", degrees=True)

def approx_equal(v1, v2, epsilon=1e-6):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if abs(v1[i] - v2[i]) > epsilon:
            return False
    return True
