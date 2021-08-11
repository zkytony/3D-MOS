from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import *
import numpy as np
import random
import math
from scipy.spatial.transform import Rotation as scipyR
import scipy.stats as stats

from pomdp_py import *

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def normalize_log_prob(likelihoods):
    """Given an np.ndarray of log values, take the values out of the log space,
    and normalize them so that they sum up to 1"""
    normalized = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                        (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
    return normalized

def uniform(size, ranges):
    return tuple(random.randrange(ranges[i][0], ranges[i][1])
                 for i in range(size))

def diff(rang):
    return rang[1] - rang[0]

def in_range(x, rang):
    return x >= rang[0] and x < rang[1]

def in_region(p, ranges):
    return in_range(p[0], ranges[0]) and in_range(p[1], ranges[1]) and in_range(p[2], ranges[2])

def remap(oldval, oldmin, oldmax, newmin, newmax):
    return (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

# Printing
def json_safe(obj):
    if isinstance(obj, bool):
        return str(obj).lower()
    elif isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {json_safe(key):json_safe(value) for key, value in obj.items()}
    else:
        return str(obj)
    return obj

# OpenGL utilities

def apply_perspective_transform(fovy, aspect, znear, zfar):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity();
    gluPerspective(fovy, aspect, znear, zfar)

def apply_orthographic_transform(left, right, bottom, top, znear, zfar):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(left, right, bottom, top, znear, zfar)


def set_viewport(x, y, width, height):
    glViewport(x, y, width, height)

def generate_vbo_arrays(data_list):
    vbos = glGenBuffers(len(data_list))
    for i, vbo in enumerate(vbos):
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     sizeof(GLfloat)*len(data_list[i]),
                     (ctypes.c_float*len(data_list[i]))(*data_list[i]),  # !!!
                     GL_STATIC_DRAW)
    return vbos

def generate_vbo_elements(vertices, indices, colors=None):
    if colors is None:
        vertex_vbo, index_vbo = glGenBuffers(2)
    else:
        vertex_vbo, index_vbo, color_vbo = glGenBuffers(3)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(GLfloat)*len(vertices),
                 (ctypes.c_float*len(vertices))(*vertices),  # !!!
                 GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_vbo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(GLint)*len(indices),
                 (ctypes.c_uint*len(indices))(*indices),  # !!!
                 GL_STATIC_DRAW)

    if colors is None:
        return vertex_vbo, index_vbo
    else:
        glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     sizeof(GLfloat)*len(colors), # byte size
                     (ctypes.c_float*len(colors))(*colors),
                     GL_STATIC_DRAW)
        return vertex_vbo, index_vbo, color_vbo

def cube(scale=0.5, color=(1,0,0), boundary_color=None, color2=None):
    s = scale
    vertices=np.array([
        -s, -s, -s,
         s, -s, -s,
         s,  s, -s,
        -s,  s, -s,
        -s, -s,  s,
         s, -s,  s,
         s,  s,  s,
        -s,  s,  s,
    ])
    if color2 is None:
        colors=np.array(color*8)
    else:
        colors=np.array(color*4 + color2*4)
    indices=np.array([
        0, 1, 2, 3,
        0, 4, 5, 1,
        1, 5, 6, 2,
        2, 6, 7, 3,
        3, 7, 4, 0,
        4, 7, 6, 5
    ])
    if boundary_color is not None:
        boundary_colors = np.array(boundary_color*8)
        return vertices, indices, colors, boundary_colors
    else:
        return vertices, indices, colors


def draw_quads(num_indices, vertex_vbo, index_vbo, color_vbo=None, bcolor_vbo=None,
               color_size=3, vertex_size=3):

    if color_vbo is not None or bcolor_vbo is not None:
        glEnableClientState(GL_COLOR_ARRAY);

    if color_vbo is not None:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    elif bcolor_vbo is not None:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo);
    glVertexPointer(vertex_size, GL_FLOAT, 0, None);
    if color_vbo is not None:
        glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
        glColorPointer(color_size, GL_FLOAT, 0, None);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_vbo);
    glDrawElements(GL_QUADS, num_indices, GL_UNSIGNED_INT, None)

    # Draw boundary
    if bcolor_vbo is not None:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glBindBuffer(GL_ARRAY_BUFFER, bcolor_vbo);
        glColorPointer(color_size, GL_FLOAT, 0, None);
        glDrawElements(GL_QUADS, num_indices, GL_UNSIGNED_INT, None)

    if color_vbo is not None or bcolor_vbo is not None:
        glDisableClientState(GL_COLOR_ARRAY)


# Math
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
        aR[:3,:3] = R.as_matrix()
        aR[3,3] = 1
        R = aR
    return R

def R_quat(x, y, z, w, affine=False):
    R = scipyR.from_quat([x,y,z,w])
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_matrix()
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


# Others
def safe_slice(arr, start, end):
    true_start = max(0, min(len(arr)-1, start))
    true_end = max(0, min(len(arr)-1, end))
    return arr[true_start:true_end]

# Colors on terminal https://stackoverflow.com/a/287944/2893053
class bcolors:
    WHITE = '\033[97m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

    @staticmethod
    def disable():
        bcolors.WHITE   = ''
        bcolors.CYAN    = ''
        bcolors.MAGENTA = ''
        bcolors.BLUE    = ''
        bcolors.GREEN   = ''
        bcolors.YELLOW  = ''
        bcolors.RED     = ''
        bcolors.ENDC    = ''

    @staticmethod
    def s(color, content):
        """Returns a string with color when shown on terminal.
        `color` is a constant in `bcolors` class."""
        return color + content + bcolors.ENDC

def print_info(content):
    print(bcolors.s(bcolors.BLUE, content))

def print_note(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_error(content):
    print(bcolors.s(bcolors.RED, content))

def print_warning(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_success(content):
    print(bcolors.s(bcolors.GREEN, content))

def print_info_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.BLUE, content))

def print_note_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))

def print_error_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.RED, content))

def print_warning_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.YELLOW, content))

def print_success_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))
# For your convenience:
# from moos3d.util import print_info, print_error, print_warning, print_success, print_info_bold, print_error_bold, print_warning_bold, print_success_bold


# confidence interval
def ci_normal(series, confidence_interval=0.95):
    series = np.asarray(series)
    tscore = stats.t.ppf((1 + confidence_interval)/2.0, df=len(series)-1)
    y_error = stats.sem(series)
    ci = y_error * tscore
    return ci
