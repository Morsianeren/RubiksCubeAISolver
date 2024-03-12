# %%
import numpy as np
import matplotlib.pyplot as plt

# Descriptor class for quaternion components
# This dataclass simply allows us to access the individual components of the quaternion
class QuaternionComponent:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.q[owner._components.index(self.name)]

    def __set__(self, instance, value):
        idx = instance._components.index(self.name)
        instance.q[idx] = value

class Quaternion():
    # The values of the quaternion is stored in a 4-element array
    #q: np.ndarray = np.array([0., 0., 0., 0.])
    # Each individual component is accessed through a descriptor
    _components = ['w', 'x', 'y', 'z']
    w = QuaternionComponent('w')
    x = QuaternionComponent('x')
    y = QuaternionComponent('y')
    z = QuaternionComponent('z')

    # Magic methods
    def __init__(self, q: np.ndarray):
        self.q = q
        if len(self.q) != 4:
            raise ValueError("Quaternion array must have length 4. Got: ", self.q)

    def __str__(self):
        return f"Quaternion({self.w} + {self.x}i + {self.y}j + {self.z}k)"

    def __truediv__(self, other):
        # Implement division by scalar
        if isinstance(other, (int, float)):
            w = self.w / other
            x = self.x / other
            y = self.y / other
            z = self.z / other
            return Quaternion(np.array([w, x, y, z]))
        # Implement division by another quaternion (element-wise division)
        elif isinstance(other, Quaternion):
            return self * other.inverse
        else:
            raise ValueError("Unsupported operand type(s) for /: ", type(self), type(other))


    def __pow__(self, exponent):
        if exponent == 0:
            return Quaternion([1, 0, 0, 0])
        elif exponent == -1:
            conjugate = self.conjugate
            magnitude = self.square_magnitude
            return conjugate / magnitude
        else:
            return quaternion_power(self, exponent)
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return qq_multiply(self, other)
        elif len(other) == 3: # Vector
            return qv_multiply(self, other)
        else:
            raise ValueError("The object must be a Quaternion or have len 3: ", other, type(other))

    # Properties
    @property
    def v(self):
        return self.q[1:]
    
    @v.setter
    def v(self, value):
        if len(value) != 3:
            raise ValueError("Value must be a 3-element array.")
        self.q[1:] = value
    
    @property
    def conjugate(self):
        return Quaternion(np.array([self.w, -self.x, -self.y, -self.z]))

    @property
    def euler_angles(self):
        return quaternion_to_euler(self)
    
    @property
    def square_magnitude(self):
        return quaternion_square_magnitude(self)

    # Methods
    def normalized(self):
        magnitude = self.square_magnitude
        if magnitude == 0:
            raise ZeroDivisionError("Cannot normalize a zero quaternion.")
        return Quaternion(self.q / np.sqrt(magnitude))

    def get_xyz_vectors(self):
        return get_direction_vectors(self)

    def plot(self, **kwargs):
        # This function plots the quaternion as a vector in 3D space
        x, y, z = self.get_xyz_vectors()
        plt.quiver(0, 0, 0, x[0], x[1], x[2], color='r', **kwargs)
        plt.quiver(0, 0, 0, y[0], y[1], y[2], color='g', **kwargs)
        plt.quiver(0, 0, 0, z[0], z[1], z[2], color='b', **kwargs)

def qv_multiply(quat: Quaternion, vector: np.array) -> Quaternion:
    q1 = quat
    q2 = Quaternion(np.array([0, vector]))
    return qq_multiply(q1, q2)


def qq_multiply(q1:Quaternion, q2:Quaternion) -> Quaternion:
    w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
    w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return Quaternion(np.array([w, x, y, z]))

def euler_to_quaternion(phi, theta, psi) -> Quaternion:
    qw = np.cos(phi/2) * np.cos(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.sin(theta/2) * np.sin(psi/2)
    qx = np.sin(phi/2) * np.cos(theta/2) * np.cos(psi/2) - np.cos(phi/2) * np.sin(theta/2) * np.sin(psi/2)
    qy = np.cos(phi/2) * np.sin(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.cos(theta/2) * np.sin(psi/2)
    qz = np.cos(phi/2) * np.cos(theta/2) * np.sin(psi/2) - np.sin(phi/2) * np.sin(theta/2) * np.cos(psi/2)

    return Quaternion(np.array([qw, qx, qy, qz]))

def quaternion_to_euler(quat: Quaternion) -> list:
    w = quat.w        
    x = quat.x
    y = quat.y
    z = quat.z

    t0 = 2 * (w * x + y * z)
    t1 = 1 - 2 * (x * x + y * y)
    X = np.arctan2(t0, t1)

    t2 = 2 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    Y = np.arcsin(t2)
        
    t3 = 2 * (w * z + x * y)
    t4 = 1 - 2 * (y * y + z * z)
    Z = np.arctan2(t3, t4)

    return [X, Y, Z]

def quaternion_square_magnitude(quat: Quaternion):
    # Extract components of the quaternion
    q0 = quat.w
    q1 = quat.x
    q2 = quat.y
    q3 = quat.z
    
    # Calculate the magnitude
    magnitude = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    
    return magnitude

def quaternion_power(quat: Quaternion, n):
    # Extract components of the quaternion
    q0 = quat.w
    q1 = quat.x
    q2 = quat.y
    q3 = quat.z
    
    # Calculate the magnitude of the vector part
    v_magnitude = np.sqrt(q1**2 + q2**2 + q3**2)
    
    # Calculate the angle between the vector part and the axis of rotation
    theta = np.arccos(q0 / np.sqrt(q0**2 + v_magnitude**2))
    
    # Calculate the magnitude raised to the power of n
    magnitude_powered = (q0**2 + v_magnitude**2)**(n / 2)
    
    # Calculate the vector part of the result
    t = np.sin(n * theta)
    x = (q1 / v_magnitude) * t
    y = (q2 / v_magnitude) * t
    z = (q3 / v_magnitude) * t
    
    # Calculate the scalar part of the result
    w = magnitude_powered * np.cos(n * theta)
    
    return Quaternion([w, x, y, z])

def quaternion_inverse(quat: Quaternion):
        # Calculate the conjugate
        conjugate = Quaternion(np.array([quat.w, -quat.x, -quat.y, -quat.z]))
        
        # Calculate the magnitude squared
        magnitude_squared = quaternion_square_magnitude(quat)
        
        # Check if the quaternion is non-zero to avoid division by zero
        if magnitude_squared == 0:
            raise ZeroDivisionError("Cannot compute inverse for a zero quaternion.")
        
        # Calculate the inverse using the formula: conjugate / magnitude squared
        return conjugate / magnitude_squared

def quaternion_to_rotation_matrix(quat: Quaternion):
    """Convert quaternion to rotation matrix."""
    q_normalized = quat.normalized()
    w = q_normalized.w
    x = q_normalized.x
    y = q_normalized.y
    z = q_normalized.z
    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return rotation_matrix

def get_direction_vectors(quat: Quaternion):
    """Get vectors pointing in x, y, and z directions relative to the quaternion."""
    rotation_matrix = quaternion_to_rotation_matrix(quat)
    x_vector = rotation_matrix[:, 0]  # Vector along x-axis
    y_vector = rotation_matrix[:, 1]  # Vector along y-axis
    z_vector = rotation_matrix[:, 2]  # Vector along z-axis

    x_vector = np.round(x_vector, 5)
    y_vector = np.round(y_vector, 5)
    z_vector = np.round(z_vector, 5)

    return x_vector, y_vector, z_vector

# Test code
#v_x = Quaternion(np.array([1, 0, 0, 0]))
#v_y = Quaternion(np.array([0, 0, 1, 0]))
#v_z = Quaternion(np.array([0, 0, 0, 1]))
#
#d = 90 # Degrees to rotate around the z-axis
#
#w1 = np.cos(d * np.pi/360)
#w2 = np.sin(d * np.pi/360)
#
#q = Quaternion(np.array([w1, 0., 0., 1.*w2]))
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#v_x.plot()
#
##for c, v in {'r':v_x, 'g':v_y, 'b':v_z}.items():
##    # Passive rotation
##    q_inv = q ** -1
##    v_new = q * v * q_inv
##    v.plot(color=c, linestyle='-')
##    v_new.plot(color=c, linestyle='-.')
#
#ax.set_xlim([-1.5, 1.5])
#ax.set_ylim([-1.5, 1.5])
#ax.set_zlim([-1.5, 1.5])
#ax.legend(['v', 'v new'])