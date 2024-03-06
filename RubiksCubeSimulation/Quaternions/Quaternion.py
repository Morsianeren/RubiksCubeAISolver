# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

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

@dataclass
class Quaternion():
    # The values of the quaternion is stored in a 4-element array
    q: np.ndarray = np.array([0., 0., 0., 0.])
    # Each individual component is accessed through a descriptor
    w = QuaternionComponent('w')
    x = QuaternionComponent('x')
    y = QuaternionComponent('y')
    z = QuaternionComponent('z')

    def __post_init__(self):
        if len(self.q) != 4:
            raise ValueError("Quaternion array must have length 4. Got: ", self.q)

    def __str__(self):
        return f"Quaternion({self.w} + {self.x}i + {self.y}j + {self.z}k)"

    # Getters
    @property
    def v(self):
        return self.q[1:]
    
    # Setters
    @v.setter
    def v(self, value):
        if len(value) != 3:
            raise ValueError("Value must be a 3-element array.")
        self.q[1:] = value
    
    @property
    def conjugate(self):
        return Quaternion(np.array([self.w, -self.x, -self.y, -self.z]))
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return qq_multiply(self, other)
        elif len(other) == 3: # Vector
            return qv_multiply(self, other)
        else:
            raise ValueError("The object must be a Quaternion or have len 3: ", other, type(other))

    def __pow__(self, exponent):
        if exponent == 0:
            return Quaternion([1, 0, 0, 0])
        #elif exponent == -1:
        #    return self.conjugate / self.magnitude
        else:
            return quaternion_power(self, exponent)


    @property
    def euler_angles(self):
        return quaternion_to_euler(self)
    
    @property
    def magnitude(self):
        return quaternion_magnitude(self)
    

def plot_quaternion(q: Quaternion, **kwargs):
    # First get euler angles
    #euler = self.euler_angles
    #print("Euler angles: ", euler)
    x = q.x
    y = q.y
    z = q.z
    plt.quiver(0, 0, 0, x, y, z, **kwargs)
        
# Link to functions:
# https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/

def qv_multiply(q:Quaternion, v:np.array) -> Quaternion:
    q1 = q
    q2 = Quaternion(np.array([0, v]))
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

def quaternion_to_euler(q: Quaternion) -> list:
    w = q.w        
    x = q.x
    y = q.y
    z = q.z

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

def quaternion_magnitude(q: Quaternion):
    # Extract components of the quaternion
    q0 = q.w
    q1 = q.x
    q2 = q.y
    q3 = q.z
    
    # Calculate the magnitude
    magnitude = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    
    return magnitude

def quaternion_power(q:Quaternion, n):
    # Extract components of the quaternion
    q0 = q.w
    q1 = q.x
    q2 = q.y
    q3 = q.z
    
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

# Test code
v_x = Quaternion(np.array([0, 1, 0, 0]))
v_y = Quaternion(np.array([0, 0, 1, 0]))
v_z = Quaternion(np.array([0, 0, 0, 1]))

q = Quaternion(np.array([0.5, 0.5, 0., .5]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c, v in {'r':v_x, 'g':v_y, 'b':v_z}.items():
    # Passive rotation
    q_inv = q ** -1
    v_new = q * v * q_inv
    print(v)
    plot_quaternion(v, color=c, linestyle='-')
    plot_quaternion(v_new, color=c, linestyle='-.')

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.legend(['v', 'v new'])