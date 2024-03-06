# %%
import numpy as np
import matplotlib.pyplot as plt

class Quaternion():
    def __init__(self, w=0., x=0., y=0., z=0., v=None) -> None:
        self._w = w
        self._v = np.array([x, y, z])
        if v is not None:
            self._v = v

    def __str__(self):
        return f"Quaternion({self.w} + {self.x}i + {self.y}j + {self.z}k)"

    # Getters
    @property
    def w(self):
        return self._w
    @property
    def x(self):
        return self._v[0]
    @property
    def y(self):
        return self._v[1]
    @property
    def z(self):
        return self._v[2]
    @property
    def v(self):
        return self._v
    
    # Setters
    @w.setter
    def w(self, value):
        self._w = value
    @x.setter
    def x(self, value):
        self._v[0] = value
    @y.setter
    def y(self, value):
        self._v[1] = value
    @z.setter
    def z(self, value):
        self._v[2] = value
    @v.setter
    def v(self, value):
        self._v = value
    
    @property
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return qq_multiply(self, other)
        elif len(other) == 3:
            return qv_multiply(self, other)
        else:
            raise ValueError("The object must be a Quaternion or have len 3: ", other, type(other))

    def __pow__(self, exponent):
        if exponent == 0:
            return Quaternion(1, 0, 0, 0)
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
    
    def plot(self, **kwargs):
        # First get euler angles
        #euler = self.euler_angles
        #print("Euler angles: ", euler)
        x = self.x
        y = self.y
        z = self.z
        plt.quiver(0, 0, 0, x, y, z, **kwargs)
        
# Link to functions:
# https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/

def qv_multiply(q:Quaternion, v:np.array) -> Quaternion:
    q1 = q
    q2 = Quaternion(w=0, v=v)
    return qq_multiply(q1, q2)


def qq_multiply(q1:Quaternion, q2:Quaternion) -> Quaternion:
    w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
    w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return Quaternion(w, x, y, z)

def euler_to_quaternion(phi, theta, psi) -> Quaternion:
    qw = np.cos(phi/2) * np.cos(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.sin(theta/2) * np.sin(psi/2)
    qx = np.sin(phi/2) * np.cos(theta/2) * np.cos(psi/2) - np.cos(phi/2) * np.sin(theta/2) * np.sin(psi/2)
    qy = np.cos(phi/2) * np.sin(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.cos(theta/2) * np.sin(psi/2)
    qz = np.cos(phi/2) * np.cos(theta/2) * np.sin(psi/2) - np.sin(phi/2) * np.sin(theta/2) * np.cos(psi/2)

    return Quaternion(qw, qx, qy, qz)

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
    
    return Quaternion(w, x, y, z)

# Test code
v_x = Quaternion(0, 1, 0, 0)
v_y = Quaternion(0, 0, 1, 0)
v_z = Quaternion(0, 0, 0, 1)

q = Quaternion(0.5, 0.5, 0., .5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c, v in {'r':v_x, 'g':v_y, 'b':v_z}.items():
    # Passive rotation
    q_inv = q ** -1
    v_new = q * v * q_inv
    print(v)
    v.plot(color=c, linestyle='-')
    v_new.plot(color=c, linestyle='-.')

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.legend(['v', 'v new'])