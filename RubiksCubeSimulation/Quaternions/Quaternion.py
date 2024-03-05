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

    @property
    def euler_angles(self):
        return quaternion_to_euler(self)
    
    def plot(self, ax = plt.Axes, color='r'):
        # First get euler angles
        euler = self.euler_angles
        print("Euler angles: ", euler)
        ax.quiver(0, 0, 0, euler[0], euler[1], euler[2], color=color)
        
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

def plot_quaternion(q: Quaternion):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Cartesian axes
    ax.quiver(-1, 0, 0, 3, 0, 0, color='#aaaaaa',linestyle='dashed')
    ax.quiver(0, -1, 0, 0,3, 0, color='#aaaaaa',linestyle='dashed')
    ax.quiver(0, 0, -1, 0, 0, 3, color='#aaaaaa',linestyle='dashed')
    # Vector after rotation
    ax.quiver(0, 0, 0, 0, 0.71, -0.71, color='r')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    plt.show()


# Test code
v = Quaternion(0, 1, 1, 1)

q = Quaternion(0.5, 0, 0.5, 0)

v_new = q * v * q.conjugate

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
v.plot(ax, 'b')
v_new.plot(ax, 'r')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
