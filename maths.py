import numpy as np
import math
import logging


class Point(object):
    def __init__(
        self,
        x: float,
        y: float
    ):
        self.x = x
        self.y = y

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __mul__(self, other):
        self.x *= other.x
        self.y *= other.y
        return self


class Vector(object):
    def __init__(
        self,
        a: Point,
        b: Point
    ):
        self.a = a
        self.b = b

    def __add__(self, other):
        if self.a != other.a:
            raise ValueError("Adding vectors with different 'a'")
        self.b += other.b
        return self

    def __sub__(self, other):
        if self.a != other.a:
            raise ValueError("Subtracting vectors with different 'a'")
        self.b -= other.b
        return self

    def dot(self, other):
        if self.a != other.a:
            raise ValueError("Dot-multiplying vectors with different 'a'")
        self.b *= other.b
        return self

    def angle(self, other):
        if self.a != other.a:
            raise ValueError("Angle measurement vectors with different 'a'")
        vec = self.b - self.a
        vec_sq = vec * vec
        return pow(vec_sq.x + vec_sq.y, 0.5)

    @property
    def length(self):
        vec = self.b - self.a
        vec_sq = vec * vec
        return pow(vec_sq.x + vec_sq.y, 0.5)

    def cross(self, other):
        if self.a != other.a:
            raise ValueError("Cross-multiplying vectors with different 'a'")
        # TODO:
        return self.length * other.length * math.sin(self.angle(other))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-5:
        logging.warning("WARNING! detected an epsilon norm!")
        return np.zeros_like(v)
    return v / norm


def rotate_vector_by_angle(v, deg):
    """Rotates a 2D vector by DEG degrees counterclockwise."""
    rotation_matrix = np.array([[np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg))],
                                [np.sin(np.deg2rad(deg)),  np.cos(np.deg2rad(deg))]])
    return np.dot(rotation_matrix, v)
