import numpy as np
import math

AREA = "area"
ABSOLUTE_MOMENTS_OF_INERTIA = "absolute_moments_of_inertia"
BARICENTRIC_MOMENTS_OF_INERTIA = "baricentric_moments_of_inertia"
BARYCENTER = "barycenter"
ANGLE = "angle"


class InertiaAnalysis:

    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_polygon_area(vertices):
        x = np.array([v[0] for v in vertices])
        y = np.array([v[1] for v in vertices])
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def compute_absolute_moments_of_inertia(vertices):
        n = len(vertices) - 1

        jx = 0
        jy = 0
        jxy = 0
        print(f"vertices: {vertices}")
        for i in range(n):
            x_i, y_i = vertices[i]
            x_next, y_next = vertices[i + 1]

            common_term = x_i * y_next - x_next * y_i

            jx += (y_i ** 2 + y_i * y_next + y_next ** 2) * common_term
            jy += (x_i ** 2 + x_i * x_next + x_next ** 2) * common_term
            jxy += (x_i * y_next + 2 * x_i * y_i + 2 * x_next * y_next + x_next * y_i) * common_term

        jx = abs(jx) / 12
        jy = abs(jy) / 12
        jxy = abs(jxy) / 24

        return jx, jy, jxy

    @staticmethod
    def compute_baricentric_moments_of_inertia(polygon):
        area = polygon[AREA]
        x_g, y_g = polygon[BARYCENTER]
        jx, jy, jxy = polygon[ABSOLUTE_MOMENTS_OF_INERTIA]
        angle_rad = polygon[ANGLE]
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        i_prime = jx - y_g ** 2 * area
        j_prime = jy - x_g ** 2 * area
        ij_prime = jxy - y_g * x_g * area

        i = i_prime * cos_angle ** 2 + j_prime * sin_angle ** 2 - ij_prime * 2 * sin_angle * cos_angle
        j = i_prime * sin_angle ** 2 + j_prime * cos_angle ** 2 + ij_prime * 2 * sin_angle * cos_angle
        ij = (i_prime - j_prime) * sin_angle * cos_angle + ij_prime * (cos_angle ** 2 - sin_angle ** 2)

        return i, j, ij

    @staticmethod
    def compute_overall_center_of_gravity(polygons):
        total_area = 0
        weighted_cx_sum = 0
        weighted_cy_sum = 0

        for poly in polygons:
            area = poly[AREA]
            barycenter_x , barycenter_y = poly[BARYCENTER]

            total_area += area
            weighted_cx_sum += area * barycenter_x
            weighted_cy_sum += area * barycenter_y

        x_g = weighted_cx_sum / total_area
        y_g = weighted_cy_sum / total_area

        return x_g, y_g

    @staticmethod
    def compute_combined_absolute_moment_of_inertia(polygons):

        jx_total, jy_total, jxy_total = 0, 0, 0

        for poly in polygons:
            jxi, jyi, jxyi = poly[ABSOLUTE_MOMENTS_OF_INERTIA]
            jx_total += jxi
            jy_total += jyi
            jxy_total += jxyi

        return jx_total, jy_total, jxy_total

    @staticmethod
    def compute_combined_baricentric_moments_of_inertia(polygons, x_G, y_G):
        jx_total, jy_total, jxy_total = 0, 0, 0
        area_total = 0

        for poly in polygons:
            jxi, jyi, jxyi = poly[ABSOLUTE_MOMENTS_OF_INERTIA]
            area_total += poly[AREA]
            jx_total += jxi
            jy_total += jyi
            jxy_total += jxyi

        jx = jx_total - y_G ** 2 * area_total
        jy = jy_total - x_G ** 2 * area_total
        jxy = jxy_total - y_G * x_G * area_total


        i = (jx + jy) / 2 - 0.5 * math.sqrt((jx - jy) ** 2 + 4 * jxy ** 2)
        j = (jx + jy) / 2 + 0.5 * math.sqrt((jx - jy) ** 2 + 4 * jxy ** 2)

        return i, j


