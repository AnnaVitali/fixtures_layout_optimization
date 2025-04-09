from inertia_analysis.moments_of_inerta import InertiaAnalysis
from utility.fixtures_utiility import create_fixture

MIN_X = 0
MIN_Y = 0
MIN_CUP_TYPE = 2
MAX_CUP_TYPE = 1


class MinMaxBounds:

    def __init__(self, workpiece_vertices):
        self.workpiece_vertices = workpiece_vertices

    def get_max_moments_fixtures(self):
        max_x, max_y = max(self.workpiece_vertices)
        vertices_x, vertices_y = create_fixture(MAX_CUP_TYPE, max_x, max_y)
        vertices = list(zip(vertices_x, vertices_y))

        jx, jy, jxy = InertiaAnalysis.compute_absolute_moments_of_inertia(vertices)

        return jx, jy, jxy

    def get_max_multiplication_of_terms(self):
        max_x, max_y = max(self.workpiece_vertices)
        vertices_x, vertices_y = create_fixture(MAX_CUP_TYPE, max_x, max_y)
        vertices = list(zip(vertices_x, vertices_y))

        terms_jx = self.__compute_jx_terms(vertices)
        terms_jy = self.__compute_jy_terms(vertices)
        terms_jxy = self.__compute_jxy_terms(vertices)
        common_terms = self.__compute_common_temrs(vertices)

        mul_term_jx = []
        mul_term_jy = []
        mul_term_jxy = []

        for i in range(len(terms_jx)):
            mul_term_jx.append(terms_jx[i] * common_terms[i])
            mul_term_jy.append(terms_jy[i] * common_terms[i])
            mul_term_jxy.append(terms_jxy[i] * common_terms[i])

        return mul_term_jx, mul_term_jy, mul_term_jxy

    def get_min_multiplication_of_terms(self):
        vertices_x, vertices_y = create_fixture(MIN_CUP_TYPE, MIN_X, MIN_Y)
        vertices = list(zip(vertices_x, vertices_y))

        terms_jx = self.__compute_jx_terms(vertices)
        terms_jy = self.__compute_jy_terms(vertices)
        terms_jxy = self.__compute_jxy_terms(vertices)
        common_terms = self.__compute_common_temrs(vertices)

        mul_term_jx = []
        mul_term_jy = []
        mul_term_jxy = []

        for i in range(len(terms_jx)):
            mul_term_jx.append(terms_jx[i] * common_terms[i])
            mul_term_jy.append(terms_jy[i] * common_terms[i])
            mul_term_jxy.append(terms_jxy[i] * common_terms[i])

        return mul_term_jx, mul_term_jy, mul_term_jxy


    def get_min_jx_terms(self):
        vertices_x, vertices_y = create_fixture(MIN_CUP_TYPE, MIN_X, MIN_Y)
        vertices = list(zip(vertices_x, vertices_y))

        return self.__compute_jx_terms(vertices)

    def get_max_jx_terms(self):
        max_x, max_y = max(self.workpiece_vertices)
        vertices_x, vertices_y = create_fixture(MAX_CUP_TYPE, max_x, max_y)
        vertices = list(zip(vertices_x, vertices_y))

        return self.__compute_jx_terms(vertices)

    def get_min_jy_terms(self):
        vertices_x, vertices_y = create_fixture(MIN_CUP_TYPE, MIN_X, MIN_Y)
        vertices = list(zip(vertices_x, vertices_y))

        return self.__compute_jy_terms(vertices)

    def get_max_jy_terms(self):
        max_x, max_y = max(self.workpiece_vertices)
        vertices_x, vertices_y =  create_fixture(MAX_CUP_TYPE, max_x, max_y)
        vertices = list(zip(vertices_x, vertices_y))

        return self.__compute_jy_terms(vertices)

    def get_min_jxy_terms(self):
        vertices_x, vertices_y = create_fixture(MIN_CUP_TYPE, MIN_X, MIN_Y)
        vertices = list(zip(vertices_x, vertices_y))

        return self.__compute_jxy_terms(vertices)

    def get_max_jxy_terms(self):
        max_x, max_y = max(self.workpiece_vertices)
        vertices_x, vertices_y =  create_fixture(MAX_CUP_TYPE, max_x, max_y)
        vertices = list(zip(vertices_x, vertices_y))

        return self.__compute_jxy_terms(vertices)

    def get_min_common_terms(self):
        vertices_x, vertices_y = create_fixture(MIN_CUP_TYPE, MIN_X, MIN_Y)
        vertices = list(zip(vertices_x, vertices_y))

        return self.__compute_common_temrs(vertices)

    def get_max_common_terms(self):
        max_x, max_y = max(self.workpiece_vertices)
        vertices_x, vertices_y =  create_fixture(MAX_CUP_TYPE, max_x, max_y)
        vertices = list(zip(vertices_x, vertices_y))

        return self.__compute_common_temrs(vertices)

    def __compute_jx_terms(self, vertices):
        n = len(vertices) - 1

        term_jx = []

        for i in range(n):
            x_i, y_i = vertices[i]
            x_next, y_next = vertices[i + 1]

            term_jx.append((y_i ** 2 + y_i * y_next + y_next ** 2))

        return term_jx

    def __compute_jy_terms(self, vertices):
        n = len(vertices) - 1

        term_jy = []

        for i in range(n):
            x_i, y_i = vertices[i]
            x_next, y_next = vertices[i + 1]

            term_jy.append((x_i ** 2 + x_i * x_next + x_next ** 2))

        return term_jy

    def __compute_jxy_terms(self, vertices):
        n = len(vertices) - 1

        term_jxy = []

        for i in range(n):
            x_i, y_i = vertices[i]
            x_next, y_next = vertices[i + 1]

            term_jxy.append((x_i * y_next + 2 * x_i * y_i + 2 * x_next * y_next + x_next * y_i))

        return term_jxy

    def __compute_common_temrs(self, vertices):
        n = len(vertices) - 1
        common_term = []

        for i in range(n):
            x_i, y_i = vertices[i]
            x_next, y_next = vertices[i + 1]

            common_term.append(x_i * y_next - x_next * y_i)

        return common_term