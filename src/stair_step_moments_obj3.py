import gurobipy as gp
from gurobipy import GRB

from optimization.min_max_bounds import MinMaxBounds
from utility.geometry_utility import line_equation
from utility.result_displayer import ResultDisplayer
from optimization.optimization_callback import OptimizationCallback
import os
import json

WIDTH = 69.8
HEIGHT = 38.0
AREA = 1757.69
VERTEX_A = (66.5, 39.4)
VERTEX_B = (2.0, 26.2)
VERTEX_C = (2.0, 13.5)
VERTEX_D = (66.5, 0.4)
VERTEX_E = (68.9, 1.7)
VERTEX_F = (68.9, 38.0)

BAR_SIZE = 14.5

FIXTURE_A_NUMBER = 5
FIXTURE_A_DIMX = 14.5
FIXTURE_A_DIMY = 14.5

FIXTURE_B_NUMBER = 5
FIXTURE_B_DIMX = 14.5
FIXTURE_B_DIMY = 5.5

N_FIXTURES_TYPE = 2
FIXTURE_TYPE = [1] * FIXTURE_A_NUMBER + [2] * FIXTURE_B_NUMBER
FIXTURE_AREA = [FIXTURE_A_DIMX * FIXTURE_A_DIMY] * FIXTURE_A_NUMBER + [
    FIXTURE_B_DIMX * FIXTURE_B_DIMY] * FIXTURE_B_NUMBER

FIXTURES_NUMBER = FIXTURE_A_NUMBER + FIXTURE_B_NUMBER

DIMS = [[FIXTURE_A_DIMX, FIXTURE_A_DIMY]] * FIXTURE_A_NUMBER + [[FIXTURE_B_DIMX, FIXTURE_B_DIMY]] * FIXTURE_B_NUMBER
#HALF_DIM = [[73, 73]] * FIXTURE_A_NUMBER + [[73, 28]] * FIXTURE_B_NUMBER#[[FIXTURE_A_DIMX / 2, FIXTURE_A_DIMY / 2]] * FIXTURE_A_NUMBER + [[FIXTURE_B_DIMX / 2, FIXTURE_B_DIMY / 2]] * FIXTURE_B_NUMBER
HALF_DIM = [[FIXTURE_A_DIMX / 2, FIXTURE_A_DIMY / 2]] * FIXTURE_A_NUMBER + [[FIXTURE_B_DIMX / 2, FIXTURE_B_DIMY / 2]] * FIXTURE_B_NUMBER
EPS = 1.0
M = (WIDTH * HEIGHT) ** 2

line_a_q = VERTEX_B[0]
line_b_m, line_b_q = line_equation(VERTEX_B[0], VERTEX_B[1], VERTEX_A[0], VERTEX_A[1])
line_c_m, line_c_q = line_equation(VERTEX_A[0], VERTEX_A[1], VERTEX_F[0], VERTEX_F[1])
line_d_q = VERTEX_E[0]
line_e_m, line_e_q = line_equation(VERTEX_E[0], VERTEX_E[1], VERTEX_D[0], VERTEX_D[1])
line_f_m, line_f_q = line_equation(VERTEX_D[0], VERTEX_D[1], VERTEX_C[0], VERTEX_C[1])

print("\n----------------------Geometry Line Equation---------------------\n")
print(f"line a: x={line_a_q}")
print(f"line b: y={line_b_m:.2f}x + {line_b_q:.2f}")
print(f"line c: y={line_c_m:.2f}x + {line_c_q:.2f}")
print(f"line d: x={line_d_q}")
print(f"line e: y={line_e_m:.2f}x {line_e_q:.2f}")
print(f"line f: y={line_f_m:.2f}x + {line_f_q:.2f}")

print(print("\n----------------------Model resolution---------------------\n"))

n_stay_below_line = 2
m_stay_below = [line_b_m, line_c_m]
q_stay_below = [line_b_q, line_c_q]

n_stay_above_line = 2
m_stay_above = [line_e_m, line_f_m]
q_stay_above = [line_e_q, line_f_q]

n_stay_left_line = 1
q_stay_left = [line_a_q]

n_stay_right_line = 1
q_stay_right = [line_d_q]

model = gp.Model("fixture_layout_optimization")

bounds = MinMaxBounds([VERTEX_A, VERTEX_B, VERTEX_C, VERTEX_D, VERTEX_E, VERTEX_F])

#Variables
x = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x")
x2 = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x2")
x3 = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x3")
x4 = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x4")

y = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y")
y2 = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y2")
y3 = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y3")
y4 = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y4")

selected_fixture = model.addVars(FIXTURES_NUMBER, vtype=GRB.BINARY, name="selected_fixture")
pair_selected = model.addVars(FIXTURES_NUMBER, FIXTURES_NUMBER, vtype=GRB.BINARY, name="x_selected")

no_overlap = model.addVars(FIXTURES_NUMBER, FIXTURES_NUMBER, 4, vtype=GRB.BINARY, name="no_overlap")

x1_greater_x2 = model.addVars(FIXTURES_NUMBER, FIXTURES_NUMBER, vtype=GRB.BINARY, name=f"x1_greater_x2")
x2_greater_x1 = model.addVars(FIXTURES_NUMBER, FIXTURES_NUMBER, vtype=GRB.BINARY, name=f"x2_greater_x1")

fixtures_center_x = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="fixtures_center_x")
fixtures_center_y = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="fixtures_center_y")

weighted_cx_sum = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="weighted_cx_sum")
weighted_cy_sum = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="weighted_cy_sum")
x_g = model.addVar(lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x_g")
y_g = model.addVar(lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y_g")

x_square = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x1_square")
x2_square = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x2_square")
x3_square = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x3_square")
x4_square = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x4_square")

y_square = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y1_square")
y2_square = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y2_square")
y3_square = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y3_square")
y4_square = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y4_square")

xy_bl = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="xy_bl")
xy_br = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="xy_br")
xy_tl = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="xy_tl")
xy_tr = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="xy_tr")

max_jx, max_jy, max_jxy = bounds.get_max_moments_fixtures()
jx_fixture = model.addVars(FIXTURES_NUMBER, lb=0, ub=max_jx, vtype=GRB.CONTINUOUS, name="jx_fixture")
jy_fixture = model.addVars(FIXTURES_NUMBER, lb=0, ub=max_jy, vtype=GRB.CONTINUOUS, name="jy_fixture")
jxy_fixture = model.addVars(FIXTURES_NUMBER, lb=0, ub=max_jxy, vtype=GRB.CONTINUOUS, name="jxy_fixture")

max_jx_terms, max_jy_terms, max_jxy_terms = bounds.get_max_multiplication_of_terms()
min_jx_terms, min_jy_terms, min_jxy_terms  = bounds.get_min_multiplication_of_terms()
term1_jx = model.addVars(FIXTURES_NUMBER, lb=min_jx_terms[0] if min_jx_terms[0] < 0 else 0,
                         ub=max_jx_terms[0] if max_jx_terms[0] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term1_jx")
term2_jx = model.addVars(FIXTURES_NUMBER,  lb=max_jx_terms[1] if max_jx_terms[1] < 0 else 0,
                         ub=min_jx_terms[1] if min_jx_terms[1] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term2_jx")
term3_jx = model.addVars(FIXTURES_NUMBER,  lb=max_jx_terms[2] if max_jx_terms[2] < 0 else 0,
                         ub=min_jx_terms[2] if min_jx_terms[2] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term3_jx")
term4_jx = model.addVars(FIXTURES_NUMBER,  lb=min_jx_terms[0] if min_jx_terms[0] < 0 else 0,
                         ub=max_jx_terms[3] if max_jx_terms[3] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term4_jx")

term1_jy = model.addVars(FIXTURES_NUMBER, lb=min_jy_terms[0] if min_jy_terms[0] < 0 else 0,
                         ub=max_jy_terms[0] if max_jy_terms[0] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term1_jy")
term2_jy = model.addVars(FIXTURES_NUMBER, lb=max_jy_terms[1] if max_jy_terms[1] < 0 else 0,
                         ub=min_jy_terms[1] if min_jy_terms[1] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term2_jy")
term3_jy = model.addVars(FIXTURES_NUMBER, lb=max_jy_terms[2] if max_jy_terms[2] < 0 else 0,
                         ub=min_jy_terms[2] if min_jy_terms[2] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term3_jy")
term4_jy = model.addVars(FIXTURES_NUMBER, lb=min_jy_terms[3] if min_jy_terms[3] < 0 else 0,
                         ub=max_jy_terms[3] if max_jy_terms[3] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term4_jy")

term1_jxy = model.addVars(FIXTURES_NUMBER, lb=min_jxy_terms[0] if min_jxy_terms[0] < 0 else 0,
                         ub=max_jxy_terms[0] if max_jxy_terms[0] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term1_jxy")
term2_jxy = model.addVars(FIXTURES_NUMBER, lb=max_jxy_terms[1] if max_jxy_terms[1] < 0 else 0,
                         ub=min_jxy_terms[1] if min_jxy_terms[1] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term2_jxy")
term3_jxy = model.addVars(FIXTURES_NUMBER, lb=max_jxy_terms[2] if max_jxy_terms[2] < 0 else 0,
                         ub=min_jxy_terms[2] if min_jxy_terms[2] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term3_jxy")
term4_jxy = model.addVars(FIXTURES_NUMBER, lb=min_jxy_terms[3] if min_jxy_terms[3] < 0 else 0,
                         ub=max_jxy_terms[3] if max_jxy_terms[3] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term4_jxy")

jx_g = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="jx_g")
jy_g = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="jy_g")
jxy_g = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="jxy_g")

area_total = model.addVar(lb=0, ub=sum(FIXTURE_AREA), vtype=GRB.CONTINUOUS, name="area_total")
x_g_squared = model.addVar(lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x_g_squared")
y_g_squared = model.addVar(lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y_g_squared")
mul_x_g_y_g = model.addVar(lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="mul_x_g_y_g")

jx = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="jx")
jy = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="jy")
jxy = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="jxy")

sub_jx_jy = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="jx_jy_squared")
sub_jx_jy_squared = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="sub_jx_jy_squared")
jxy_squared = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="jxy_squared")

root = model.addVar(lb=0, ub=1e20, vtype=GRB.CONTINUOUS, name="root")
square_of_root = model.addVar(lb=0, ub=1e20, vtype=GRB.CONTINUOUS, name="square_of_root")

i = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="i")
j = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="i")

#Geometry constraints
#fit in workpiece area
model.addConstrs((x[c] + DIMS[c][0]) * selected_fixture[c] <= WIDTH for c in range(FIXTURES_NUMBER))
model.addConstrs((y[c] + DIMS[c][1]) * selected_fixture[c] <= HEIGHT for c in range(FIXTURES_NUMBER))

model.addConstrs(x[c] >= (q_stay_left[l] + EPS) * selected_fixture[c] for c in range(FIXTURES_NUMBER) for l in
                 range(n_stay_left_line))
model.addConstrs(
    (x[c] + DIMS[c][0] + EPS) * selected_fixture[c] <= q_stay_right[l] for c in range(FIXTURES_NUMBER) for l in
    range(n_stay_right_line))

model.addConstrs(
    (y[c] - EPS) * selected_fixture[c] >= (m_stay_above[l] * x[c] + q_stay_above[l]) * selected_fixture[c]
    for c in range(FIXTURES_NUMBER) for l in range(n_stay_above_line))
model.addConstrs((y[c] - EPS) * selected_fixture[c] >= (m_stay_above[l] * (x[c] + DIMS[c][0]) + q_stay_above[l]) *
                 selected_fixture[c]
                 for c in range(FIXTURES_NUMBER) for l in range(n_stay_above_line))
# model.addConstrs(((y[c] + DIMS[c][1]) - EPS) * selected_fixture[c] >= (m_stay_above[l] * x[c] + q_stay_above[l]) *
#                  selected_fixture[c]
#                  for c in range(FIXTURES_NUMBER) for l in range(n_stay_above_line))
# model.addConstrs(
#     ((y[c] + DIMS[c][1]) - EPS) * selected_fixture[c] >= (m_stay_above[l] * (x[c] + DIMS[c][0]) + q_stay_above[l]) *
#     selected_fixture[
#         c]
#     for c in range(FIXTURES_NUMBER) for l in range(n_stay_above_line))

# model.addConstrs(
#     (y[c] + EPS) * selected_fixture[c] <= (m_stay_below[l] * x[c] + q_stay_below[l]) * selected_fixture[c]
#     for c in range(FIXTURES_NUMBER) for l in range(n_stay_below_line))
# model.addConstrs((y[c] + EPS) * selected_fixture[c] <= (m_stay_below[l] * (x[c] + DIMS[c][0]) + q_stay_below[l]) *
#                  selected_fixture[c]
#                  for c in range(FIXTURES_NUMBER) for l in range(n_stay_below_line))
model.addConstrs(((y[c] + DIMS[c][1]) + EPS) * selected_fixture[c] <= (m_stay_below[l] * x[c] + q_stay_below[l]) *
                 selected_fixture[c]
                 for c in range(FIXTURES_NUMBER) for l in range(n_stay_below_line))
model.addConstrs(
    ((y[c] + DIMS[c][1]) + EPS) * selected_fixture[c] <= (m_stay_below[l] * (x[c] + DIMS[c][0]) + q_stay_below[l]) *
    selected_fixture[
        c]
    for c in range(FIXTURES_NUMBER) for l in range(n_stay_below_line))


# no overlap between fixtures
model.addConstrs(
    pair_selected[c1, c2] <= selected_fixture[c1] for c1 in range(FIXTURES_NUMBER) for c2 in range(FIXTURES_NUMBER)
    if c1 != c2)
model.addConstrs(
    pair_selected[c1, c2] <= selected_fixture[c2] for c1 in range(FIXTURES_NUMBER) for c2 in range(FIXTURES_NUMBER)
    if c1 != c2)
model.addConstrs(
    pair_selected[c1, c2] >= selected_fixture[c1] + selected_fixture[c2] - 1 for c1 in range(FIXTURES_NUMBER) for c2
    in
    range(FIXTURES_NUMBER) if c1 != c2)

model.addConstrs((x[c1] + DIMS[c1][0]) * pair_selected[c1, c2] <= x[c2] + M * (1 - no_overlap[c1, c2, 0]) for c1 in
                 range(FIXTURES_NUMBER) for c2 in range(c1 + 1, FIXTURES_NUMBER))
model.addConstrs((x[c2] + DIMS[c2][0]) * pair_selected[c1, c2] <= x[c1] + M * (1 - no_overlap[c1, c2, 1]) for c1 in
                 range(FIXTURES_NUMBER) for c2 in range(c1 + 1, FIXTURES_NUMBER))
model.addConstrs((y[c1] + DIMS[c1][1]) * pair_selected[c1, c2] <= y[c2] + M * (1 - no_overlap[c1, c2, 2]) for c1 in
                 range(FIXTURES_NUMBER) for c2 in range(c1 + 1, FIXTURES_NUMBER))
model.addConstrs((y[c2] + DIMS[c2][1]) * pair_selected[c1, c2] <= y[c1] + M * (1 - no_overlap[c1, c2, 3]) for c1 in
                 range(FIXTURES_NUMBER) for c2 in range(c1 + 1, FIXTURES_NUMBER))

model.addConstrs(
    gp.quicksum(no_overlap[c1, c2, i] for i in range(4)) >= 1 for c1 in range(FIXTURES_NUMBER) for c2 in
    range(c1 + 1, FIXTURES_NUMBER))

# respect bar distance
model.addConstrs(
    x[c1] * pair_selected[c1, c2] >= (x[c2] + EPS - M * (1 - x1_greater_x2[c1, c2])) * pair_selected[c1, c2]
    for c1 in range(FIXTURES_NUMBER) for c2 in range(FIXTURES_NUMBER) if c1 != c2)
model.addConstrs(x[c1] * pair_selected[c1, c2] <= (x[c2] + M * x1_greater_x2[c1, c2]) * pair_selected[c1, c2]
                 for c1 in range(FIXTURES_NUMBER) for c2 in range(FIXTURES_NUMBER) if c1 != c2)
model.addConstrs(
    x[c2] * pair_selected[c1, c2] >= (x[c1] + EPS - M * (1 - x2_greater_x1[c1, c2])) * pair_selected[c1, c2]
    for c1 in range(FIXTURES_NUMBER) for c2 in range(FIXTURES_NUMBER) if c1 != c2)
model.addConstrs(x[c2] * pair_selected[c1, c2] <= (x[c1] + M * x2_greater_x1[c1, c2]) * pair_selected[c1, c2]
                 for c1 in range(FIXTURES_NUMBER) for c2 in range(FIXTURES_NUMBER) if c1 != c2)

model.addConstrs(
    (x[c2] + DIMS[c2][0] + BAR_SIZE) * pair_selected[c1, c2] <= (x[c1] + M * (1 - x1_greater_x2[c1, c2])) *
    pair_selected[c1, c2]
    for c1 in range(FIXTURES_NUMBER) for c2 in range(FIXTURES_NUMBER) if c1 != c2)
model.addConstrs(
    (x[c1] + DIMS[c1][0] + BAR_SIZE) * pair_selected[c1, c2] <= (x[c2] + M * (1 - x2_greater_x1[c1, c2])) *
    pair_selected[c1, c2]
    for c1 in range(FIXTURES_NUMBER) for c2 in range(FIXTURES_NUMBER) if c1 != c2)

# symmetry breaking constraints
model.addConstr(gp.quicksum(selected_fixture[c] for c in range(FIXTURES_NUMBER)) >= 1, name="at_least_one_fixture")

model.addConstrs((selected_fixture[c] == 0) >> (x[c] <= 0) for c in range(FIXTURES_NUMBER))
model.addConstrs((selected_fixture[c] == 0) >> (y[c] <= 0) for c in range(FIXTURES_NUMBER))

model.addConstrs(
    (x[c1] + y[c1]) + EPS >= (x[c2] + y[c2]) for c1 in range(FIXTURES_NUMBER) for c2 in
    range(c1 + 1, FIXTURES_NUMBER) if
    FIXTURE_TYPE[c1] == FIXTURE_TYPE[c2])

#Objective function
#compute the center of cup
model.addConstrs(
    fixtures_center_x[c] == (x[c] + HALF_DIM[c][0]) * selected_fixture[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(
    fixtures_center_y[c] == (y[c] + HALF_DIM[c][1]) * selected_fixture[c] for c in range(FIXTURES_NUMBER))

model.addConstr(
    weighted_cx_sum == gp.quicksum(
        FIXTURE_AREA[c] * fixtures_center_x[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)),
    name="weighted_cx_sum")
model.addConstr(
    weighted_cy_sum == gp.quicksum(
        FIXTURE_AREA[c] * fixtures_center_y[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)),
    name="weighted_cy_sum")

model.addConstr(area_total == gp.quicksum(FIXTURE_AREA[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)),
                name="area_total")

#compute the overall center of gravity
model.addGenConstrNL(x_g, weighted_cx_sum / area_total, name="x_g")
model.addGenConstrNL(y_g, weighted_cy_sum / area_total, name="y_g")


#compute fixtures vertices
model.addConstrs(x2[c] == x[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(x3[c] == (x[c] + DIMS[c][0]) * selected_fixture[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(x4[c] == (x[c] + DIMS[c][0]) * selected_fixture[c] for c in range(FIXTURES_NUMBER))

model.addConstrs(y2[c] == (y[c] + DIMS[c][1]) * selected_fixture[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(y3[c] == (y[c] + DIMS[c][1]) * selected_fixture[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(y4[c] == y[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER))

for c in range(FIXTURES_NUMBER):
    model.addGenConstrPow(y[c], y_square[c], 2)
    model.addGenConstrPow(y2[c], y2_square[c], 2)
    model.addGenConstrPow(y3[c], y3_square[c], 2)
    model.addGenConstrPow(y4[c], y4_square[c], 2)

    model.addGenConstrPow(x[c], x_square[c], 2)
    model.addGenConstrPow(x2[c], x2_square[c], 2)
    model.addGenConstrPow(x3[c], x3_square[c], 2)
    model.addGenConstrPow(x4[c], x4_square[c], 2)

#moments of inertia of fixtures
model.addConstrs(xy_bl[c] == x[c] * y[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(xy_tl[c] == x2[c] * y2[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(xy_tr[c] == x3[c] * y3[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(xy_br[c] == x4[c] * y4[c] for c in range(FIXTURES_NUMBER))

max_jx, max_jy, max_jxy = bounds.get_max_moments_fixtures()

min_common_terms = bounds.get_min_common_terms()
max_common_terms = bounds.get_max_common_terms()

max_jx_terms_part1 = bounds.get_max_jx_terms()
min_jx_terms_part1 = bounds.get_min_jx_terms()

max_jy_terms_part1 = bounds.get_max_jy_terms()
min_jy_terms_part1 = bounds.get_min_jy_terms()

max_jxy_terms_part1 = bounds.get_max_jxy_terms()
min_jxy_terms_part1 = bounds.get_min_jxy_terms()


for c in range(FIXTURES_NUMBER):
    com_term1 = model.addVar(lb=min_common_terms[0] if min_common_terms[0] < 0 else 0,
                      ub=max_common_terms[0] if max_common_terms[0] > 0 else 0, vtype=GRB.CONTINUOUS, name="com_term1")
    com_term2 = model.addVar(lb=max_common_terms[1] if max_common_terms[1] < 0 else 0,
                      ub=min_common_terms[1] if min_common_terms[1] > 0 else 0, vtype=GRB.CONTINUOUS, name="com_term2")
    com_term3 = model.addVar(lb=max_common_terms[2] if max_common_terms[2] < 0 else 0,
                      ub=min_common_terms[2] if min_common_terms[2] > 0 else 0, vtype=GRB.CONTINUOUS, name="com_term3")
    com_term4 = model.addVar(lb=min_common_terms[3] if min_common_terms[3] < 0 else 0,
                      ub=max_common_terms[3] if max_common_terms[3] > 0 else 0, vtype=GRB.CONTINUOUS, name="com_term4")

    model.addConstr(com_term1 == xy_tl[c] - xy_bl[c])
    model.addConstr(com_term2 == xy_tl[c] - xy_tr[c])
    model.addConstr(com_term3 == xy_br[c] - xy_tr[c])
    model.addConstr(com_term4 == xy_br[c] - xy_bl[c])

    #jx moments of fixture's area
    jx_term1_part1 = gp.QuadExpr(y_square[c] + y[c] * y2[c] + y2_square[c])
    jx1 = model.addVar(lb=min_jx_terms_part1[0] if min_jx_terms_part1[0] < 0 else 0,
                       ub=max_jx_terms_part1[0] if max_jx_terms_part1[0] > 0 else 0, vtype=GRB.CONTINUOUS, name="jx1")
    model.addGenConstrNL(jx1, jx_term1_part1)
    model.addGenConstrNL(term1_jx[c], jx_term1_part1 * com_term1, name=f"term1_jx_{c}")

    jx_term2_part1 = gp.QuadExpr(y2_square[c] + y2[c] * y3[c] + y3_square[c])
    jx2 = model.addVar(lb=min_jx_terms_part1[1] if min_jx_terms_part1[1] < 0 else 0,
                       ub=max_jx_terms_part1[1] if max_jx_terms_part1[1] > 0 else 0, vtype=GRB.CONTINUOUS, name="jx2")
    model.addGenConstrNL(jx2, jx_term2_part1)
    model.addGenConstrNL(term2_jx[c], jx_term2_part1 * com_term2, name=f"term2_jx_{c}")

    jx_term3_part1 = gp.QuadExpr(y3_square[c] + y3[c] * y4[c] + y4_square[c])
    jx3 = model.addVar(lb=min_jx_terms_part1[2] if min_jx_terms_part1[2] < 0 else 0,
                       ub=max_jx_terms_part1[2] if max_jx_terms_part1[2] > 0 else 0, vtype=GRB.CONTINUOUS, name="jx3")
    model.addGenConstrNL(jx3, jx_term3_part1)
    model.addGenConstrNL(term3_jx[c], jx_term3_part1 * com_term3, name=f"term3_jx_{c}")

    jx_term4_part1 = gp.QuadExpr(y4_square[c] + y4[c] * y[c] + y_square[c])
    jx4 = model.addVar(lb=min_jx_terms_part1[3] if min_jx_terms_part1[3] < 0 else 0,
                       ub=max_jx_terms_part1[3] if max_jx_terms_part1[3] > 0 else 0, vtype=GRB.CONTINUOUS, name="jx4")
    model.addGenConstrNL(jx4, jx_term4_part1)
    model.addGenConstrNL(term4_jx[c], jx_term4_part1 * com_term4, name=f"term4_jx_{c}")

    res_jx = model.addVar(lb=-max_jx, ub=max_jx, vtype=GRB.CONTINUOUS, name="res_jx")
    model.addGenConstrNL(res_jx, (term1_jx[c] + term2_jx[c] + term3_jx[c] + term4_jx[c]) / 12)
    model.addGenConstrAbs(jx_fixture[c], res_jx, "jx_fixture_abs")

    #jy moments of fixture's area
    jy_term1_part1 = gp.QuadExpr(x_square[c] + x[c] * x2[c] + x2_square[c])
    jy1 = model.addVar(lb=min_jy_terms_part1[0] if min_jy_terms_part1[0] < 0 else 0,
                       ub=max_jy_terms_part1[0] if max_jy_terms_part1[0] > 0 else 0, vtype=GRB.CONTINUOUS, name="jy1")
    model.addGenConstrNL(jy1, jy_term1_part1)
    model.addGenConstrNL(term1_jy[c], jy1 * com_term1, name=f"term1_jy_{c}")

    jy_term2_part1 = gp.QuadExpr(x2_square[c] + x2[c] * x3[c] + x3_square[c])
    jy2 = model.addVar(lb=min_jy_terms_part1[1] if min_jy_terms_part1[1] < 0 else 0,
                       ub=max_jy_terms_part1[1] if max_jy_terms_part1[1] > 0 else 0, vtype=GRB.CONTINUOUS, name="jy2")
    model.addGenConstrNL(jy2, jy_term2_part1)
    model.addGenConstrNL(term2_jy[c], jy2 * com_term2, name=f"term2_jy_{c}")

    jy_term3_part1 = gp.QuadExpr(x3_square[c] + x3[c] * x4[c] + x4_square[c])
    jy3 = model.addVar(lb=min_jy_terms_part1[2] if min_jy_terms_part1[2] < 0 else 0,
                       ub=max_jy_terms_part1[2] if max_jy_terms_part1[2] > 0 else 0, vtype=GRB.CONTINUOUS, name="jy3")
    model.addGenConstrNL(jy3, jy_term3_part1)
    model.addGenConstrNL(term3_jy[c], jy3 * com_term3, name=f"term3_jy_{c}")

    jy_term4_part1 = gp.QuadExpr(x4_square[c] + x4[c] * x[c] + x_square[c])
    jy4 = model.addVar(lb=min_jy_terms_part1[3] if min_jy_terms_part1[3] < 0 else 0,
                       ub=max_jy_terms_part1[3] if max_jy_terms_part1[3] > 0 else 0, vtype=GRB.CONTINUOUS, name="jy4")
    model.addGenConstrNL(jy4, jy_term4_part1)
    model.addGenConstrNL(term4_jy[c], jy4 * com_term4, name=f"term4_jy_{c}")

    res_jy = model.addVar(lb=-max_jy, ub=max_jy, vtype=GRB.CONTINUOUS, name="res_jy")
    model.addGenConstrNL(res_jy, (term1_jy[c] + term2_jy[c] + term3_jy[c] + term4_jy[c]) / 12)

    model.addGenConstrAbs(jy_fixture[c], res_jy, "jy_fixture_abs")

    #jxy moments of fixture's area
    jxy_term1_part1 = gp.QuadExpr(x[c] * y2[c] + 2 * x[c] * y[c] + 2 * x2[c] * y2[c] + x2[c] * y[c])
    jxy1 = model.addVar(lb=min_jxy_terms_part1[0] if min_jxy_terms_part1[0] < 0 else 0,
                        ub=max_jxy_terms_part1[0] if max_jxy_terms_part1[0] > 0 else 0, vtype=GRB.CONTINUOUS,
                        name="jxy1")
    model.addGenConstrNL(jxy1, jxy_term1_part1)
    model.addGenConstrNL(term1_jxy[c], jxy1 * com_term1, name=f"term1_jxy_{c}")

    jxy_term2_part1 = gp.QuadExpr(x2[c] * y3[c] + 2 * x2[c] * y2[c] + 2 * x3[c] * y3[c] + x3[c] * y2[c])
    jxy2 = model.addVar(lb=min_jxy_terms_part1[1] if min_jxy_terms_part1[1] < 0 else 0,
                        ub=max_jxy_terms_part1[1] if max_jxy_terms_part1[1] > 0 else 0, vtype=GRB.CONTINUOUS,
                        name="jxy2")
    model.addGenConstrNL(jxy2, jxy_term2_part1)
    model.addGenConstrNL(term2_jxy[c], jxy2 * com_term2, name=f"term2_jxy_{c}")

    jxy_term3_part1 = gp.QuadExpr(x3[c] * y4[c] + 2 * x3[c] * y3[c] + 2 * x4[c] * y4[c] + x4[c] * y3[c])
    jxy3 = model.addVar(lb=min_jxy_terms_part1[2] if min_jxy_terms_part1[2] < 0 else 0,
                        ub=max_jxy_terms_part1[2] if max_jxy_terms_part1[2] > 0 else 0, vtype=GRB.CONTINUOUS,
                        name="jxy3")
    model.addGenConstrNL(jxy3, jxy_term3_part1)
    model.addGenConstrNL(term3_jxy[c], jxy3 * com_term3, name=f"term3_jxy_{c}")

    jxy_term4_part1 = gp.QuadExpr(x4[c] * y[c] + 2 * x4[c] * y4[c] + 2 * x[c] * y[c] + x[c] * y4[c])
    jxy4 = model.addVar(lb=min_jxy_terms_part1[3] if min_jxy_terms_part1[2] < 0 else 0,
                        ub=max_jxy_terms_part1[3] if max_jxy_terms_part1[2] > 0 else 0, vtype=GRB.CONTINUOUS,
                        name="jxy4")
    model.addGenConstrNL(jxy4, jxy_term4_part1)
    model.addGenConstrNL(term4_jxy[c], jxy4 * com_term4, name=f"term4_jxy_{c}")

    res_jxy = model.addVar(lb=-max_jxy, ub=max_jxy, vtype=GRB.CONTINUOUS, name="res_jxy")
    model.addGenConstrNL(res_jxy, (term1_jxy[c] + term2_jxy[c] + term3_jxy[c] + term4_jxy[c]) / 24)

    model.addGenConstrAbs(jxy_fixture[c], res_jxy, "jxy_fixture_abs")

#overall moments of inertia
model.addConstr(area_total == gp.quicksum(FIXTURE_AREA[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)), name="area_total")

model.addConstr(jx_g == gp.quicksum(jx_fixture[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)), name="jx_g")
model.addConstr(jy_g == gp.quicksum(jy_fixture[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)), name="jy_g")
model.addConstr(jxy_g == gp.quicksum(jxy_fixture[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)), name="jxy_g")

model.addGenConstrPow(x_g, x_g_squared, 2)
model.addGenConstrPow(y_g, y_g_squared, 2)
model.addConstr(mul_x_g_y_g == x_g * y_g, name="mul_x_g_y_g")

model.addGenConstrNL(jx, jx_g - area_total * y_g_squared, name="jx")
model.addGenConstrNL(jy, jy_g - area_total * x_g_squared, name="jy")
model.addGenConstrNL(jxy, jxy_g - area_total * mul_x_g_y_g, name="jxy")

model.addConstr(sub_jx_jy == jx - jy, name="jx_jy")

model.addGenConstrPow(sub_jx_jy, sub_jx_jy_squared, 2, name="sub_jx_jy_squared")

model.addGenConstrPow(jxy, jxy_squared, 2, name="jxy_squared")

quad_expr1 = gp.QuadExpr(sub_jx_jy_squared)
quad_expr2 = gp.QuadExpr(4 * jxy_squared)

#model.addConstr(root == sub_jx_jy_squared + 4 * jxy_squared, name="root")
#model.addGenConstrPow(root, square_of_root, 0.5, name="sqrt_of_root")
model.addConstr(root == sub_jx_jy_squared + 4 * jxy_squared, name="root")
#model.addConstr(square_of_root**2 == root)
model.addGenConstrPow(square_of_root, root, 2.0, name="square_of_root")

model.addGenConstrNL(i, 0.5 * (jx + jy) - 0.5 * square_of_root, name="i_equation")
model.addGenConstrNL(j, 0.5 * (jx + jy) + 0.5 * square_of_root, name="j_equation")


# model.addConstr(x[0] == 543.0)
# model.addConstr(y[0] == 31.459999999999994)
# model.addConstr(x[1] == 0.0)
# model.addConstr(y[1] == 0.0)
# model.addConstr(x[2] == 0.0)
# model.addConstr(y[2] == 0.0)
# model.addConstr(x[3] == 0.0)
# model.addConstr(y[3] == 0.0)
# model.addConstr(x[4] == 0.0)
# model.addConstr(y[4] == 0.0)
# model.addConstr(x[5] == 543.0)
# model.addConstr(y[5] == 310.51000000000005)
# model.addConstr(x[6] == 543.0000000000002)
# model.addConstr(y[6] == 255.51000000000008)
# model.addConstr(x[7] == 20.999999999999545)
# model.addConstr(y[7] == 206.10999999999993)
# model.addConstr(x[8] == 20.999999999999545)
# model.addConstr(y[8] == 135.8600000000001)
# model.addConstr(x[9] == 0.0)
# model.addConstr(y[9] == 0.0)
# model.addConstr(selected_fixture[0] == 1)
# model.addConstr(selected_fixture[1] == 0)
# model.addConstr(selected_fixture[2] == 0)
# model.addConstr(selected_fixture[3] == 0)
# model.addConstr(selected_fixture[4] == 0)
# model.addConstr(selected_fixture[5] == 1)
# model.addConstr(selected_fixture[6] == 1)
# model.addConstr(selected_fixture[7] == 1)
# model.addConstr(selected_fixture[8] == 1)
# model.addConstr(selected_fixture[9] == 0)


model.setObjective(i + j, GRB.MAXIMIZE)

x_init = [543.0, 0.0, 0.0, 0.0, 0.0, 543.0, 543.0000000000002, 20.999999999999545, 20.999999999999545, 0.0]
y_init = [31.459999999999994, 0.0, 0.0, 0.0, 0.0, 310.51000000000005, 255.51000000000008, 206.10999999999993, 135.8600000000001, 0.0]
selected_fixture_init = [1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
i_init = 4.63510e+08
j_init = 3.12943e+09


for c in range(FIXTURES_NUMBER):
    x[c].Start = x_init[c]
    y[c].Start = y_init[c]
    selected_fixture[c].Start = selected_fixture_init[c]

i.Start = i_init
j.Start = j_init

# model.setParam("MIPFocus", 1)  # Focus on finding a feasible solution
# model.setParam("Cuts", 2)       # Use aggressive cuts
# model.setParam("VarBranch", 1)  # Change variable selection for branching
# model.setParam("BranchDir", 1)  # Favor improving bounds
# model.setParam("Heuristics", 0.5)  # Increase heuristic search
# model.setParam("RINS", 10)      # Use RINS heuristic
# model.setParam("NonConvex", 2)

model.setParam("MIPFocus", 1)        # Feasibility focus
model.setParam("Heuristics", 0.5)    # Stronger emphasis on heuristics
model.setParam("Presolve", 2)        # Aggressive presolve
model.setParam("Cuts", 2)            # More cuts to reduce search space
model.setParam("RINS", 10)           # More RINS to explore near .Start
model.setParam("NoRelHeurTime", 100) # Delay root relaxation, try feasible first
model.setParam("NonConvex", 2)       # Allow nonconvex parts
model.optimize()

if model.SolCount > 0:
    to_mm = 0.1
    x_vals = [x[c].x * to_mm for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    y_vals = [y[c].x  * to_mm for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    selected_fixture_type = [FIXTURE_TYPE[c] for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    selected_fixture_vals = [int(selected_fixture[c].x) for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    fixtures_center_x_vals = [fixtures_center_x[c].x * to_mm for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    fixtures_center_y_vals = [fixtures_center_y[c].x * to_mm for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]

    solution = {
        "x": x_vals,
        "y": y_vals,
        "selected_fixture": selected_fixture_vals,
        "fixture_type": selected_fixture_type,
        "fixtures_center_x": fixtures_center_x_vals,
        "fixtures_center_y": fixtures_center_y_vals,
        "x_g": x_g.x * to_mm,
        "y_g": y_g.x * to_mm,
        "objective_value": model.objVal
    }

    file_path = '../resources/results.json'

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as json_file:
        json.dump(solution, json_file, indent=4)

    solution_displayer = ResultDisplayer(workpiece_vertices=[VERTEX_A, VERTEX_B, VERTEX_C, VERTEX_D, VERTEX_E, VERTEX_F, VERTEX_A])
    solution_displayer.show_results(file_path)
else:
    print("No solution found :(")




