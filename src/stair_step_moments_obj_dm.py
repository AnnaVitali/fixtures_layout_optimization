import gurobipy as gp
from gurobipy import GRB

from optimization.min_max_bounds import MinMaxBounds
from utility.geometry_utility import line_equation
from utility.fixtures_utiility import SQUARE_CUP_DIM, RECTANGULAR_CUP_DIM_X, RECTANGULAR_CUP_DIM_Y
from utility.result_displayer import ResultDisplayer
import os
import json

MM_TO_DM = 0.01
MM2_TO_DM2 = 0.0001

WIDTH = 698 * MM_TO_DM
HEIGHT = 380 * MM_TO_DM
AREA = 175769 * MM2_TO_DM2

VERTEX_A = (665 * MM_TO_DM, 394 * MM_TO_DM)
VERTEX_B = (20 * MM_TO_DM, 262 * MM_TO_DM)
VERTEX_C = (20 * MM_TO_DM, 135 * MM_TO_DM)
VERTEX_D = (665 * MM_TO_DM, 4 * MM_TO_DM)
VERTEX_E = (689 * MM_TO_DM, 17 * MM_TO_DM)
VERTEX_F = (689 * MM_TO_DM, 380 * MM_TO_DM)

BAR_SIZE = 145 * MM_TO_DM

FIXTURE_A_NUMBER = 5
FIXTURE_A_DIMX = SQUARE_CUP_DIM * MM_TO_DM
FIXTURE_A_DIMY = SQUARE_CUP_DIM * MM_TO_DM

FIXTURE_B_NUMBER = 5
FIXTURE_B_DIMX = RECTANGULAR_CUP_DIM_X * MM_TO_DM
FIXTURE_B_DIMY = RECTANGULAR_CUP_DIM_Y * MM_TO_DM

N_FIXTURES_TYPE = 2
FIXTURE_TYPE = [1] * FIXTURE_A_NUMBER + [2] * FIXTURE_B_NUMBER
FIXTURE_AREA = [FIXTURE_A_DIMX * FIXTURE_A_DIMY] * FIXTURE_A_NUMBER + [
    FIXTURE_B_DIMX * FIXTURE_B_DIMY
] * FIXTURE_B_NUMBER

FIXTURES_NUMBER = FIXTURE_A_NUMBER + FIXTURE_B_NUMBER

DIMS = [[FIXTURE_A_DIMX, FIXTURE_A_DIMY], [FIXTURE_B_DIMX, FIXTURE_B_DIMY]]
HALF_DIM = [[FIXTURE_A_DIMX / 2, FIXTURE_A_DIMY / 2], [FIXTURE_B_DIMX / 2, FIXTURE_B_DIMY / 2]]

EPS = 0.1 * MM_TO_DM
M = (WIDTH * HEIGHT) ** 2
MAX_F = int(AREA / (145 * 145 * 2 * MM2_TO_DM2))

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
x = model.addVars(MAX_F, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x")
x2 = model.addVars(MAX_F, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x2")
x3 = model.addVars(MAX_F, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x3")
x4 = model.addVars(MAX_F, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x4")

y = model.addVars(MAX_F, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y")
y2 = model.addVars(MAX_F, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y2")
y3 = model.addVars(MAX_F, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y3")
y4 = model.addVars(MAX_F, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y4")

selected_fixture = model.addVars(MAX_F, N_FIXTURES_TYPE, vtype=GRB.BINARY, name="selected_fixture")
fixture_type = model.addVars(MAX_F, lb=0, ub=N_FIXTURES_TYPE, vtype=GRB.CONTINUOUS, name="fixture_type")
pair_selected = model.addVars(MAX_F, MAX_F, vtype=GRB.BINARY, name="x_selected")

no_overlap = model.addVars(MAX_F, MAX_F, 4, vtype=GRB.BINARY, name="no_overlap")

x1_greater_x2 = model.addVars(MAX_F, MAX_F, vtype=GRB.BINARY, name=f"x1_greater_x2")
x2_greater_x1 = model.addVars(MAX_F, MAX_F, vtype=GRB.BINARY, name=f"x2_greater_x1")

fixtures_center_x = model.addVars(MAX_F, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="fixtures_center_x")
fixtures_center_y = model.addVars(MAX_F, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="fixtures_center_y")

weighted_cx_sum = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="weighted_cx_sum")
weighted_cy_sum = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="weighted_cy_sum")
x_g = model.addVar(lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x_g")
y_g = model.addVar(lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y_g")

q = model.addVars(MAX_F, vtype=GRB.BINARY, name="q")
fixture_dims_x = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMX, vtype=GRB.CONTINUOUS, name="fixture_type")
fixture_dims_y = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMY, vtype=GRB.CONTINUOUS, name="fixture_type")
fixture_half_dims_x = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMX / 2, vtype=GRB.CONTINUOUS, name="half_dims_x")
fixture_half_dims_y = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMY / 2, vtype=GRB.CONTINUOUS, name="half_dims_y")

x_square = model.addVars(MAX_F, lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x1_square")
x2_square = model.addVars(MAX_F, lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x2_square")
x3_square = model.addVars(MAX_F, lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x3_square")
x4_square = model.addVars(MAX_F, lb=0, ub=WIDTH * WIDTH, vtype=GRB.CONTINUOUS, name="x4_square")

y_square = model.addVars(MAX_F, lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y1_square")
y2_square = model.addVars(MAX_F, lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y2_square")
y3_square = model.addVars(MAX_F, lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y3_square")
y4_square = model.addVars(MAX_F, lb=0, ub=HEIGHT * HEIGHT, vtype=GRB.CONTINUOUS, name="y4_square")

xy_bl = model.addVars(MAX_F, lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="xy_bl")
xy_br = model.addVars(MAX_F, lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="xy_br")
xy_tl = model.addVars(MAX_F, lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="xy_tl")
xy_tr = model.addVars(MAX_F, lb=0, ub=WIDTH * HEIGHT, vtype=GRB.CONTINUOUS, name="xy_tr")

max_jx, max_jy, max_jxy = bounds.get_max_moments_fixtures()
jx_fixture = model.addVars(MAX_F, lb=0, ub=max_jx, vtype=GRB.CONTINUOUS, name="jx_fixture")
jy_fixture = model.addVars(MAX_F, lb=0, ub=max_jy, vtype=GRB.CONTINUOUS, name="jy_fixture")
jxy_fixture = model.addVars(MAX_F, lb=0, ub=max_jxy, vtype=GRB.CONTINUOUS, name="jxy_fixture")

max_jx_terms, max_jy_terms, max_jxy_terms = bounds.get_max_multiplication_of_terms()
min_jx_terms, min_jy_terms, min_jxy_terms  = bounds.get_min_multiplication_of_terms()
term1_jx = model.addVars(MAX_F, lb=min_jx_terms[0] if min_jx_terms[0] < 0 else 0,
                         ub=max_jx_terms[0] if max_jx_terms[0] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term1_jx")
term2_jx = model.addVars(MAX_F,  lb=max_jx_terms[1] if max_jx_terms[1] < 0 else 0,
                         ub=min_jx_terms[1] if min_jx_terms[1] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term2_jx")
term3_jx = model.addVars(MAX_F,  lb=max_jx_terms[2] if max_jx_terms[2] < 0 else 0,
                         ub=min_jx_terms[2] if min_jx_terms[2] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term3_jx")
term4_jx = model.addVars(MAX_F,  lb=min_jx_terms[0] if min_jx_terms[0] < 0 else 0,
                         ub=max_jx_terms[3] if max_jx_terms[3] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term4_jx")

term1_jy = model.addVars(MAX_F, lb=min_jy_terms[0] if min_jy_terms[0] < 0 else 0,
                         ub=max_jy_terms[0] if max_jy_terms[0] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term1_jy")
term2_jy = model.addVars(MAX_F, lb=max_jy_terms[1] if max_jy_terms[1] < 0 else 0,
                         ub=min_jy_terms[1] if min_jy_terms[1] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term2_jy")
term3_jy = model.addVars(MAX_F, lb=max_jy_terms[2] if max_jy_terms[2] < 0 else 0,
                         ub=min_jy_terms[2] if min_jy_terms[2] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term3_jy")
term4_jy = model.addVars(MAX_F, lb=min_jy_terms[3] if min_jy_terms[3] < 0 else 0,
                         ub=max_jy_terms[3] if max_jy_terms[3] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term4_jy")

term1_jxy = model.addVars(MAX_F, lb=min_jxy_terms[0] if min_jxy_terms[0] < 0 else 0,
                         ub=max_jxy_terms[0] if max_jxy_terms[0] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term1_jxy")
term2_jxy = model.addVars(MAX_F, lb=max_jxy_terms[1] if max_jxy_terms[1] < 0 else 0,
                         ub=min_jxy_terms[1] if min_jxy_terms[1] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term2_jxy")
term3_jxy = model.addVars(MAX_F, lb=max_jxy_terms[2] if max_jxy_terms[2] < 0 else 0,
                         ub=min_jxy_terms[2] if min_jxy_terms[2] > 0 else 0, vtype=GRB.CONTINUOUS, name=f"term3_jxy")
term4_jxy = model.addVars(MAX_F, lb=min_jxy_terms[3] if min_jxy_terms[3] < 0 else 0,
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

model.addConstrs(q[c] == gp.quicksum(selected_fixture[c,t] for t in range(N_FIXTURES_TYPE)) for c in range(MAX_F))

model.addConstrs(
    (selected_fixture[c,t] == 1) >> (fixture_type[c] == t + 1)
    for c in range(MAX_F)
    for t in range(N_FIXTURES_TYPE)
)

model.addConstrs(
    (selected_fixture[c,t] == 1) >> (fixture_dims_x[c] == DIMS[t][0])
    for c in range(MAX_F)
    for t in range(N_FIXTURES_TYPE)
)
model.addConstrs(
    (selected_fixture[c,t] == 1) >> (fixture_dims_y[c] == DIMS[t][1])
    for c in range(MAX_F)
    for t in range(N_FIXTURES_TYPE)
)
model.addConstrs(
    (selected_fixture[c,t] == 1) >> (fixture_half_dims_x[c] == HALF_DIM[t][0])
    for c in range(MAX_F)
    for t in range(N_FIXTURES_TYPE)
)
model.addConstrs(
    (selected_fixture[c,t] == 1) >> (fixture_half_dims_y[c] == HALF_DIM[t][1])
    for c in range(MAX_F)
    for t in range(N_FIXTURES_TYPE)
)

model.addConstrs(gp.quicksum(selected_fixture[c, t] for t in range(N_FIXTURES_TYPE)) <= 1 for c in range(MAX_F))
model.addConstr(gp.quicksum(selected_fixture[c, 0] for c in range(MAX_F)) <= FIXTURE_A_NUMBER)
model.addConstr(gp.quicksum(selected_fixture[c, 1] for c in range(MAX_F)) <= FIXTURE_B_NUMBER)

#Geometry constraints
# fit in workpiece area
model.addConstrs((x[c] + fixture_dims_x[c]) * selected_fixture[c, t] <= WIDTH for c in range(MAX_F) for t in range(N_FIXTURES_TYPE))
model.addConstrs((y[c] + fixture_dims_y[c]) * selected_fixture[c, t] <= WIDTH for c in range(MAX_F) for t in range(N_FIXTURES_TYPE))

model.addConstrs(x[c] >= (q_stay_left[l] + EPS) * selected_fixture[c,t] for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in
                 range(n_stay_left_line))
model.addConstrs(
    (x[c] + fixture_dims_x[c] + EPS) * selected_fixture[c, t] <= q_stay_right[l] for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in
    range(n_stay_right_line))

model.addConstrs(
    (y[c] - EPS) * selected_fixture[c,t] >= (m_stay_above[l] * x[c] + q_stay_above[l]) * selected_fixture[c,t]
    for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in range(n_stay_above_line))
model.addConstrs((y[c] - EPS) * selected_fixture[c,t] >= (m_stay_above[l] * (x[c] + fixture_dims_x[c]) + q_stay_above[l]) *
                 selected_fixture[c,t]
                 for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in range(n_stay_above_line))
model.addConstrs(((y[c] + fixture_dims_y[c]) - EPS) * selected_fixture[c,t] >= (m_stay_above[l] * x[c] + q_stay_above[l]) *
                 selected_fixture[c,t]
                 for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in range(n_stay_above_line))
model.addConstrs(
    ((y[c] + fixture_dims_y[c]) - EPS) * selected_fixture[c,t] >= (m_stay_above[l] * (x[c] + fixture_dims_x[c]) + q_stay_above[l]) *
    selected_fixture[
        c,t]
    for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in range(n_stay_above_line))

model.addConstrs(
    (y[c] + EPS) * selected_fixture[c,t] <= (m_stay_below[l] * x[c] + q_stay_below[l]) * selected_fixture[c,t]
    for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in range(n_stay_below_line))
model.addConstrs((y[c] + EPS) * selected_fixture[c,t] <= (m_stay_below[l] * (x[c] + fixture_dims_x[c]) + q_stay_below[l]) *
                 selected_fixture[c,t]
                 for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in range(n_stay_below_line))
model.addConstrs(((y[c] + fixture_dims_y[c]) + EPS) * selected_fixture[c,t] <= (m_stay_below[l] * x[c] + q_stay_below[l]) *
                 selected_fixture[c,t]
                 for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in range(n_stay_below_line))
model.addConstrs(
    ((y[c] + fixture_dims_y[c]) + EPS) * selected_fixture[c,t] <= (m_stay_below[l] * (x[c] + fixture_dims_x[c]) + q_stay_below[l]) *
    selected_fixture[
        c,t]
    for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) for l in range(n_stay_below_line))

# no overlap between fixtures
model.addConstrs(
    pair_selected[c1, c2] <= gp.quicksum(selected_fixture[c1,t] for t in range(N_FIXTURES_TYPE)) for c1 in range(MAX_F) for c2 in range(MAX_F)
    if c1 != c2)
model.addConstrs(
    pair_selected[c1, c2] <= gp.quicksum(selected_fixture[c2,t] for t in range(N_FIXTURES_TYPE)) for c1 in range(MAX_F) for c2 in range(MAX_F)
    if c1 != c2)
model.addConstrs(
    pair_selected[c1, c2] >= gp.quicksum(selected_fixture[c1,t] + selected_fixture[c2,t] for t in range(N_FIXTURES_TYPE)) - 1 for c1 in range(MAX_F) for c2
    in range(MAX_F) if c1 != c2)

model.addConstrs((x[c1] + fixture_dims_x[c1]) * pair_selected[c1, c2] <= x[c2] + M * (1 - no_overlap[c1, c2, 0]) for c1 in
                 range(MAX_F) for c2 in range(c1 + 1, MAX_F))
model.addConstrs((x[c2] + fixture_dims_x[c2]) * pair_selected[c1, c2] <= x[c1] + M * (1 - no_overlap[c1, c2, 1]) for c1 in
                 range(MAX_F) for c2 in range(c1 + 1, MAX_F))
model.addConstrs((y[c1] + fixture_dims_y[c1]) * pair_selected[c1, c2] <= y[c2] + M * (1 - no_overlap[c1, c2, 2]) for c1 in
                 range(MAX_F) for c2 in range(c1 + 1, MAX_F))
model.addConstrs((y[c2] + fixture_dims_y[c2]) * pair_selected[c1, c2] <= y[c1] + M * (1 - no_overlap[c1, c2, 3]) for c1 in
                 range(MAX_F) for c2 in range(c1 + 1, MAX_F))
#
model.addConstrs(
    gp.quicksum(no_overlap[c1, c2, i] for i in range(4)) >= 1 for c1 in range(MAX_F) for c2 in
    range(c1 + 1, MAX_F))

# respect bar distance
model.addConstrs(
    x[c1] * pair_selected[c1, c2] >= (x[c2] + EPS - M * (1 - x1_greater_x2[c1, c2])) * pair_selected[c1, c2]
    for c1 in range(MAX_F) for c2 in range(MAX_F) if c1 != c2)
model.addConstrs(x[c1] * pair_selected[c1, c2] <= (x[c2] + M * x1_greater_x2[c1, c2]) * pair_selected[c1, c2]
                 for c1 in range(MAX_F) for c2 in range(MAX_F) if c1 != c2)
model.addConstrs(
    x[c2] * pair_selected[c1, c2] >= (x[c1] + EPS - M * (1 - x2_greater_x1[c1, c2])) * pair_selected[c1, c2]
    for c1 in range(MAX_F) for c2 in range(MAX_F) if c1 != c2)
model.addConstrs(x[c2] * pair_selected[c1, c2] <= (x[c1] + M * x2_greater_x1[c1, c2]) * pair_selected[c1, c2]
                 for c1 in range(MAX_F) for c2 in range(MAX_F) if c1 != c2)

model.addConstrs(
    (x[c2] + fixture_dims_x[c2] + BAR_SIZE) * pair_selected[c1, c2] <= (x[c1] + M * (1 - x1_greater_x2[c1, c2])) *
    pair_selected[c1, c2]
    for c1 in range(MAX_F) for c2 in range(MAX_F) if c1 != c2)
model.addConstrs(
    (x[c1] + fixture_dims_x[c1] + BAR_SIZE) * pair_selected[c1, c2] <= (x[c2] + M * (1 - x2_greater_x1[c1, c2])) *
    pair_selected[c1, c2]
    for c1 in range(MAX_F) for c2 in range(MAX_F) if c1 != c2)

# symmetry breaking constraints
model.addConstr(gp.quicksum(selected_fixture[c, t] for c in range(MAX_F) for t in range(N_FIXTURES_TYPE)) >= 2, name="at_least_one_fixture")


model.addConstrs((q[c] == 0) >> (x[c] <= 0) for c in range(MAX_F))
model.addConstrs((q[c] == 0) >> (y[c] <= 0) for c in range(MAX_F))

model.addConstrs(
    (x[c1] + y[c1]) + EPS >= (x[c2] + y[c2]) for c1 in range(MAX_F) for c2 in
    range(c1 + 1, MAX_F) if
    FIXTURE_TYPE[c1] == FIXTURE_TYPE[c2])

#Objective function
# #compute the center of the fixtures
model.addConstrs(
    fixtures_center_x[c] == (x[c] + fixture_half_dims_x[c]) * q[c] for c in range(MAX_F))
model.addConstrs(
    fixtures_center_y[c] == (y[c] + fixture_half_dims_y[c]) * q[c] for c in range(MAX_F))

#compute the weighted sum for the objective
area_total = model.addVar(lb=0, ub=sum(FIXTURE_AREA), vtype=GRB.CONTINUOUS, name="area_total")
area_fixtures = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMX * FIXTURE_A_DIMY, vtype=GRB.CONTINUOUS, name="area_fixtures")
area_selected = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMX * FIXTURE_A_DIMY, vtype=GRB.CONTINUOUS, name="area_mul_center")

model.addConstrs(area_selected[c] == area_fixtures[c] * q[c] for c in range(MAX_F))

model.addConstr(
    weighted_cx_sum == gp.quicksum(
        area_selected[c] * fixtures_center_x[c] for c in range(MAX_F)),
    name="weighted_cx_sum")
model.addConstr(
    weighted_cy_sum == gp.quicksum(
        area_selected[c] * fixtures_center_y[c] for c in range(MAX_F)),
    name="weighted_cy_sum")

model.addConstrs(area_fixtures[c] == fixture_dims_x[c] * fixture_dims_y[c] for c in range(MAX_F))
model.addConstr(area_total == gp.quicksum(area_fixtures[c] * q[c] for c in range(MAX_F)),
                name="area_total")

model.addConstr(x_g * area_total == weighted_cx_sum, name="x_g")
model.addConstr(y_g * area_total == weighted_cy_sum, name="y_g")


#compute fixtures vertices
model.addConstrs(x2[c] == x[c] * q[c] for c in range(MAX_F))
model.addConstrs(x3[c] == (x[c] + fixture_dims_x[c]) * q[c] for c in range(MAX_F))
model.addConstrs(x4[c] == (x[c] + fixture_dims_x[c]) * q[c] for c in range(MAX_F))

model.addConstrs(y2[c] == (y[c] + fixture_dims_y[c]) * q[c] for c in range(MAX_F))
model.addConstrs(y3[c] == (y[c] + fixture_dims_y[c]) * q[c] for c in range(MAX_F))
model.addConstrs(y4[c] == y[c] * q[c] for c in range(MAX_F))

for c in range(MAX_F):
    model.addGenConstrPow(y[c], y_square[c], 2)
    model.addGenConstrPow(y2[c], y2_square[c], 2)
    model.addGenConstrPow(y3[c], y3_square[c], 2)
    model.addGenConstrPow(y4[c], y4_square[c], 2)

    model.addGenConstrPow(x[c], x_square[c], 2)
    model.addGenConstrPow(x2[c], x2_square[c], 2)
    model.addGenConstrPow(x3[c], x3_square[c], 2)
    model.addGenConstrPow(x4[c], x4_square[c], 2)

#moments of inertia of fixtures
model.addConstrs(xy_bl[c] == x[c] * y[c] for c in range(MAX_F))
model.addConstrs(xy_tl[c] == x2[c] * y2[c] for c in range(MAX_F))
model.addConstrs(xy_tr[c] == x3[c] * y3[c] for c in range(MAX_F))
model.addConstrs(xy_br[c] == x4[c] * y4[c] for c in range(MAX_F))

max_jx, max_jy, max_jxy = bounds.get_max_moments_fixtures()

min_common_terms = bounds.get_min_common_terms()
max_common_terms = bounds.get_max_common_terms()

max_jx_terms_part1 = bounds.get_max_jx_terms()
min_jx_terms_part1 = bounds.get_min_jx_terms()

max_jy_terms_part1 = bounds.get_max_jy_terms()
min_jy_terms_part1 = bounds.get_min_jy_terms()

max_jxy_terms_part1 = bounds.get_max_jxy_terms()
min_jxy_terms_part1 = bounds.get_min_jxy_terms()

print("PRINTTTTTTTTTTTT")
print(max_jxy)

for c in range(MAX_F):
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
model.addConstr(area_total == gp.quicksum(FIXTURE_AREA[c] * q[c] for c in range(MAX_F)), name="area_total")

model.addConstr(jx_g == gp.quicksum(jx_fixture[c] * q[c] for c in range(MAX_F)), name="jx_g")
model.addConstr(jy_g == gp.quicksum(jy_fixture[c] * q[c] for c in range(MAX_F)), name="jy_g")
model.addConstr(jxy_g == gp.quicksum(jxy_fixture[c] * q[c] for c in range(MAX_F)), name="jxy_g")

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

model.addConstr(root == sub_jx_jy_squared + 4 * jxy_squared, name="root")
model.addGenConstrPow(square_of_root, root, 2.0, name="square_of_root")

model.addGenConstrNL(i, 0.5 * (jx + jy) - 0.5 * square_of_root, name="i_equation")
model.addGenConstrNL(j, 0.5 * (jx + jy) + 0.5 * square_of_root, name="j_equation")



model.setObjective(i + j, GRB.MAXIMIZE)
model.optimize()

DM_TO_MM_CONVERSION = 100

if model.SolCount > 0:
    x_vals_DM = [x[c].x for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if selected_fixture[c, t].x == 1]
    y_vals_DM = [y[c].x for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if selected_fixture[c, t].x == 1]

    x_vals_mm = [val * DM_TO_MM_CONVERSION for val in x_vals_DM]
    y_vals_mm = [val * DM_TO_MM_CONVERSION for val in y_vals_DM]

    selected_fixture_type = [
        int(fixture_type[c].x)
        for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if selected_fixture[c, t].x == 1
    ]
    selected_fixture_vals = [
        int(selected_fixture[c, t].x) for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if
        selected_fixture[c, t].x == 1
    ]

    fixtures_center_x_vals_DM = [fixtures_center_x[c].x for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if
                                 selected_fixture[c, t].x == 1]
    fixtures_center_y_vals_DM = [fixtures_center_y[c].x for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if
                                 selected_fixture[c, t].x == 1]

    fixtures_center_x_vals_mm = [val * DM_TO_MM_CONVERSION for val in fixtures_center_x_vals_DM]
    fixtures_center_y_vals_mm = [val * DM_TO_MM_CONVERSION for val in fixtures_center_y_vals_DM]

    solution = {
        "x": x_vals_mm,
        "y": y_vals_mm,
        "selected_fixture": selected_fixture_vals,
        "fixture_type": selected_fixture_type,
        "fixtures_center_x": fixtures_center_x_vals_mm,
        "fixtures_center_y": fixtures_center_y_vals_mm,
        "x_g": x_g.x * DM_TO_MM_CONVERSION,
        "y_g": y_g.x * DM_TO_MM_CONVERSION,
        "objective_value": model.objVal
    }

    file_path = '../resources/results_moments_obj.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as json_file:
        json.dump(solution, json_file, indent=4)

    vertices = [VERTEX_A, VERTEX_B, VERTEX_C, VERTEX_D, VERTEX_E , VERTEX_F, VERTEX_A]

    solution_displayer = ResultDisplayer(
        workpiece_vertices=[(x * DM_TO_MM_CONVERSION, y * DM_TO_MM_CONVERSION) for x, y in vertices])
    solution_displayer.show_results(file_path)

else:
    print("No optimal solution found :(")




