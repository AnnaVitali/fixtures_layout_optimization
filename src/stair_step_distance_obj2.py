import gurobipy as gp
from gurobipy import GRB
from utility.geometry_utility import line_equation
from utility.result_displayer import ResultDisplayer
from optimization.optimization_callback import OptimizationCallback
import os
import json

WIDTH = 698
HEIGHT = 380
AREA = 175769
VERTEX_A = (665, 394)
VERTEX_B = (20, 262)
VERTEX_C = (20, 135)
VERTEX_D = (665, 4)
VERTEX_E = (689, 17)
VERTEX_F = (689, 380)

BAR_SIZE = 145

FIXTURE_A_NUMBER = 5
FIXTURE_A_DIMX = 145
FIXTURE_A_DIMY = 145

FIXTURE_B_NUMBER = 5
FIXTURE_B_DIMX = 145
FIXTURE_B_DIMY = 55

N_FIXTURES_TYPE = 2
FIXTURE_TYPE = [1] * FIXTURE_A_NUMBER + [2] * FIXTURE_B_NUMBER
FIXTURE_AREA = [FIXTURE_A_DIMX * FIXTURE_A_DIMY] * FIXTURE_A_NUMBER + [
    FIXTURE_B_DIMX * FIXTURE_B_DIMY] * FIXTURE_B_NUMBER

FIXTURES_NUMBER = FIXTURE_A_NUMBER + FIXTURE_B_NUMBER

DIMS = [[FIXTURE_A_DIMX, FIXTURE_A_DIMY]] * FIXTURE_A_NUMBER + [[FIXTURE_B_DIMX, FIXTURE_B_DIMY]] * FIXTURE_B_NUMBER
HALF_DIM = [[FIXTURE_A_DIMX / 2, FIXTURE_A_DIMY / 2]] * FIXTURE_A_NUMBER + [
    [FIXTURE_B_DIMX / 2, FIXTURE_B_DIMY / 2]] * FIXTURE_B_NUMBER

EPS = 1
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

#Variables
x = model.addVars(FIXTURES_NUMBER, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(FIXTURES_NUMBER, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y")

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

diff_cx = model.addVars(FIXTURES_NUMBER, lb=-WIDTH, vtype=GRB.CONTINUOUS, name="cx_diff")
diff_cy = model.addVars(FIXTURES_NUMBER, lb=-HEIGHT, vtype=GRB.CONTINUOUS, name="cy_diff")
cx_dist = model.addVars(FIXTURES_NUMBER, vtype=GRB.CONTINUOUS, name="cx_dist")
cy_dist = model.addVars(FIXTURES_NUMBER, vtype=GRB.CONTINUOUS, name="cy_dist")

#Geometry constraints
# fit in workpiece area
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
model.addConstrs(((y[c] + DIMS[c][1]) - EPS) * selected_fixture[c] >= (m_stay_above[l] * x[c] + q_stay_above[l]) *
                 selected_fixture[c]
                 for c in range(FIXTURES_NUMBER) for l in range(n_stay_above_line))
model.addConstrs(
    ((y[c] + DIMS[c][1]) - EPS) * selected_fixture[c] >= (m_stay_above[l] * (x[c] + DIMS[c][0]) + q_stay_above[l]) *
    selected_fixture[
        c]
    for c in range(FIXTURES_NUMBER) for l in range(n_stay_above_line))

model.addConstrs(
    (y[c] + EPS) * selected_fixture[c] <= (m_stay_below[l] * x[c] + q_stay_below[l]) * selected_fixture[c]
    for c in range(FIXTURES_NUMBER) for l in range(n_stay_below_line))
model.addConstrs((y[c] + EPS) * selected_fixture[c] <= (m_stay_below[l] * (x[c] + DIMS[c][0]) + q_stay_below[l]) *
                 selected_fixture[c]
                 for c in range(FIXTURES_NUMBER) for l in range(n_stay_below_line))
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
#compute the center of the fixtures
model.addConstrs(
    fixtures_center_x[c] == (x[c] + HALF_DIM[c][0]) * selected_fixture[c] for c in range(FIXTURES_NUMBER))
model.addConstrs(
    fixtures_center_y[c] == (y[c] + HALF_DIM[c][1]) * selected_fixture[c] for c in range(FIXTURES_NUMBER))

#compute the weighted sum for the objective
model.addConstr(
    weighted_cx_sum == gp.quicksum(
        FIXTURE_AREA[c] * fixtures_center_x[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)),
    name="weighted_cx_sum")
model.addConstr(
    weighted_cy_sum == gp.quicksum(
        FIXTURE_AREA[c] * fixtures_center_y[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)),
    name="weighted_cy_sum")

area_total = model.addVar(lb=0, ub=sum(FIXTURE_AREA), vtype=GRB.CONTINUOUS, name="area_total")
model.addConstr(area_total == gp.quicksum(FIXTURE_AREA[c] * selected_fixture[c] for c in range(FIXTURES_NUMBER)),
                name="area_total")

model.addConstr(x_g * area_total == weighted_cx_sum, name="x_g")
model.addConstr(y_g * area_total == weighted_cy_sum, name="y_g")

#compute the distances from fixture center to the overall center of gravity
for c in range(FIXTURES_NUMBER):
    model.addConstr(diff_cx[c] == (x_g - fixtures_center_x[c]) * selected_fixture[c], name=f"diff_cx_{c}")
    model.addGenConstrAbs(cx_dist[c], diff_cx[c], name=f"abs_diff_cx_{c}")

    model.addConstr(diff_cy[c] == (y_g - fixtures_center_y[c]) * selected_fixture[c], name=f"diff_cy_{c}")
    model.addGenConstrAbs(cy_dist[c], diff_cy[c], name=f"abs_diff_cy_{c}")

model.setObjective(
    gp.quicksum(cx_dist[c] for c in range(FIXTURES_NUMBER)) + gp.quicksum(
        cy_dist[c] for c in range(FIXTURES_NUMBER)), GRB.MAXIMIZE)



optimization_callback = OptimizationCallback(threshold=5)
model.optimize(optimization_callback)

if model.SolCount > 0:
    x_vals = [x[c].x for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    y_vals = [y[c].x for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    selected_fixture_type = [FIXTURE_TYPE[c] for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    selected_fixture_vals = [int(selected_fixture[c].x) for c in range(FIXTURES_NUMBER) if
                             selected_fixture[c].x == 1]
    fixtures_center_x_vals = [fixtures_center_x[c].x for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]
    fixtures_center_y_vals = [fixtures_center_y[c].x for c in range(FIXTURES_NUMBER) if selected_fixture[c].x == 1]

    solution = {
        "x": x_vals,
        "y": y_vals,
        "selected_fixture": selected_fixture_vals,
        "fixture_type": selected_fixture_type,
        "fixtures_center_x": fixtures_center_x_vals,
        "fixtures_center_y": fixtures_center_y_vals,
        "x_g": x_g.x,
        "y_g": y_g.x,
        "objective_value": model.objVal
    }

    file_path = '../resources/results.json'

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as json_file:
        json.dump(solution, json_file, indent=4)

    solution_displayer = ResultDisplayer(workpiece_vertices=[VERTEX_A, VERTEX_B, VERTEX_C, VERTEX_D, VERTEX_E, VERTEX_F, VERTEX_A])
    solution_displayer.show_results(file_path)
else:
    print("No optimal solution found :(")

