import gurobipy as gp
from gurobipy import GRB
from utility.geometry_utility import line_equation
from utility.fixtures_utiility import SQUARE_CUP_DIM, RECTANGULAR_CUP_DIM_X, RECTANGULAR_CUP_DIM_Y
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
FIXTURE_A_DIMX = SQUARE_CUP_DIM
FIXTURE_A_DIMY = SQUARE_CUP_DIM

FIXTURE_B_NUMBER = 5
FIXTURE_B_DIMX = RECTANGULAR_CUP_DIM_X
FIXTURE_B_DIMY = RECTANGULAR_CUP_DIM_Y

N_FIXTURES_TYPE = 2
FIXTURE_TYPE = [1] * FIXTURE_A_NUMBER + [2] * FIXTURE_B_NUMBER
FIXTURE_AREA = [FIXTURE_A_DIMX * FIXTURE_A_DIMY] * FIXTURE_A_NUMBER + [
    FIXTURE_B_DIMX * FIXTURE_B_DIMY] * FIXTURE_B_NUMBER

FIXTURES_NUMBER = FIXTURE_A_NUMBER + FIXTURE_B_NUMBER

DIMS = [[FIXTURE_A_DIMX, FIXTURE_A_DIMY], [FIXTURE_B_DIMX, FIXTURE_B_DIMY]]
HALF_DIM = [[FIXTURE_A_DIMX / 2, FIXTURE_A_DIMY / 2], [FIXTURE_B_DIMX / 2, FIXTURE_B_DIMY / 2]]

EPS = 1
M = (WIDTH * HEIGHT) ** 2
MAX_F = int(AREA / (21025 * 2)) # numero vertici

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
x = model.addVars(MAX_F, lb=0, ub=WIDTH, vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(MAX_F, lb=0, ub=HEIGHT, vtype=GRB.CONTINUOUS, name="y")

selected_fixture = model.addVars(MAX_F, N_FIXTURES_TYPE, vtype=GRB.BINARY, name="selected_fixture")
#fixture_number = model.addVars(lb=2, ub=MAX_F, vtype=GRB.CONTINUOUS, name="fixture_number")

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

diff_cx = model.addVars(MAX_F, lb=-WIDTH, vtype=GRB.CONTINUOUS, name="cx_diff")
diff_cy = model.addVars(MAX_F, lb=-HEIGHT, vtype=GRB.CONTINUOUS, name="cy_diff")
cx_dist = model.addVars(MAX_F, vtype=GRB.CONTINUOUS, name="cx_dist")
cy_dist = model.addVars(MAX_F, vtype=GRB.CONTINUOUS, name="cy_dist")

q = model.addVars(MAX_F, vtype=GRB.BINARY, name="q")
fixture_dims_x = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMX, vtype=GRB.CONTINUOUS, name="fixture_type")
fixture_dims_y = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMY, vtype=GRB.CONTINUOUS, name="fixture_type")
fixture_half_dims_x = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMX / 2, vtype=GRB.CONTINUOUS, name="half_dims_x")
fixture_half_dims_y = model.addVars(MAX_F, lb=0, ub=FIXTURE_A_DIMY / 2, vtype=GRB.CONTINUOUS, name="half_dims_y")

model.addConstrs(q[c] == gp.quicksum(selected_fixture[c,t] for t in range(N_FIXTURES_TYPE)) for c in range(MAX_F))
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
#compute the center of the fixtures
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

#compute the distances from fixture center to the overall center of gravity
for c in range(MAX_F):
    model.addConstr(diff_cx[c] == (x_g - fixtures_center_x[c]) * q[c], name=f"diff_cx_{c}")
    model.addGenConstrAbs(cx_dist[c], diff_cx[c], name=f"abs_diff_cx_{c}")

    model.addConstr(diff_cy[c] == (y_g - fixtures_center_y[c]) * q[c], name=f"diff_cy_{c}")
    model.addGenConstrAbs(cy_dist[c], diff_cy[c], name=f"abs_diff_cy_{c}")

model.setObjective(
    gp.quicksum(cx_dist[c] for c in range(MAX_F)) + gp.quicksum(
        cy_dist[c] for c in range(MAX_F)), GRB.MAXIMIZE)

model.setParam("Cuts", 2)       # Use aggressive cuts
model.setParam("VarBranch", 1)  # Change variable selection for branching
model.setParam("BranchDir", 1)  # Favor improving bounds
model.setParam("Heuristics", 0.5)  # Increase heuristic search
model.setParam("RINS", 10)      # Use RINS heuristic
#
optimization_callback = OptimizationCallback(threshold=0)

# x = [544.0, 544.0, 544.0, 544.0, 21.0, 21.0];
# y = [86.26001118122852, 310.71000000000004, 255.70999999999998, 31.259999999999998, 206.10999999999981, 135.86000000000004];

model.optimize(optimization_callback)

if model.SolCount > 0:
    #print([round(fixture_dims_y[c].x) for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if selected_fixture[c,t].x == 1])

    x_vals = [x[c].x for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if selected_fixture[c,t].x == 1]
    y_vals = [y[c].x for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if selected_fixture[c,t].x == 1]
    selected_fixture_type = [1 if round(fixture_dims_y[c].x)==FIXTURE_A_DIMY else 2 if round(fixture_dims_y[c].x)==FIXTURE_B_DIMY else 0 for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if
                             selected_fixture[c,t].x == 1]
    selected_fixture_vals = [int(selected_fixture[c,t].x) for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if
                             selected_fixture[c,t].x == 1]
    fixtures_center_x_vals = [fixtures_center_x[c].x for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if selected_fixture[c,t].x == 1]
    fixtures_center_y_vals = [fixtures_center_y[c].x for c in range(MAX_F) for t in range(N_FIXTURES_TYPE) if selected_fixture[c,t].x == 1]

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

    file_path = '../resources/results_distance_obj.json'

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as json_file:
        json.dump(solution, json_file, indent=4)

    solution_displayer = ResultDisplayer(workpiece_vertices=[VERTEX_A, VERTEX_B, VERTEX_C, VERTEX_D, VERTEX_E, VERTEX_F, VERTEX_A])
    solution_displayer.show_results(file_path)
else:
    print("No optimal solution found :(")

