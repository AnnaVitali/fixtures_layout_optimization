from utility.result_displayer import ResultDisplayer

MM_TO_DM = 0.01
MM2_TO_DM2 = 0.0001

# Geometry dimensions (converted to DM)
WIDTH = 698 * MM_TO_DM
HEIGHT = 380 * MM_TO_DM
AREA = 175769 * MM2_TO_DM2

VERTEX_A = (665 * MM_TO_DM, 394 * MM_TO_DM)
VERTEX_B = (20 * MM_TO_DM, 262 * MM_TO_DM)
VERTEX_C = (20 * MM_TO_DM, 135 * MM_TO_DM)
VERTEX_D = (665 * MM_TO_DM, 4 * MM_TO_DM)
VERTEX_E = (689 * MM_TO_DM, 17 * MM_TO_DM)
VERTEX_F = (689 * MM_TO_DM, 380 * MM_TO_DM)

file_path = '../resources/results_moments_obj_mzn.json'

DM_TO_MM_CONVERSION = 100
vertices = [VERTEX_A, VERTEX_B, VERTEX_C, VERTEX_D, VERTEX_E, VERTEX_F, VERTEX_A]

# Display the results
solution_displayer = ResultDisplayer(
    workpiece_vertices=[(x * DM_TO_MM_CONVERSION, y * DM_TO_MM_CONVERSION) for x, y in vertices])
solution_displayer.show_results(file_path)