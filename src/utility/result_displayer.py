import matplotlib.pyplot as plt
from src.inertia_analysis.moments_of_inerta import InertiaAnalysis
import math
import json


class ResultDisplayer:
    def __init__(self, workpiece_vertices):
        self.workpiece_vertices = workpiece_vertices

    def show_results(self, file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        x = data["x"]
        y = data["y"]
        selected_fixtures = data["selected_fixture"]
        fixture_types = data["fixture_type"]
        fixtures_center_x = data["fixtures_center_x"]
        fixtures_center_y = data["fixtures_center_y"]
        x_g = data["x_g"]
        y_g = data["y_g"]

        polygons = []

        for idx in range(len(fixture_types)):
            if selected_fixtures[idx] == 1:
                fixture_x, fixture_y = self.__create_fixture(fixture_types[idx], x[idx], y[idx])
                fixture_coords = list(zip(fixture_x, fixture_y))
                print(f"fixture coords{fixture_coords}")
                area = InertiaAnalysis.compute_polygon_area(fixture_coords)
                jx, jy, jxy = InertiaAnalysis.compute_absolute_moments_of_inertia(fixture_coords)

                fixture = {
                    "area": 21025 if fixture_types[idx] == 1 else 7975,
                    "absolute_moments_of_inertia": [jx, jy, jxy],
                    "barycenter": (fixtures_center_x[idx], fixtures_center_y[idx]),
                    "angle": 0,
                    "idx": idx
                }

                jxg, jyg, jxyg = InertiaAnalysis.compute_baricentric_moments_of_inertia(fixture)

                fixture["baricentric_moments_of_inertia"] = [jxg, jyg, jxyg]
                polygons.append(fixture)

        # Compute overall values
        j_xg, j_yg = InertiaAnalysis.compute_combined_baricentric_moments_of_inertia(polygons, x_g, y_g)

        # Print results
        print("\n----------------------Areas---------------------\n")
        area_sum = 0
        for i, poly in enumerate(polygons, 1):
            area_sum += poly['area']
            print(f"Area fixture{i}: {poly['area']}")
        print(f"Sum of areas: {area_sum}")

        print("\n----------Absolute Moment Of Inertia jx, jy, jxy fixtures----------\n")
        for i, poly in enumerate(polygons, 1):
            print(f"Moment of Inertia jx{i}: {poly['absolute_moments_of_inertia'][0]:.5e}")
            print(f"Moment of Inertia jy{i}: {poly['absolute_moments_of_inertia'][1]:.5e}")
            print(f"Moment of Inertia jxy{i}: {poly['absolute_moments_of_inertia'][2]:.5e}")

        print("\n----------Baricentric Moment Of Inertia jxg, jyg, jxyg fixtures----------\n")
        for i, poly in enumerate(polygons, 1):
            print(f"Moment of Inertia jxg{i}: {poly['baricentric_moments_of_inertia'][0]:.5e}")
            print(f"Moment of Inertia jyg{i}: {poly['baricentric_moments_of_inertia'][1]:.5e}")
            print(f"Moment of Inertia jxyg{i}: {poly['baricentric_moments_of_inertia'][2]:.5e}")

        print("\n----------Baricentric Moment Of Inertia JxG, JyPG----------\n")
        print(f"Principal Moment of Inertia I: {j_xg:.5e}")
        print(f"Principal Moment of Inertia J: {j_yg:.5e}")
        print(f"Sum of I and J: {j_xg + j_yg:.5e}")

        # Plot
        plt.figure(figsize=(8, 6))
        x_workpiece, y_workpiece = zip(*self.workpiece_vertices )
        plt.plot(x_workpiece, y_workpiece, '-o', label="Polygon", linewidth=2)
        plt.scatter(*zip(*self.workpiece_vertices ), color='red', zorder=5, label="Vertices")

        plt.scatter(x, y, color='blue', label='Lower-left corner coordinates', zorder=5)
        for i, (x_val, y_val) in enumerate(zip(x, y)):
            plt.text(x_val + 5, y_val + 5, f"({self.__truncate(x_val, 3)}, {self.__truncate(y_val, 3)})", color="blue", fontsize=10)

        for idx, fixture in enumerate(polygons):
            fixture_x, fixture_y = self.__create_fixture(fixture_types[fixture["idx"]], x[idx], y[idx])
            plt.plot(fixture_x, fixture_y)
            plt.scatter(*fixture["barycenter"], color='green', zorder=5)
            plt.text(fixture["barycenter"][0] + 5, fixture["barycenter"][1] + 5, f"g{idx + 1}", color="green",
                     fontsize=10)

        plt.scatter(x_g, y_g, color='black', label="Overall center of gravity", zorder=5)
        plt.text(x_g + 5, y_g + 5, f"G ({self.__truncate(x_g, 3)}, {self.__truncate(y_g, 3)})", color="black", fontsize=10)

        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def __square_suction_fixture(self, lb_x, lb_y):
        square_side = 145

        square_coords = [
            (lb_x, lb_y),
            (lb_x, lb_y + square_side),
            (lb_x + square_side, lb_y + square_side),
            (lb_x + square_side, lb_y),
            (lb_x, lb_y),  # Close the square
        ]
        return zip(*square_coords)

    def __rectangle_suction_fixture(self, lb_x, lb_y):
        width = 145
        height = 55

        rectangle_coords = [
            (lb_x, lb_y),
            (lb_x, lb_y + height),
            (lb_x + width, lb_y + height),
            (lb_x + width, lb_y),
            (lb_x, lb_y),  # Close the square
        ]
        return zip(*rectangle_coords)

    def __create_fixture(self, fixture_type, center_x, center_y):
        if fixture_type == 1:
            return list(self.__square_suction_fixture(center_x, center_y))
        elif fixture_type == 2:
            return list(self.__rectangle_suction_fixture(center_x, center_y))
        else:
            raise ValueError("Invalid fixture type: must be 1 (square) or 2 (rectangle)")


    def __truncate(self, f, n):
        return math.floor(f * 10**n) / 10**n



