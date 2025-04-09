from gurobipy import GRB

class OptimizationCallback:

    def __init__(self, threshold=5):
        self.previous_best_objective = None
        self.threshold = threshold

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            best_objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)

            if self.previous_best_objective is None:
                print(f"First solution found with objective: {best_objective}")
            else:
                print(f"{self.previous_best_objective} -> {best_objective}")
                improvement = self.previous_best_objective - best_objective
                print(f"New solution found; improved by {abs(improvement)}")
                if improvement != 0 and abs(improvement) < self.threshold:
                   print(f"Improvement {abs(improvement)} below threshold {self.threshold}, stopping optimization.")
                   model.terminate()

            self.previous_best_objective = best_objective

