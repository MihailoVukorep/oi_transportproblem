import numpy as np
import pandas as pd

class TransportationProblem:
    def __init__(self, supply, demand, costs):
        # Check if the problem is balanced
        total_supply = sum(supply)
        total_demand = sum(demand)
        
        # If unbalanced, add a dummy supply or demand
        if total_supply > total_demand:
            demand.append(total_supply - total_demand)
            costs = np.column_stack((costs, np.zeros(len(supply), dtype=int)))
        elif total_demand > total_supply:
            supply.append(total_demand - total_supply)
            costs = np.vstack((costs, np.zeros(len(demand), dtype=int)))
        
        # Update attributes
        self.supply = supply
        self.demand = demand
        self.costs = costs
        self.rows = len(supply)
        self.cols = len(demand)
        self.solution = np.zeros((self.rows, self.cols), dtype=int)

    # Northwest Corner Method
    def northwest_corner(self):
        supply = self.supply.copy()
        demand = self.demand.copy()
        i, j = 0, 0
        
        while i < self.rows and j < self.cols:
            allocation = min(supply[i], demand[j])
            self.solution[i][j] = allocation
            supply[i] -= allocation
            demand[j] -= allocation
            
            if supply[i] == 0:
                i += 1
            elif demand[j] == 0:
                j += 1
        
        return self.solution

    def least_cost(self):
        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy()
        solution = np.zeros((self.rows, self.cols), dtype=int)
        
        while sum(supply) > 0 and sum(demand) > 0:
            # Find the cell with the minimum cost
            min_cost = 9999999
            min_i, min_j = -1, -1
            for i in range(self.rows):
                for j in range(self.cols):
                    if supply[i] > 0 and demand[j] > 0 and costs[i][j] < min_cost:
                        min_cost = costs[i][j]
                        min_i, min_j = i, j
            
            # Allocate to the cell with the minimum cost
            allocation = min(supply[min_i], demand[min_j])
            solution[min_i][min_j] = allocation
            supply[min_i] -= allocation
            demand[min_j] -= allocation
            
            # Mark row or column as used if supply or demand is exhausted
            if supply[min_i] == 0:
                for j in range(self.cols):
                    costs[min_i][j] = 99999999  # mark row as used
            if demand[min_j] == 0:
                for i in range(self.rows):
                    costs[i][min_j] = 99999999  # mark column as used

        self.solution = solution
        return solution

    # Vogel's Approximation Method
    def vogel_approximation(self):
        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy()
        solution = np.zeros((self.rows, self.cols), dtype=int)
        
        while sum(supply) > 0 and sum(demand) > 0:
            # Calculate penalties for each row and column
            penalties = []
            for i in range(self.rows):
                if supply[i] > 0:
                    row_costs = [costs[i][j] for j in range(self.cols) if demand[j] > 0]
                    if len(row_costs) > 1:
                        row_costs.sort()
                        penalties.append((row_costs[1] - row_costs[0], i, 'row'))
                    else:
                        penalties.append((row_costs[0], i, 'row'))
            
            for j in range(self.cols):
                if demand[j] > 0:
                    col_costs = [costs[i][j] for i in range(self.rows) if supply[i] > 0]
                    if len(col_costs) > 1:
                        col_costs.sort()
                        penalties.append((col_costs[1] - col_costs[0], j, 'col'))
                    else:
                        penalties.append((col_costs[0], j, 'col'))

            # Choose the highest penalty
            penalties.sort(reverse=True, key=lambda x: x[0])
            penalty, idx, rc_type = penalties[0]

            # Allocate in the cell with the minimum cost in the row or column
            if rc_type == 'row':
                i = idx
                j = min(range(self.cols), key=lambda col: costs[i][col] if demand[col] > 0 else 99999999)
            else:
                j = idx
                i = min(range(self.rows), key=lambda row: costs[row][j] if supply[row] > 0 else 99999999)
            
            allocation = min(supply[i], demand[j])
            solution[i][j] = allocation
            supply[i] -= allocation
            demand[j] -= allocation
            costs[i][j] = 999999999  # Mark cell as used
            
            # Mark row or column as used if supply or demand is exhausted
            if supply[i] == 0:
                for col in range(self.cols):
                    costs[i][col] = 999999999
            if demand[j] == 0:
                for row in range(self.rows):
                    costs[row][j] = 999999999

        self.solution = solution
        return solution
    
    # Calculate total cost for the current solution
    def calculate_total_cost(self):
            total_cost = 0
            for i in range(self.rows):
                for j in range(self.cols):
                    total_cost += self.solution[i][j] * self.costs[i][j]
            return total_cost

    # Display the solution in a readable format
    def display_solution(self):
        return pd.DataFrame(self.solution, columns=[f"Demand {i+1}" for i in range(self.cols)],
                            index=[f"Supply {i+1}" for i in range(self.rows)])    
# Example usage:
supply = [260, 280]
demand = [220, 200, 80, 180, 160]
costs = np.array([[6, 4, 5, 6, 3], [9, 5, 4, 3, 3]])

tp = TransportationProblem(supply, demand, costs)

# Using Northwest Corner
print("Northwest Corner Method:")
northwest_solution = tp.northwest_corner()
print(tp.display_solution())
print("Total Cost (Z):", tp.calculate_total_cost())

# Using Least Cost Method
print("\nLeast Cost Method:")
least_cost_solution = tp.least_cost()
print(tp.display_solution())
print("Total Cost (Z):", tp.calculate_total_cost())

# Using Vogel's Approximation Method
print("\nVogel's Approximation Method:")
vogel_solution = tp.vogel_approximation()
print(tp.display_solution())
print("Total Cost (Z):", tp.calculate_total_cost())


