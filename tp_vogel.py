import numpy as np

def find_row_penalties(costs, supply, demand):
    penalties = []
    for i, row in enumerate(costs):
        if supply[i] == 0:
            penalties.append(-np.inf)
            continue
        valid_costs = [c for j, c in enumerate(row) if demand[j] > 0 and c != np.inf]
        if len(valid_costs) >= 2:
            sorted_costs = sorted(valid_costs)
            penalties.append(sorted_costs[1] - sorted_costs[0])
        elif len(valid_costs) == 1:
            penalties.append(valid_costs[0])
        else:
            penalties.append(-np.inf)
    return penalties

def find_column_penalties(costs, supply, demand):
    penalties = []
    for j in range(len(demand)):
        if demand[j] == 0:
            penalties.append(-np.inf)
            continue
        valid_costs = [costs[i][j] for i in range(len(supply)) if supply[i] > 0 and costs[i][j] != np.inf]
        if len(valid_costs) >= 2:
            sorted_costs = sorted(valid_costs)
            penalties.append(sorted_costs[1] - sorted_costs[0])
        elif len(valid_costs) == 1:
            penalties.append(valid_costs[0])
        else:
            penalties.append(-np.inf)
    return penalties

def vogel_method(costs, supply, demand):
    # Kreiranje kopije originalne matrice troškova i vektora ponude i potražnje
    original_costs = costs.copy()
    supply = supply.copy()
    demand = demand.copy()
    num_suppliers = len(supply)
    num_customers = len(demand)

    total_supply = np.sum(supply)
    total_demand = np.sum(demand)

    # Provera balansiranosti
    if total_supply > total_demand:
        demand = np.append(demand, total_supply - total_demand)
        costs = np.column_stack([costs, np.zeros(num_suppliers)])
        original_costs = np.column_stack([original_costs, np.zeros(num_suppliers)])
    elif total_demand > total_supply:
        supply = np.append(supply, total_demand - total_supply)
        costs = np.vstack([costs, np.zeros(num_customers)])
        original_costs = np.vstack([original_costs, np.zeros(num_customers)])

    num_suppliers = len(supply)
    num_customers = len(demand)

    # Inicijalizacija matrice alokacija
    allocations = np.zeros((num_suppliers, num_customers))
    costs = costs.astype(float)

    while np.sum(supply) > 0 and np.sum(demand) > 0:
        # Računanje kazni
        row_penalties = find_row_penalties(costs, supply, demand)
        col_penalties = find_column_penalties(costs, supply, demand)

        # Pronalaženje najveće kazne
        max_row_penalty = max(row_penalties) if row_penalties else -np.inf
        max_col_penalty = max(col_penalties) if col_penalties else -np.inf

        if max_row_penalty >= max_col_penalty and max_row_penalty != -np.inf:
            i = row_penalties.index(max_row_penalty)
            valid_costs = [(j, costs[i][j]) for j in range(num_customers)
                          if demand[j] > 0 and costs[i][j] != np.inf]
            if not valid_costs:
                continue
            j = min(valid_costs, key=lambda x: x[1])[0]
        elif max_col_penalty != -np.inf:
            j = col_penalties.index(max_col_penalty)
            valid_costs = [(i, costs[i][j]) for i in range(num_suppliers)
                          if supply[i] > 0 and costs[i][j] != np.inf]
            if not valid_costs:
                continue
            i = min(valid_costs, key=lambda x: x[1])[0]
        else:
            break

        # Alokacija
        quantity = min(supply[i], demand[j])
        allocations[i][j] = quantity
        supply[i] -= quantity
        demand[j] -= quantity

        # Ažuriranje matrice troškova
        if supply[i] == 0:
            costs[i, :] = np.inf
        if demand[j] == 0:
            costs[:, j] = np.inf

    # Računanje ukupnog troška
    total_cost = np.sum(allocations * original_costs[:num_suppliers, :num_customers])

    return allocations, total_cost

if __name__ == "__main__":
    costs = np.array([[40, 60, 90], [60, 80, 70], [50, 50, 100]])
    supply = np.array([40, 35, 50])
    demand = np.array([35, 45, 35])

    """costs = np.array([[10, 12, 0], [8, 4, 3], [6, 9, 4], [7, 8, 5]])
    supply = np.array([20, 30, 20, 10])
    demand = np.array([10, 40, 30])"""

    optimal_allocations, total_cost = vogel_method(costs, supply, demand)

    print("Optimal Allocations:")
    print(optimal_allocations)
    print("Total Cost:", total_cost)
