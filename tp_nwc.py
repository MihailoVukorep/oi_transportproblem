import numpy as np

def transportation(costs, supply, demand):
    if np.any(costs < 0) or np.any(supply < 0) or np.any(demand < 0):
        raise ValueError("All values in costs, supply, and demand must be non-negative.")

    if costs.shape != (len(supply), len(demand)):
        raise ValueError("Dimensions of costs must match lengths of supply and demand.")

    num_suppliers = len(supply)
    num_customers = len(demand)

    total_supply = np.sum(supply)
    total_demand = np.sum(demand)

    # Check if unbalanced and add a dummy supply or demand
    if total_supply > total_demand:
        # Add a dummy demand
        demand = np.append(demand, total_supply - total_demand)
        costs = np.column_stack([costs, np.zeros(num_suppliers)])
    elif total_demand > total_supply:
        # Add a dummy supply
        supply = np.append(supply, total_demand - total_supply)
        costs = np.vstack([costs, np.zeros(num_customers)])

    num_suppliers = len(supply)
    num_customers = len(demand)

    # Initialize arrays to store allocations and track remaining supply/demand
    allocations = np.zeros((num_suppliers, num_customers))

    i, j = 0, 0

    #North west corner method
    while i < num_suppliers and j < num_customers:
        # Find the minimum between supply[i] and demand[j]
        quantity = min(supply[i], demand[j])

        # Allocate the quantity and update supply/demand
        allocations[i][j] = quantity
        supply[i] -= quantity
        demand[j] -= quantity

        # Move to the next supplier or customer based on remaining supply or demand
        if supply[i] == 0:
            i += 1
        else:
            j += 1

    # Calculate total cost based on original costs matrix
    total_cost = np.sum(allocations * costs[:num_suppliers, :num_customers])

    return allocations, total_cost

if __name__ == "__main__":
    # Example inputs
    # Unbalanced problem (supply > demand)
    """costs = np.array([[40, 60, 90], [60, 80, 70], [50, 50, 100]])
    supply = np.array([40, 35, 50])
    demand = np.array([35, 45, 35])"""

    costs = np.array([[10, 12, 0], [8, 4, 3], [6, 9, 4], [7, 8, 5]])
    supply = np.array([20, 30, 20, 10])
    demand = np.array([10, 40, 30])

    optimal_allocations, total_cost = transportation(costs, supply, demand)

    print("Optimal Allocations:")
    print(optimal_allocations)
    print("Total Cost:", total_cost)

