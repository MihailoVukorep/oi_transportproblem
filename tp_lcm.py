import numpy as np

def transportation_least_cost_method(costs, supply, demand):
    if np.any(costs < 0) or np.any(supply < 0) or np.any(demand < 0):
        raise ValueError("All values in costs, supply, and demand must be non-negative.")

    if costs.shape != (len(supply), len(demand)):
        raise ValueError("Dimensions of costs must match lengths of supply and demand.")

    # Kreiranje kopije originalne matrice troškova i osiguranje da je tip float
    original_costs = costs.copy().astype(float)
    costs = costs.astype(float)  # Uveravamo se da je costs float
    num_suppliers = len(supply)
    num_customers = len(demand)

    total_supply = np.sum(supply)
    total_demand = np.sum(demand)

    # Provera balansiranosti problema i dodavanje fiktivnog dobavljača ili potrošača
    if total_supply > total_demand:
        demand = np.append(demand, total_supply - total_demand)
        costs = np.column_stack([costs, np.zeros(num_suppliers, dtype=float)])  # Osiguraj da je tip float
        original_costs = np.column_stack([original_costs, np.zeros(num_suppliers, dtype=float)])
    elif total_demand > total_supply:
        supply = np.append(supply, total_demand - total_supply)
        costs = np.vstack([costs, np.zeros(num_customers, dtype=float)])  # Osiguraj da je tip float
        original_costs = np.vstack([original_costs, np.zeros(num_customers, dtype=float)])

    num_suppliers = len(supply)
    num_customers = len(demand)

    # Inicijalizacija matrice alokacija sa nulama
    allocations = np.zeros((num_suppliers, num_customers))

    # Metoda najmanjih cena
    while np.sum(supply) > 0 and np.sum(demand) > 0:
        # Pronalazi najmanji trošak u matrici troškova
        min_cost_index = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
        i, j = min_cost_index

        # Određuje koliko se može alocirati na ovu poziciju
        quantity = min(supply[i], demand[j])

        # Alocira tu količinu i ažurira zalihe i potražnju
        allocations[i][j] = quantity
        supply[i] -= quantity
        demand[j] -= quantity

        # Blokira ovu poziciju u matrici troškova postavljanjem cene na inf
        if supply[i] == 0:
            costs[i, :] = np.inf  # Ovaj dobavljač je iscrpljen
        if demand[j] == 0:
            costs[:, j] = np.inf  # Ovaj potrošač je iscrpljen

    # Računanje ukupnog troška koristeći originalnu matricu troškova
    total_cost = np.nansum(allocations * original_costs[:num_suppliers, :num_customers])

    return allocations, total_cost

if __name__ == "__main__":
    costs = np.array([[40, 60, 90], [60, 80, 70], [50, 50, 100]])
    supply = np.array([40, 35, 50])
    demand = np.array([35, 45, 35])

    """costs = np.array([[10, 12, 0], [8, 4, 3], [6, 9, 4], [7, 8, 5]])
    supply = np.array([20, 30, 20, 10])
    demand = np.array([10, 40, 30])"""

    optimal_allocations, total_cost = transportation_least_cost_method(costs, supply, demand)

    print("Optimal Allocations:")
    print(optimal_allocations)
    print("Total Cost:", total_cost)

