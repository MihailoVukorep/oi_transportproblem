import numpy as np

def north_west_corner(supply, demand):
    """
    Inicijalizuje početno rešenje koristeći metodu severozapadnog ugla.
    """
    supply = supply.copy()
    demand = demand.copy()
    rows, cols = len(supply), len(demand)
    solution = np.zeros((rows, cols))

    i = j = 0
    while i < rows and j < cols:
        if supply[i] < demand[j]:
            solution[i][j] = supply[i]
            demand[j] -= supply[i]
            i += 1
        else:
            solution[i][j] = demand[j]
            supply[i] -= demand[j]
            j += 1

    return solution

def check_degeneracy(solution, supply, demand):
    """
    Proverava da li rešenje ima dovoljno baznih promenljivih.
    """
    m, n = solution.shape
    basic_variables = np.count_nonzero(solution)
    expected_variables = m + n - 1

    if basic_variables < expected_variables:
        print("Rešenje je degenerisano. Dodajemo male vrednosti na nule kako bi se zadovoljila degeneracija.")
        # Dodaj male vrednosti na neka mesta kako bi rešenje postalo degenerisano
        for i in range(m):
            for j in range(n):
                if solution[i][j] == 0 and basic_variables < expected_variables:
                    solution[i][j] = 1e-5
                    basic_variables += 1
    return solution

def calculate_u_v(solution, costs):
    """
    Računa potencijale u i v za MODI metodu.
    """
    rows, cols = solution.shape
    u = np.full(rows, None)
    v = np.full(cols, None)
    u[0] = 0  # Postavljamo prvi potencijal na 0

    # Petlja za računanje vrednosti u i v
    while None in u or None in v:
        for i in range(rows):
            for j in range(cols):
                if solution[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = costs[i][j] - u[i]
                    elif u[i] is None and v[j] is not None:
                        u[i] = costs[i][j] - v[j]
    return u, v

def find_entering_variable(solution, costs, u, v):
    """
    Pronalazi promenljivu koja ulazi u bazu na osnovu modifikovanih troškova.
    """
    rows, cols = solution.shape
    delta = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if solution[i][j] == 0:
                delta[i][j] = costs[i][j] - (u[i] + v[j])

    min_delta = np.min(delta)
    if min_delta >= 0:
        return None, None  # Optimalno rešenje

    # Pronalazi indeks sa najmanjim delta
    enter_i, enter_j = np.unravel_index(np.argmin(delta), delta.shape)
    return enter_i, enter_j

def find_cycle(solution, start):
    """
    Pronalazi ciklus koji uključuje poziciju `start` koristeći iterativni pristup (BFS).
    """
    rows, cols = solution.shape
    queue = [(start, [start])]
    visited = set()

    while queue:
        (current, path) = queue.pop(0)
        x, y = current
        visited.add((x, y))

        # Definišemo susede (pravolinijske pomake)
        neighbors = []
        for i in range(rows):
            if solution[i][y] > 0 or (i, y) == start:
                neighbors.append((i, y))
        for j in range(cols):
            if solution[x][j] > 0 or (x, j) == start:
                neighbors.append((x, j))

        # Prolazimo kroz susede
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            if neighbor == start and len(path) >= 4:
                return path + [start]  # Pronađen ciklus
            elif neighbor not in path:
                queue.append((neighbor, path + [neighbor]))

    return None

def adjust_solution(solution, cycle, entering_value):
    min_val = min(solution[x][y] for x, y in cycle[1::2])
    for idx, (x, y) in enumerate(cycle):
        if idx % 2 == 0:
            solution[x][y] += min_val
        else:
            solution[x][y] -= min_val
    solution[cycle[0][0]][cycle[0][1]] = entering_value

option_add = True

def transportation_algorithm(supply, demand, costs):
    """
    Glavna funkcija koja rešava transportni problem koristeći MODI metodu.
    """

    # Provera da li je problem balansiran
    if sum(supply) != sum(demand):
        raise ValueError("Problem nije balansiran! Ukupna ponuda mora biti jednaka ukupnoj potražnji.")

    # Provera dimenzija troškova
    if costs.shape != (len(supply), len(demand)):
        raise ValueError("Dimenzije matrice troškova nisu u skladu sa brojem dobavljača i potrošača.")

    # Provera nenegativnosti ponude i potražnje
    if any(x < 0 for x in supply) or any(x < 0 for x in demand):
        raise ValueError("Ponuda i potražnja moraju biti nenegativne vrednosti.")

    solution = north_west_corner(supply, demand)
    solution = check_degeneracy(solution, supply, demand)

    max_iterations = 1000
    iterations = 0

    while True:
        if iterations >= max_iterations:
            raise RuntimeError("Prekoračen maksimalni broj iteracija; algoritam možda ne konvergira.")

        u, v = calculate_u_v(solution, costs)
        enter_i, enter_j = find_entering_variable(solution, costs, u, v)

        if enter_i is None:  # Ako je optimalno
            break

        # Implementacija petlje za izlazak promenljive (izmena ciklusa)
        # Nakon optimizacije petlje, ažuriraj rešenje sa novim vrednostima
        # Pronađi ciklus za MODI metodu
        cycle = find_cycle(solution, (enter_i, enter_j))
        if option_add:
            if cycle is None:
                print("Dodavanje degenerisanih promenljivih zbog manjka baznih promenljivih.")
            solution = check_degeneracy(solution, supply, demand)
            continue
        else:
            if cycle is None:
                print("Greška: Nije pronađen ciklus!")
                break
        adjust_solution(solution, cycle, entering_value=1e-5)

        iterations += 1

    return solution

# Primer podataka
supply = [20, 30, 25]  # Kapaciteti dobavljača
demand = [10, 25, 20, 20]  # Potražnje potrošača
costs = np.array([[8, 6, 10, 9],  # Troškovi transporta
                  [9, 12, 13, 7],
                  [14, 9, 16, 5]])

solution = transportation_algorithm(supply, demand, costs)
print("Optimalno rešenje:\n", solution)

