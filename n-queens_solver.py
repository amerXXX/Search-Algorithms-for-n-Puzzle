import random
import time
import math

# ------------ Part A: Incremental Formulation Solvers ------------

def solve_backtracking(n):
    """
    Simple backtracking solver for N-Queens.
    Returns one solution as a list of column indices by row.
    """
    def is_safe(queens, row, col):
        for r, c in enumerate(queens):
            if c == col or abs(row - r) == abs(col - c):
                return False
        return True

    def backtrack(queens):
        row = len(queens)
        if row == n:
            return queens
        for col in range(n):
            if is_safe(queens, row, col):
                res = backtrack(queens + [col])
                if res:
                    return res
        return None

    return backtrack([])


def solve_forward_checking(n):
    """
    Backtracking with forward checking: maintain domains of columns for remaining rows.
    """
    domains = [set(range(n)) for _ in range(n)]

    def backtrack(queens, domains):
        row = len(queens)
        if row == n:
            return queens
        for col in sorted(domains[row]):
            # copy domains
            new_domains = [d.copy() for d in domains]
            valid = True
            # assign and forward propagate
            for r in range(row+1, n):
                if col in new_domains[r]:
                    new_domains[r].remove(col)
                d1 = col + (r-row)
                d2 = col - (r-row)
                for d in (d1, d2):
                    if 0 <= d < n and d in new_domains[r]:
                        new_domains[r].remove(d)
                if not new_domains[r]:
                    valid = False
                    break
            if not valid:
                continue
            res = backtrack(queens + [col], new_domains)
            if res:
                return res
        return None

    return backtrack([], domains)


def solve_mrv_lcv(n):
    """
    Backtracking + forward checking + MRV + LCV heuristics.
    """
    domains = [set(range(n)) for _ in range(n)]

    def select_row(domains, assigned):
        # MRV: pick unassigned row with smallest domain
        unassigned = [r for r in range(n) if r not in assigned]
        return min(unassigned, key=lambda r: len(domains[r]))

    def order_values(row, domains, assigned):
        # LCV: least constraining values first
        def count_constraints(col):
            count = 0
            for r in range(n):
                if r not in assigned and r != row:
                    if col in domains[r]: count += 1
                    dist = abs(r-row)
                    for d in (col+dist, col-dist):
                        if 0 <= d < n and d in domains[r]: count += 1
            return count
        return sorted(domains[row], key=count_constraints)

    def backtrack(queens, domains, assigned):
        if len(assigned) == n:
            # reconstruct solution
            sol = [None]*n
            for r, c in queens.items(): sol[r] = c
            return sol
        row = select_row(domains, assigned)
        for col in order_values(row, domains, assigned):
            # copy domains
            new_domains = [d.copy() for d in domains]
            valid = True
            # propagate
            for r in range(n):
                if r not in assigned and r != row:
                    if col in new_domains[r]: new_domains[r].remove(col)
                    dist = abs(r-row)
                    for d in (col+dist, col-dist):
                        if 0 <= d < n and d in new_domains[r]: new_domains[r].remove(d)
                    if not new_domains[r]:
                        valid = False
                        break
            if not valid:
                continue
            assigned.add(row)
            queens[row] = col
            res = backtrack(queens, new_domains, assigned)
            if res:
                return res
            assigned.remove(row)
            del queens[row]
        return None

    return backtrack({}, domains, set())

# ------------ Part B: Hill Climbing / Simulated Annealing ------------

def random_initial(n):
    # one queen per row, random column
    return [random.randrange(n) for _ in range(n)]


def conflicts(state):
    # count number of attacking pairs
    n = len(state)
    c = 0
    for i in range(n):
        for j in range(i+1, n):
            if state[i] == state[j] or abs(state[i]-state[j]) == j-i:
                c += 1
    return c


def simulated_annealing(n, max_steps=100000, initial_temp=1.0, cooling_rate=0.99):
    state = random_initial(n)
    best = state[:]
    temp = initial_temp
    for step in range(max_steps):
        if conflicts(state) == 0:
            return state
        # pick random row and move to best neighbor
        row = random.randrange(n)
        current_conflicts = conflicts(state)
        new_col = random.choice([c for c in range(n) if c != state[row]])
        new_state = state[:]
        new_state[row] = new_col
        delta = conflicts(new_state) - current_conflicts
        if delta < 0 or random.random() < math.exp(-delta/temp):
            state = new_state
            if conflicts(state) < conflicts(best): best = state[:]
        temp *= cooling_rate
    return best

# ------------ Timing and Table Generation ------------

def measure_runtimes(solver, ns, trials=5):
    results = {}
    for n in ns:
        times = []
        for _ in range(trials):
            start = time.time()
            solver(n)
            times.append(time.time() - start)
        results[n] = sum(times)/trials
    return results

if __name__ == "__main__":
    # Example usage
    ns = [8, 10, 12]
    print("Backtracking:", measure_runtimes(solve_backtracking, ns))
    print("FC:", measure_runtimes(solve_forward_checking, ns))
    print("MRV+LCV:", measure_runtimes(solve_mrv_lcv, ns))
    print("SimAnneal:", measure_runtimes(lambda n: simulated_annealing(n), ns))
