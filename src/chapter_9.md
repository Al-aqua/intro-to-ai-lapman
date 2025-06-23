# Chapter 9: Heuristic Search Techniques & Optimization

## Learning Objectives

By the end of this chapter, you will:

- Understand what heuristic functions are and why they make search more efficient
- Master the concepts of admissibility and consistency in heuristics
- Implement and analyze Hill Climbing and its variations
- Understand and implement Best-First Search and A\* Search algorithms
- Apply heuristic search techniques to solve complex problems like pathfinding and the 8-puzzle
- Recognize when to use different heuristic search strategies

---

## Requirements

- Python 3.8 or later
- Understanding of Chapter 7 & 8 (uninformed search techniques)
- Basic knowledge of priority queues and heaps
- Familiarity with mathematical concepts like distance calculations

---

## 9.1 Introduction to Heuristic Functions

In Chapter 7, we learned about uninformed search techniques that explore the search space without any domain-specific knowledge. While these methods are guaranteed to find solutions (when they exist), they can be very inefficient for large problems.

**Heuristic search** uses domain-specific knowledge to guide the search toward promising areas of the search space, making it much more efficient.

### 9.1.1 What is a Heuristic Function?

A **heuristic function** \( h(n) \) estimates the cost from a given state \( n \) to the nearest goal state. It provides "informed guidance" about which direction to search.

```python
def heuristic_example(current_state, goal_state):
    """
    Example: Manhattan distance heuristic for grid-based problems

    Args:
        current_state: Current position (row, col)
        goal_state: Goal position (row, col)

    Returns:
        int: Estimated distance to goal
    """
    current_row, current_col = current_state
    goal_row, goal_col = goal_state

    # Manhattan distance: |x1-x2| + |y1-y2|
    return abs(current_row - goal_row) + abs(current_col - goal_col)

# Example usage
current = (1, 1)
goal = (4, 5)
estimated_cost = heuristic_example(current, goal)
print(f"Estimated cost from {current} to {goal}: {estimated_cost}")
# Output: Estimated cost from (1, 1) to (4, 5): 7
```

### 9.1.2 Why Heuristics Are Useful

Consider searching for a path in a large maze:

- **Without heuristic**: BFS explores all directions equally, potentially wasting time going away from the goal
- **With heuristic**: We can prioritize exploring states that are closer to the goal

```python
# Example: Comparing exploration without and with heuristics
def demonstrate_heuristic_benefit():
    """
    Demonstrate how heuristics guide search more efficiently
    """
    # Without heuristic: all states seem equally promising
    states_without_heuristic = [(2,1), (1,2), (3,1), (1,3)]
    print("Without heuristic - all states seem equal:")
    for state in states_without_heuristic:
        print(f"  State {state}: no guidance")

    print("\nWith Manhattan distance heuristic to goal (5,5):")
    goal = (5, 5)
    for state in states_without_heuristic:
        h_value = heuristic_example(state, goal)
        print(f"  State {state}: h = {h_value}")

    # The heuristic clearly shows (3,1) is most promising!

demonstrate_heuristic_benefit()
```

### 9.1.3 Common Types of Heuristics

#### 1. Manhattan Distance (L1 Distance)

Used for grid-based problems where you can only move horizontally or vertically:

```python
def manhattan_distance(state1, state2):
    """Calculate Manhattan distance between two points"""
    x1, y1 = state1
    x2, y2 = state2
    return abs(x1 - x2) + abs(y1 - y2)
```

#### 2. Euclidean Distance (L2 Distance)

Used when diagonal movement is allowed:

```python
import math

def euclidean_distance(state1, state2):
    """Calculate Euclidean distance between two points"""
    x1, y1 = state1
    x2, y2 = state2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
```

#### 3. Hamming Distance

Used for problems like the 8-puzzle - counts misplaced tiles:

```python
def hamming_distance(state, goal):
    """
    Calculate Hamming distance (number of misplaced elements)

    Args:
        state: Current state tuple
        goal: Goal state tuple

    Returns:
        int: Number of misplaced elements
    """
    misplaced = 0
    for i in range(len(state)):
        if state[i] != 0 and state[i] != goal[i]:  # Don't count empty space
            misplaced += 1
    return misplaced

# Example for 8-puzzle
current_puzzle = (1, 2, 3, 4, 0, 6, 7, 5, 8)
goal_puzzle = (1, 2, 3, 4, 5, 6, 7, 8, 0)
h_hamming = hamming_distance(current_puzzle, goal_puzzle)
print(f"Hamming distance: {h_hamming}")  # Output: 2 (tiles 5 and 8 are misplaced)
```

---

## 9.2 Properties of Heuristics: Admissibility and Consistency

Not all heuristics are created equal. The quality of a heuristic determines how well it guides the search.

### 9.2.1 Admissibility

A heuristic \( h(n) \) is **admissible** if it never overestimates the true cost to reach the goal. In other words:

\[ h(n) \leq h^\*(n) \]

where \( h^\*(n) \) is the true optimal cost from state \( n \) to the goal.

```python
def check_admissibility_example():
    """
    Example demonstrating admissible vs inadmissible heuristics
    """
    # Suppose the true shortest path from (1,1) to (4,4) is 6 steps
    current = (1, 1)
    goal = (4, 4)
    true_optimal_cost = 6

    # Admissible heuristic (Manhattan distance)
    h_manhattan = manhattan_distance(current, goal)  # = 6
    print(f"Manhattan distance: {h_manhattan}")
    print(f"Admissible? {h_manhattan <= true_optimal_cost}")

    # Inadmissible heuristic (overestimates)
    h_overestimate = h_manhattan * 2  # = 12
    print(f"Overestimate heuristic: {h_overestimate}")
    print(f"Admissible? {h_overestimate <= true_optimal_cost}")

check_admissibility_example()
```

**Why admissibility matters**: Admissible heuristics guarantee that A\* search finds optimal solutions.

### 9.2.2 Consistency (Monotonicity)

A heuristic \( h(n) \) is **consistent** if for every state \( n \) and every successor \( n' \) of \( n \):

\[ h(n) \leq c(n, n') + h(n') \]

where \( c(n, n') \) is the cost of going from \( n \) to \( n' \).

```python
def check_consistency_example():
    """
    Example demonstrating consistent vs inconsistent heuristics
    """
    # States in a path
    state_n = (2, 2)
    state_n_prime = (2, 3)  # One step right
    goal = (5, 5)

    step_cost = 1  # Cost to move from n to n'

    # Check Manhattan distance consistency
    h_n = manhattan_distance(state_n, goal)
    h_n_prime = manhattan_distance(state_n_prime, goal)

    print(f"h({state_n}) = {h_n}")
    print(f"h({state_n_prime}) = {h_n_prime}")
    print(f"Step cost = {step_cost}")
    print(f"Consistency check: {h_n} <= {step_cost} + {h_n_prime}")
    print(f"Consistent? {h_n <= step_cost + h_n_prime}")

check_consistency_example()
```

**Important**: Consistency implies admissibility, but not vice versa.

---

## 9.3 Hill Climbing and Its Variations

**Hill Climbing** is a local search algorithm that continuously moves toward states with better heuristic values. It's like climbing a hill in the dark - you always move upward until you can't go any higher.

### 9.3.1 Basic Hill Climbing Algorithm

```python
def hill_climbing(problem, heuristic_func):
    """
    Basic Hill Climbing algorithm.

    Args:
        problem: A SearchProblem instance
        heuristic_func: Function that takes a state and returns heuristic value

    Returns:
        tuple: (final_state, path_to_final_state)
    """
    current_state = problem.get_initial_state()
    path = [current_state]

    print(f"Starting Hill Climbing from: {current_state}")

    while True:
        # Get all possible next states
        actions = problem.get_actions(current_state)

        if not actions:
            print("No more actions available")
            break

        # Find the best next state
        best_next_state = None
        best_heuristic = heuristic_func(current_state)

        print(f"Current state: {current_state}, h = {best_heuristic}")

        for action in actions:
            next_state = problem.get_result(current_state, action)
            next_heuristic = heuristic_func(next_state)

            print(f"  Action {action} -> {next_state}, h = {next_heuristic}")

            # For hill climbing, we want to minimize heuristic (closer to goal)
            if next_heuristic < best_heuristic:
                best_next_state = next_state
                best_heuristic = next_heuristic

        # If no improvement found, we're at a local optimum
        if best_next_state is None:
            print("Local optimum reached - no better neighbors")
            break

        # Move to the best next state
        current_state = best_next_state
        path.append(current_state)

    return current_state, path
```

### 9.3.2 Problems with Basic Hill Climbing

Hill climbing can get stuck in several situations:

1. **Local Maxima**: A state that's better than all neighbors but not the global optimum
2. **Plateaus**: Flat areas where all neighbors have the same heuristic value
3. **Ridges**: Areas where progress requires moving in multiple dimensions

```python
def demonstrate_hill_climbing_problems():
    """
    Demonstrate the problems with basic hill climbing
    """
    # Create a simple maze where hill climbing might get stuck
    problematic_maze = [
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    # Import our MazeProblem from Chapter 7
    from chapter7 import MazeProblem  # Assuming you have this

    maze_problem = MazeProblem(problematic_maze, start=(0, 0), goal=(4, 4))

    def maze_heuristic(state):
        return manhattan_distance(state, maze_problem.goal_state)

    print("Trying hill climbing on a problematic maze:")
    final_state, path = hill_climbing(maze_problem, maze_heuristic)

    print(f"\nFinal state reached: {final_state}")
    print(f"Goal state: {maze_problem.goal_state}")
    print(f"Reached goal? {final_state == maze_problem.goal_state}")
    print(f"Path length: {len(path)}")

# demonstrate_hill_climbing_problems()  # Uncomment to run
```

### 9.3.3 Steepest Ascent Hill Climbing

Instead of taking the first improvement, **steepest ascent** examines all neighbors and chooses the best one:

```python
def steepest_ascent_hill_climbing(problem, heuristic_func):
    """
    Steepest Ascent Hill Climbing - always choose the best neighbor.

    Args:
        problem: A SearchProblem instance
        heuristic_func: Function that takes a state and returns heuristic value

    Returns:
        tuple: (final_state, path_to_final_state)
    """
    current_state = problem.get_initial_state()
    path = [current_state]

    print(f"Starting Steepest Ascent Hill Climbing from: {current_state}")

    while True:
        actions = problem.get_actions(current_state)

        if not actions:
            break

        # Evaluate ALL neighbors and pick the best
        current_heuristic = heuristic_func(current_state)
        best_next_state = None
        best_heuristic = current_heuristic

        print(f"Current state: {current_state}, h = {current_heuristic}")
        print("Evaluating all neighbors:")

        for action in actions:
            next_state = problem.get_result(current_state, action)
            next_heuristic = heuristic_func(next_state)

            print(f"  {action} -> {next_state}, h = {next_heuristic}")

            if next_heuristic < best_heuristic:
                best_next_state = next_state
                best_heuristic = next_heuristic

        if best_next_state is None:
            print("No improvement found - stopping")
            break

        print(f"Best neighbor: {best_next_state} with h = {best_heuristic}")
        current_state = best_next_state
        path.append(current_state)

    return current_state, path
```

### 9.3.4 Stochastic Hill Climbing

**Stochastic hill climbing** introduces randomness to escape local optima:

```python
import random

def stochastic_hill_climbing(problem, heuristic_func, max_iterations=1000):
    """
    Stochastic Hill Climbing - randomly choose among improving moves.

    Args:
        problem: A SearchProblem instance
        heuristic_func: Function that takes a state and returns heuristic value
        max_iterations: Maximum number of iterations

    Returns:
        tuple: (final_state, path_to_final_state)
    """
    current_state = problem.get_initial_state()
    path = [current_state]

    print(f"Starting Stochastic Hill Climbing from: {current_state}")

    for iteration in range(max_iterations):
        actions = problem.get_actions(current_state)

        if not actions:
            break

        current_heuristic = heuristic_func(current_state)

        # Find all improving moves
        improving_moves = []
        for action in actions:
            next_state = problem.get_result(current_state, action)
            next_heuristic = heuristic_func(next_state)

            if next_heuristic < current_heuristic:
                improving_moves.append((action, next_state, next_heuristic))

        if not improving_moves:
            print(f"No improving moves at iteration {iteration}")
            break

        # Randomly choose among improving moves
        chosen_action, chosen_state, chosen_heuristic = random.choice(improving_moves)

        print(f"Iteration {iteration}: {current_state} -> {chosen_state} (h: {current_heuristic} -> {chosen_heuristic})")

        current_state = chosen_state
        path.append(current_state)

        # Check if we reached the goal
        if problem.is_goal(current_state):
            print(f"Goal reached at iteration {iteration}!")
            break

    return current_state, path
```

---

## 9.4 Best-First Search

**Best-First Search** is a general approach that uses a heuristic to decide which node to expand next. It maintains a priority queue of nodes ordered by their heuristic values.

### 9.4.1 Best-First Search Implementation

```python
import heapq

def best_first_search(problem, heuristic_func):
    """
    Best-First Search algorithm using heuristic to guide search.

    Args:
        problem: A SearchProblem instance
        heuristic_func: Function that takes a state and returns heuristic value

    Returns:
        list: Solution path if found, None if no solution exists
    """
    # Priority queue: (heuristic_value, state, path_to_state)
    frontier = [(heuristic_func(problem.get_initial_state()),
                 problem.get_initial_state(),
                 [problem.get_initial_state()])]

    visited = set()
    nodes_expanded = 0

    print("Starting Best-First Search...")

    while frontier:
        # Get the state with lowest heuristic value
        current_heuristic, current_state, path = heapq.heappop(frontier)

        if current_state in visited:
            continue

        visited.add(current_state)
        nodes_expanded += 1

        print(f"Expanding: {current_state}, h = {current_heuristic}, depth = {len(path) - 1}")

        # Check if goal reached
        if problem.is_goal(current_state):
            print(f"Goal found! Nodes expanded: {nodes_expanded}")
            return path

        # Add successors to frontier
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_result(current_state, action)

            if next_state not in visited:
                next_heuristic = heuristic_func(next_state)
                new_path = path + [next_state]
                heapq.heappush(frontier, (next_heuristic, next_state, new_path))

    print(f"No solution found. Nodes expanded: {nodes_expanded}")
    return None
```

### 9.4.2 Best-First Search Example

```python
def test_best_first_search():
    """Test Best-First Search on a maze problem"""
    # Create a test maze
    test_maze = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    from chapter7 import MazeProblem  # Import from previous chapter
    maze_problem = MazeProblem(test_maze, start=(0, 0), goal=(4, 4))

    def maze_heuristic(state):
        return manhattan_distance(state, maze_problem.goal_state)

    print("Maze:")
    maze_problem.print_maze()

    solution = best_first_search(maze_problem, maze_heuristic)

    if solution:
        print(f"\nSolution found with {len(solution)} steps:")
        print("Path:", solution)
        maze_problem.print_maze(solution)
    else:
        print("No solution found!")

# test_best_first_search()  # Uncomment to run
```

---

## 9.5 A\* Search Algorithm

**A\* (A-star)** is the most famous and widely-used heuristic search algorithm. It combines the benefits of uniform-cost search (optimality) with the efficiency of best-first search (heuristic guidance).

### 9.5.1 How A\* Works

A\* uses an evaluation function \( f(n) = g(n) + h(n) \) where:

- \( g(n) \) = actual cost from start to node \( n \)
- \( h(n) \) = heuristic estimate from node \( n \) to goal
- \( f(n) \) = estimated total cost of path through \( n \)

### 9.5.2 A\* Implementation

```python
def a_star_search(problem, heuristic_func):
    """
    A* Search algorithm - optimal heuristic search.

    Args:
        problem: A SearchProblem instance
        heuristic_func: Admissible heuristic function

    Returns:
        list: Optimal solution path if found, None if no solution exists
    """
    # Priority queue: (f_value, g_value, state, path_to_state)
    # We include g_value to break ties consistently
    initial_state = problem.get_initial_state()
    initial_h = heuristic_func(initial_state)
    initial_g = 0
    initial_f = initial_g + initial_h

    frontier = [(initial_f, initial_g, initial_state, [initial_state])]

    # Keep track of best g-value for each state
    best_g_values = {initial_state: 0}

    nodes_expanded = 0

    print("Starting A* Search...")
    print(f"Initial state: {initial_state}, f = g + h = {initial_g} + {initial_h} = {initial_f}")

    while frontier:
        # Get state with lowest f-value
        current_f, current_g, current_state, path = heapq.heappop(frontier)

        # Skip if we've found a better path to this state
        if current_state in best_g_values and current_g > best_g_values[current_state]:
            continue

        nodes_expanded += 1
        current_h = heuristic_func(current_state)

        print(f"Expanding: {current_state}, f = g + h = {current_g} + {current_h} = {current_f}")

        # Check if goal reached
        if problem.is_goal(current_state):
            print(f"Goal found! Nodes expanded: {nodes_expanded}")
            print(f"Solution cost: {current_g}")
            return path

        # Expand successors
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_result(current_state, action)

            # Calculate g-value for successor
            step_cost = problem.get_cost(current_state, action, next_state)
            next_g = current_g + step_cost

            # Skip if we've seen this state with a better g-value
            if next_state in best_g_values and next_g >= best_g_values[next_state]:
                continue

            # Calculate f-value for successor
            next_h = heuristic_func(next_state)
            next_f = next_g + next_h

            # Add to frontier
            new_path = path + [next_state]
            heapq.heappush(frontier, (next_f, next_g, next_state, new_path))
            best_g_values[next_state] = next_g

            print(f"  Adding successor: {next_state}, f = {next_g} + {next_h} = {next_f}")

    print(f"No solution found. Nodes expanded: {nodes_expanded}")
    return None
```

### 9.5.3 A\* Properties and Guarantees

**Theorem**: If the heuristic is admissible, A\* is optimal.

**Proof idea**: A\* will never expand a node with f-value greater than the optimal solution cost, so it must find the optimal solution first.

```python
def demonstrate_a_star_optimality():
    """
    Demonstrate that A* finds optimal solutions with admissible heuristics
    """
    # Create a simple maze where multiple paths exist
    maze_with_multiple_paths = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ]

    from chapter7 import MazeProblem
    maze_problem = MazeProblem(maze_with_multiple_paths, start=(0, 0), goal=(4, 4))

    def admissible_heuristic(state):
        return manhattan_distance(state, maze_problem.goal_state)

    print("Testing A* optimality:")
    print("Maze with multiple possible paths:")
    maze_problem.print_maze()

    solution = a_star_search(maze_problem, admissible_heuristic)

    if solution:
        print(f"A* solution length: {len(solution) - 1}")
        print("A* path:", solution)
        maze_problem.print_maze(solution)

        # Compare with BFS (also optimal for unweighted graphs)
        from chapter8 import breadth_first_search
        bfs_solution = breadth_first_search(maze_problem)
        print(f"BFS solution length: {len(bfs_solution) - 1}")
        print(f"Both found optimal solution: {len(solution) == len(bfs_solution)}")

# demonstrate_a_star_optimality()  # Uncomment to run
```

---

## 9.6 Python Implementation of Heuristic Search Techniques

Let's create a comprehensive framework that brings together all the heuristic search techniques:

### 9.6.1 Enhanced Search Problem Class

```python
class HeuristicSearchProblem:
    """
    Enhanced search problem class with heuristic support
    """

    def __init__(self, initial_state, goal_state, heuristic_func):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.heuristic_func = heuristic_func

    def get_initial_state(self):
        return self.initial_state

    def is_goal(self, state):
        return state == self.goal_state

    def get_actions(self, state):
        raise NotImplementedError

    def get_result(self, state, action):
        raise NotImplementedError

    def get_cost(self, state, action, next_state):
        return 1  # Default uniform cost

    def heuristic(self, state):
        """Get heuristic value for a state"""
        return self.heuristic_func(state)
```

### 9.6.2 Search Algorithm Comparison Framework

```python
import time

def compare_search_algorithms(problem, algorithms=None):
    """
    Compare different search algorithms on the same problem.

    Args:
        problem: A search problem instance
        algorithms: List of (name, algorithm_function) tuples

    Returns:
        dict: Results for each algorithm
    """
    if algorithms is None:
        algorithms = [
            ("BFS", lambda p: breadth_first_search(p)),
            ("Best-First", lambda p: best_first_search(p, p.heuristic)),
            ("A*", lambda p: a_star_search(p, p.heuristic))
        ]

    results = {}

    print("=" * 60)
    print("SEARCH ALGORITHM COMPARISON")
    print("=" * 60)

    for name, algorithm in algorithms:
        print(f"\n--- {name} ---")

        start_time = time.time()
        solution = algorithm(problem)
        end_time = time.time()

        results[name] = {
            'solution': solution,
            'time': end_time - start_time,
            'path_length': len(solution) - 1 if solution else None,
            'found_solution': solution is not None
        }

        if solution:
            print(f"Solution found: {len(solution) - 1} steps")
        else:
            print("No solution found")
        print(f"Time taken: {results[name]['time']:.4f} seconds")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        if result['found_solution']:
            print(f"{name:12}: {result['path_length']:2} steps, {result['time']:.4f}s")
        else:
            print(f"{name:12}: No solution, {result['time']:.4f}s")

    return results
```

### 9.6.3 Comprehensive Algorithm Comparison Examples

Now let's demonstrate the power of this comparison framework with several practical examples:

**First**, let's create a few maze problems to test our algorithms:

```python
def create_test_maze_problems():
    """
    Create different maze problems for testing search algorithms
    """
    # Simple maze - easy problem
    simple_maze = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    # Complex maze - harder problem
    complex_maze = [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0]
    ]

    # Maze with long optimal path
    long_path_maze = [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    return {
        'simple': simple_maze,
        'complex': complex_maze,
        'long_path': long_path_maze
    }
```

**Next**, let's create a function to compare the performance of different search algorithms on these maze problems:

```python
def demonstrate_algorithm_comparison():
    """
    Comprehensive demonstration of search algorithm comparison
    """
    print("COMPREHENSIVE SEARCH ALGORITHM COMPARISON")
    print("=" * 80)

    # Get test mazes
    test_mazes = create_test_maze_problems()

    for maze_name, maze_grid in test_mazes.items():
        print(f"\n{'='*20} {maze_name.upper()} MAZE {'='*20}")

        # Create maze problem with heuristic
        maze_problem = MazeProblem(maze_grid,
                                 start=(0, 0),
                                 goal=(len(maze_grid)-1, len(maze_grid[0])-1))

        # Add heuristic method to the problem
        maze_problem.heuristic = lambda state: manhattan_distance(state, maze_problem.goal_state)

        print(f"Maze size: {len(maze_grid)} x {len(maze_grid[0])}")
        print(f"Start: {maze_problem.initial_state}")
        print(f"Goal: {maze_problem.goal_state}")
        print("\nMaze layout (0=open, 1=wall):")
        for row in maze_grid:
            print(' '.join(str(cell) for cell in row))

        # Compare algorithms
        results = compare_search_algorithms(maze_problem)

        # Additional analysis
        print(f"\n--- DETAILED ANALYSIS ---")

        # Find the optimal solution length
        optimal_length = None
        for name, result in results.items():
            if result['found_solution']:
                if optimal_length is None or result['path_length'] < optimal_length:
                    optimal_length = result['path_length']

        if optimal_length:
            print(f"Optimal solution length: {optimal_length}")

            # Check which algorithms found optimal solutions
            optimal_algorithms = []
            for name, result in results.items():
                if result['found_solution'] and result['path_length'] == optimal_length:
                    optimal_algorithms.append(name)

            print(f"Algorithms finding optimal solution: {', '.join(optimal_algorithms)}")

            # Speed comparison among optimal algorithms
            if len(optimal_algorithms) > 1:
                fastest_time = min(results[name]['time'] for name in optimal_algorithms)
                fastest_algorithm = [name for name in optimal_algorithms
                                   if results[name]['time'] == fastest_time][0]
                print(f"Fastest optimal algorithm: {fastest_algorithm} ({fastest_time:.4f}s)")

# demonstrate_algorithm_comparison()  # Uncomment to run
```

---

## 9.7 Practical Exercises

### 9.7.1 Exercise 1: Enhanced 8-Puzzle with Multiple Heuristics

Complete the enhanced 8-puzzle implementation with different heuristics:

```python
class Enhanced8Puzzle(HeuristicSearchProblem):
    """
    Enhanced 8-Puzzle with multiple heuristic options
    """

    def __init__(self, initial_state, heuristic_type="manhattan"):
        goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)

        # Choose heuristic function based on type
        if heuristic_type == "hamming":
            heuristic_func = self.hamming_distance
        elif heuristic_type == "manhattan":
            heuristic_func = self.manhattan_distance
        elif heuristic_type == "linear_conflict":
            heuristic_func = self.linear_conflict_distance
        else:
            raise ValueError(f"Unknown heuristic type: {heuristic_type}")

        super().__init__(initial_state, goal_state, heuristic_func)

    def hamming_distance(self, state):
        """
        TODO: Implement Hamming distance heuristic
        Count the number of misplaced tiles (excluding empty space)

        Args:
            state: Current state tuple

        Returns:
            int: Number of misplaced tiles
        """
        # Your code here
        pass

    def manhattan_distance(self, state):
        """
        TODO: Implement Manhattan distance heuristic
        Sum of distances each tile is from its goal position

        Args:
            state: Current state tuple

        Returns:
            int: Sum of Manhattan distances for all tiles
        """
        # Your code here
        # Hint: Convert 1D index to 2D coordinates and back
        pass

    def linear_conflict_distance(self, state):
        """
        TODO: Implement Linear Conflict heuristic (advanced)
        Manhattan distance + 2 * number of linear conflicts

        A linear conflict occurs when two tiles are in their goal row/column
        but in wrong order relative to each other.

        Args:
            state: Current state tuple

        Returns:
            int: Manhattan distance + linear conflict penalty
        """
        # Your code here (this is challenging!)
        # Start with Manhattan distance, then add conflict detection
        pass

    def get_actions(self, state):
        """Get possible moves for the empty space"""
        empty_pos = state.index(0)
        row, col = empty_pos // 3, empty_pos % 3

        actions = []

        # Check each possible move
        if row > 0: actions.append("UP")
        if row < 2: actions.append("DOWN")
        if col > 0: actions.append("LEFT")
        if col < 2: actions.append("RIGHT")

        return actions

    def get_result(self, state, action):
        """Apply action and return new state"""
        state_list = list(state)
        empty_pos = state.index(0)
        row, col = empty_pos // 3, empty_pos % 3

        if action == "UP":
            swap_pos = (row - 1) * 3 + col
        elif action == "DOWN":
            swap_pos = (row + 1) * 3 + col
        elif action == "LEFT":
            swap_pos = row * 3 + (col - 1)
        elif action == "RIGHT":
            swap_pos = row * 3 + (col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Swap empty space with target position
        state_list[empty_pos], state_list[swap_pos] = state_list[swap_pos], state_list[empty_pos]

        return tuple(state_list)

    def print_state(self, state):
        """Print state in 3x3 format"""
        print("Current state:")
        for i in range(0, 9, 3):
            row = []
            for j in range(3):
                if state[i + j] == 0:
                    row.append('_')
                else:
                    row.append(str(state[i + j]))
            print(' '.join(row))
        print()

# Test different heuristics
def test_8puzzle_heuristics():
    """Test 8-puzzle with different heuristics"""
    # A moderately difficult puzzle
    initial_puzzle = (1, 2, 3, 4, 0, 6, 7, 5, 8)

    heuristics = ["hamming", "manhattan", "linear_conflict"]

    for heuristic_type in heuristics:
        print(f"\n{'='*50}")
        print(f"Testing with {heuristic_type.upper()} heuristic")
        print(f"{'='*50}")

        puzzle = Enhanced8Puzzle(initial_puzzle, heuristic_type)
        puzzle.print_state(initial_puzzle)

        solution = a_star_search(puzzle, puzzle.heuristic)

        if solution:
            print(f"Solution found with {len(solution) - 1} moves")
        else:
            print("No solution found")

# test_8puzzle_heuristics()  # Uncomment to run
```

### 9.7.2 Exercise 2: Pathfinding with Obstacles

Create a pathfinding problem with different terrain costs:

```python
class TerrainPathfinding(HeuristicSearchProblem):
    """
    Pathfinding problem with different terrain costs

    Terrain types:
    0 = impassable (wall)
    1 = normal terrain (cost 1)
    2 = difficult terrain (cost 3)
    3 = water (cost 5)
    """

    def __init__(self, terrain_map, start, goal):
        """
        TODO: Initialize the terrain pathfinding problem

        Args:
            terrain_map: 2D list with terrain costs
            start: Starting position (row, col)
            goal: Goal position (row, col)
        """
        # Your code here
        # Use Euclidean distance as heuristic
        pass

    def euclidean_heuristic(self, state):
        """
        TODO: Calculate Euclidean distance to goal

        Args:
            state: Current position (row, col)

        Returns:
            float: Euclidean distance to goal
        """
        # Your code here
        pass

    def get_actions(self, state):
        """
        TODO: Get valid actions (8-directional movement)
        Actions: N, NE, E, SE, S, SW, W, NW
        """
        # Your code here
        pass

    def get_result(self, state, action):
        """
        TODO: Get resulting state after action
        """
        # Your code here
        pass

    def get_cost(self, state, action, next_state):
        """
        TODO: Get cost of moving to next_state

        Consider:
        - Terrain cost of destination
        - Diagonal moves cost sqrt(2) times more than orthogonal
        """
        # Your code here
        pass

# Test terrain pathfinding
def test_terrain_pathfinding():
    """Test pathfinding with different terrain costs"""
    # Create a terrain map
    terrain = [
        [1, 1, 2, 2, 1],
        [1, 0, 2, 3, 1],
        [1, 1, 1, 3, 1],
        [2, 2, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]

    pathfinding = TerrainPathfinding(terrain, start=(0, 0), goal=(4, 4))

    print("Terrain Map (0=wall, 1=normal, 2=difficult, 3=water):")
    for row in terrain:
        print(' '.join(str(cell) for cell in row))

    solution = a_star_search(pathfinding, pathfinding.heuristic)

    if solution:
        print(f"Path found with {len(solution) - 1} steps")
        print("Path:", solution)

        # Calculate total cost
        total_cost = 0
        for i in range(len(solution) - 1):
            current = solution[i]
            next_state = solution[i + 1]
            # You'll need to implement cost calculation
            # total_cost += pathfinding.get_cost(current, None, next_state)

        # print(f"Total path cost: {total_cost}")

# test_terrain_pathfinding()  # Uncomment to run
```

### 9.7.3 Exercise 3: Heuristic Quality Analysis

Create a tool to analyze heuristic quality:

```python
def analyze_heuristic_quality(problem, heuristic_func, sample_states=None):
    """
    TODO: Analyze the quality of a heuristic function

    Metrics to calculate:
    1. Average heuristic value vs actual distance
    2. Percentage of admissible estimates
    3. Effective branching factor

    Args:
        problem: Search problem instance
        heuristic_func: Heuristic function to analyze
        sample_states: List of states to test (if None, generate random states)

    Returns:
        dict: Analysis results
    """
    # Your code here
    # This is a research-level exercise - implement what you can!
    pass

def compare_heuristic_effectiveness():
    """
    TODO: Compare different heuristics on the same problem

    Create multiple versions of the same problem with different heuristics
    and compare:
    - Number of nodes expanded
    - Time taken
    - Solution quality
    """
    # Your code here
    pass
```

---

## 9.8 Key Takeaways

1. **Heuristic Functions**: Domain-specific knowledge that guides search toward promising areas
2. **Admissibility**: Never overestimate → guarantees optimal solutions in A\*
3. **Consistency**: Ensures efficient search without reopening nodes
4. **Algorithm Trade-offs**:
   - Hill Climbing: Fast but can get stuck in local optima
   - Best-First: Uses heuristic but not optimal
   - A\*: Optimal with admissible heuristics but uses more memory
5. **Real-World Impact**: Heuristic search powers GPS navigation, game AI, robotics, and many other applications

---

## Summary

In this chapter, we explored heuristic search techniques that use domain knowledge to search more efficiently:

- **Heuristic Functions**: Estimate cost to goal, guide search direction
- **Properties**: Admissibility ensures optimality, consistency ensures efficiency
- **Hill Climbing**: Local search that can get stuck but is memory-efficient
- **Best-First Search**: Uses heuristic to prioritize exploration
- **A\* Search**: Combines actual cost and heuristic for optimal solutions
- **Practical Applications**: From GPS navigation to puzzle solving

These techniques form the foundation for intelligent search in AI systems. The key insight is that a little domain knowledge (the heuristic) can dramatically improve search efficiency while maintaining solution quality.

---

> **An advice for you**: Always remember to look for the best heuristic for your problem. It's not always obvious, but it's worth the effort to find the right one.
