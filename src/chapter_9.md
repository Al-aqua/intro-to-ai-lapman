# Chapter 9: Introduction to Heuristic Functions and Basic Search Techniques

## Learning Objectives

By the end of this chapter, you will:

- Understand what heuristic functions are and why they make search more efficient
- Master the concepts of admissibility and consistency in heuristics
- Implement and analyze Hill Climbing and its variations
- Understand and implement Best-First Search
- Recognize when to use different heuristic search strategies for basic problems

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

**Why admissibility matters**: Admissible heuristics guarantee that certain search algorithms find optimal solutions.

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

## 9.5 Key Takeaways

Here are the key insights from this chapter to reinforce your understanding:

1. **Heuristic Functions**: Domain-specific knowledge that estimates the cost to the goal, guiding search toward more promising areas and improving efficiency.
2. **Admissibility**: A heuristic that never overestimates the true cost ensures that algorithms can find optimal solutions when used correctly.
3. **Consistency**: This property guarantees that the heuristic provides a reliable, non-increasing estimate along any path, making search more efficient by avoiding unnecessary node re-examinations.
4. **Algorithm Trade-offs**:
   - Hill Climbing: Simple and fast for local improvements but can get stuck in local optima, plateaus, or ridges.
   - Best-First Search: Uses heuristics to prioritize exploration but may not always find the optimal path.
5. **Practical Considerations**: When applying these techniques, select heuristics based on the problem domain to balance efficiency and solution quality.

---
