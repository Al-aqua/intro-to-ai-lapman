# Chapter 10: Advanced Heuristic Search and Applications

## Learning Objectives

By the end of this chapter, you will:

- Understand and implement the A\* Search algorithm
- Apply heuristic search techniques to solve complex problems like pathfinding and the 8-puzzle
- Implement and analyze heuristic search in Python
- Compare different search algorithms and heuristics
- Recognize the trade-offs and real-world applications of heuristic search strategies

---

## Requirements

- Python 3.8 or later
- Understanding of Chapters 7, 8, and 9 (including uninformed search and basic heuristic techniques)
- Basic knowledge of priority queues, heaps, and time complexity analysis
- Familiarity with mathematical concepts like distance calculations

---

## 10.1 A\* Search Algorithm

**A\* (A-star)** is the most famous and widely-used heuristic search algorithm. It combines the benefits of uniform-cost search (optimality) with the efficiency of best-first search (heuristic guidance).

### 10.1.1 How A\* Works

A\* uses an evaluation function \( f(n) = g(n) + h(n) \) where:

- \( g(n) \) = actual cost from start to node \( n \)
- \( h(n) \) = heuristic estimate from node \( n \) to goal
- \( f(n) \) = estimated total cost of path through \( n \)

### 10.1.2 A\* Implementation

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

### 10.1.3 A\* Properties and Guarantees

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

## 10.2 Python Implementation of Heuristic Search Techniques

Let's create a comprehensive framework that brings together all the heuristic search techniques:

### 10.2.1 Enhanced Search Problem Class

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

### 10.2.2 Search Algorithm Comparison Framework

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

### 10.2.3 Comprehensive Algorithm Comparison Examples

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

## 10.3 Practical Exercises

### 10.3.1 Exercise 1: Enhanced 8-Puzzle with Multiple Heuristics

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

### 10.3.2 Exercise 2: Pathfinding with Obstacles

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

### 10.3.3 Exercise 3: Heuristic Quality Analysis

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

## 10.4 Key Takeaways

1. **Heuristic Functions**: Domain-specific knowledge that guides search toward promising areas
2. **Admissibility**: Never overestimate → guarantees optimal solutions in A\*
3. **Consistency**: Ensures efficient search without reopening nodes
4. **Algorithm Trade-offs**:
   - Hill Climbing: Fast but can get stuck in local optima
   - Best-First: Uses heuristic but not optimal
   - A\*: Optimal with admissible heuristics but uses more memory
5. **Real-World Impact**: Heuristic search powers GPS navigation, game AI, robotics, and many other applications

---

## 10.5 Summary

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
