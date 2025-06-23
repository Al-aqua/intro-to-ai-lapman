# Chapter 8: Breadth-First Search & Search Algorithm Analysis

## Learning Objectives

By the end of this chapter, you will:

- Implement and analyze Breadth-First Search (BFS)
- Compare DFS and BFS characteristics
- Understand completeness and optimality in search
- Apply search techniques to more complex problems
- Analyze search algorithm performance

---

## Requirements

- Python 3.8 or later
- Basic understanding of Python data structures (lists, dictionaries, sets)
- Familiarity with classes and functions in Python

---

## 8.1 Breadth-First Search (BFS)

**Breadth-First Search** explores all states at the current depth before moving to states at the next depth level. It uses a **queue** data structure to manage the search frontier.

### 8.1.1 How BFS Works

1. Start with the initial state in a queue
2. Remove the first state from the queue
3. If it's the goal, we're done
4. Otherwise, add all its successors to the end of the queue
5. Repeat until queue is empty or goal is found

### 8.1.2 BFS Characteristics

- **Complete**: Yes (if solution exists and branching factor is finite)
- **Optimal**: Yes (finds shortest path in terms of number of steps)
- **Time Complexity**: O(b^d) where b is branching factor, d is depth of solution
- **Space Complexity**: O(b^d) - needs to store all nodes at current level

### 8.1.3 BFS Implementation

```python
from collections import deque

def breadth_first_search(problem):
    """
    Implement Breadth-First Search algorithm.

    Args:
        problem: A SearchProblem instance

    Returns:
        list: Solution path if found, None if no solution exists
    """
    # Initialize the frontier with the initial state
    # Using deque for efficient queue operations
    frontier = deque([(problem.get_initial_state(), [problem.get_initial_state()])])

    # Keep track of visited states
    visited = set()

    # Statistics
    nodes_expanded = 0

    print("Starting Breadth-First Search...")

    while frontier:
        # Remove from front (queue behavior - FIFO)
        current_state, path = frontier.popleft()

        # Skip if already visited
        if current_state in visited:
            continue

        # Mark as visited
        visited.add(current_state)
        nodes_expanded += 1

        print(f"Exploring state: {current_state} (depth: {len(path) - 1})")

        # Check if goal reached
        if problem.is_goal(current_state):
            print(f"Goal found! Nodes expanded: {nodes_expanded}")
            return path

        # Add all successors to frontier
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_result(current_state, action)

            if next_state not in visited:
                new_path = path + [next_state]
                frontier.append((next_state, new_path))

    print(f"No solution found. Nodes expanded: {nodes_expanded}")
    return None
```

### 8.1.4 Comparing DFS and BFS

Let's create a function to compare both algorithms:

```python
def compare_search_algorithms(problem):
    """
    Compare DFS and BFS on the same problem.

    Args:
        problem: A SearchProblem instance
    """
    print("=" * 50)
    print("COMPARING SEARCH ALGORITHMS")
    print("=" * 50)

    # Test DFS
    print("\n--- DEPTH-FIRST SEARCH ---")
    dfs_solution = depth_first_search(problem)

    print("\n--- BREADTH-FIRST SEARCH ---")
    bfs_solution = breadth_first_search(problem)

    # Compare results
    print("\n--- COMPARISON ---")
    if dfs_solution and bfs_solution:
        print(f"DFS solution length: {len(dfs_solution)}")
        print(f"BFS solution length: {len(bfs_solution)}")
        print(f"BFS optimal: {len(bfs_solution) <= len(dfs_solution)}")
    elif bfs_solution:
        print("BFS found solution, DFS did not")
    elif dfs_solution:
        print("DFS found solution, BFS did not")
    else:
        print("Neither algorithm found a solution")

# Test the comparison
compare_search_algorithms(maze_problem)
```

---

## 8.2 Comparing DFS and BFS

### 8.2.1 Side-by-Side Comparison

| Property         | DFS        | BFS         |
| ---------------- | ---------- | ----------- |
| Completeness     | No (loops) | Yes         |
| Optimality       | No         | Yes (steps) |
| Time Complexity  | O(b^m)     | O(b^d)      |
| Space Complexity | O(bm)      | O(b^d)      |

### 8.2.2 When to Use Each Algorithm

Add practical guidance:

- Use BFS when:
  - You need the shortest path
  - The solution is likely close to the root
- Use DFS when:
  - Memory is constrained
  - The solution is deep in the tree
  - You need any solution quickly

---

## 8.3 Practical Exercises

### 8.3.1 Exercise 1: 8-Puzzle Problem

The 8-puzzle consists of a 3×3 grid with 8 numbered tiles and one empty space. The goal is to arrange the tiles in order.

**Your Task**: Implement the `EightPuzzleProblem` class.

```python
class EightPuzzleProblem(SearchProblem):
    """
    8-Puzzle Problem: Arrange numbered tiles in a 3x3 grid.

    State: Tuple of 9 elements representing the grid (0 represents empty space)
    Example: (1, 2, 3, 4, 5, 6, 7, 8, 0) represents:
    1 2 3
    4 5 6
    7 8 _
    """

    def __init__(self, initial_state):
        """
        Initialize the 8-puzzle problem.

        Args:
            initial_state: Tuple of 9 elements (0 represents empty space)
        """
        goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        super().__init__(initial_state, goal_state)

    def find_empty_position(self, state):
        """
        TODO: Find the position of the empty space (0) in the grid.

        Args:
            state: Current state tuple

        Returns:
            int: Index of empty space (0-8)
        """
        # Your code here
        pass

    def get_actions(self, state):
        """
        TODO: Get possible moves for the empty space.

        Actions: "UP", "DOWN", "LEFT", "RIGHT"
        (These represent moving the empty space in that direction)

        Args:
            state: Current state tuple

        Returns:
            list: List of valid actions
        """
        # Your code here
        # Hint: Think about which positions the empty space can move to
        # from each position in the 3x3 grid
        pass

    def get_result(self, state, action):
        """
        TODO: Apply action to state and return new state.

        Args:
            state: Current state tuple
            action: Action to apply ("UP", "DOWN", "LEFT", "RIGHT")

        Returns:
            tuple: New state after applying action
        """
        # Your code here
        # Hint: Convert tuple to list, swap positions, convert back to tuple
        pass

    def print_state(self, state):
        """
        Print the state in a readable 3x3 format.
        """
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

# Test your implementation
initial = (1, 2, 3, 4, 0, 6, 7, 5, 8)  # Simple puzzle
puzzle = EightPuzzleProblem(initial)

print("Initial puzzle:")
puzzle.print_state(initial)

print("Goal puzzle:")
puzzle.print_state(puzzle.goal_state)

# Try to solve (warning: might take a while with BFS!)
# solution = breadth_first_search(puzzle)
```

### 8.3.2 Exercise 2: Compare Search Performance

**Your Task**: Create a function that compares DFS and BFS performance on different maze sizes.

```python
def create_random_maze(rows, cols, wall_probability=0.3):
    """
    TODO: Create a random maze with given dimensions.

    Args:
        rows: Number of rows
        cols: Number of columns
        wall_probability: Probability that a cell is a wall (0.0 to 1.0)

    Returns:
        2D list representing the maze

    Note: Make sure start (0,0) and goal (rows-1, cols-1) are always open!
    """
    import random
    # Your code here
    pass

def performance_comparison(maze_sizes):
    """
    TODO: Compare DFS vs BFS performance on different maze sizes.

    Args:
        maze_sizes: List of (rows, cols) tuples

    Should print:
    - Maze size
    - Whether each algorithm found a solution
    - Path length for each algorithm
    - Time taken (you can use time.time())
    """
    import time
    # Your code here
    pass

# Test with different maze sizes
sizes_to_test = [(5, 5), (10, 10), (15, 15)]
performance_comparison(sizes_to_test)
```

---

## Key Takeaways

1. BFS guarantees optimality but requires more memory
2. Algorithm choice depends on problem constraints
3. Performance analysis helps select appropriate algorithms
4. More complex problems require careful state representation

---

> **Next Chapter Preview**: In Chapter 9, we'll learn about Constraint Satisfaction Problems (CSPs), where we'll learn to solve problems by finding assignments that satisfy multiple constraints simultaneously, using techniques like backtracking and constraint propagation.
