# Chapter 7: Uninformed Search Techniques & Problem Formulation

## Learning Objectives

By the end of this chapter, you will:

- Understand what constitutes a search problem in AI and how to formulate problems for computational solving.
- Master the key components: states, actions, goal tests, and path costs.
- Implement and analyze Depth-First Search (DFS) and Breadth-First Search (BFS) algorithms.
- Apply search techniques to solve practical problems like mazes and puzzles.
- Understand the trade-offs between different search strategies.

---

## Requirements

- Python 3.8 or later
- Basic understanding of Python data structures (lists, dictionaries, sets)
- Familiarity with classes and functions in Python

## 7.1 What is a Search Problem?

In Artificial Intelligence, many problems can be formulated as **search problems**. A search problem involves finding a sequence of actions that leads from an initial state to a goal state.

### Real-World Examples of Search Problems:

- **Navigation**: Finding the shortest route from your home to university
- **Puzzle Solving**: Solving a Rubik's cube or sliding puzzle
- **Game Playing**: Finding the best move in chess
- **Planning**: Scheduling tasks or resources
- **Robotics**: Path planning for autonomous vehicles

### 7.1.1 Components of a Search Problem

Every search problem consists of four essential components:

1. **Initial State**: Where we start
2. **Actions**: What we can do from each state
3. **Goal Test**: How we know we've reached our objective
4. **Path Cost**: The cost of taking a sequence of actions

Let's explore each component in detail:

#### 1. States

A **state** represents a particular configuration of the world or problem. States capture all the relevant information needed to continue the search.

```python
# Example: In a maze, a state might be represented as coordinates
initial_state = (0, 0)  # Starting position (row, column)
current_state = (2, 3)  # Current position in the maze
goal_state = (4, 4)     # Target position
```

#### 2. Actions

**Actions** are the possible moves or operations we can perform from a given state. Actions transform one state into another.

```python
# Example: In a maze, possible actions might be:
def get_actions(state):
    """Return list of possible actions from current state"""
    actions = []
    row, col = state

    # Possible moves: up, down, left, right
    possible_moves = [
        ('UP', row - 1, col),
        ('DOWN', row + 1, col),
        ('LEFT', row, col - 1),
        ('RIGHT', row, col + 1)
    ]

    for action, new_row, new_col in possible_moves:
        if is_valid_position(new_row, new_col):
            actions.append(action)

    return actions
```

#### 3. Goal Test

The **goal test** determines whether we've reached our objective. It's a function that returns True if the current state satisfies our goal.

```python
def is_goal(state, goal_state):
    """Check if current state is the goal state"""
    return state == goal_state

# Example usage
current = (4, 4)
goal = (4, 4)
print(is_goal(current, goal))  # Output: True
```

#### 4. Path Cost

**Path cost** measures the cost of a solution path. This could be the number of steps, distance traveled, time taken, or any other relevant metric.

```python
def path_cost(path):
    """Calculate the cost of a path (number of steps)"""
    return len(path) - 1  # Subtract 1 because path includes initial state

# Example
path = [(0,0), (0,1), (1,1), (2,1), (2,2)]
cost = path_cost(path)
print(f"Path cost: {cost}")  # Output: Path cost: 4
```

---

## 7.2 Representing Problems for Search in Python

Let's create a general framework for representing search problems in Python:

### 7.2.1 The SearchProblem Class

```python
class SearchProblem:
    """
    Abstract base class for search problems.
    This class defines the interface that all search problems should implement.
    """

    def __init__(self, initial_state, goal_state):
        """
        Initialize the search problem.

        Args:
            initial_state: The starting state
            goal_state: The target state we want to reach
        """
        self.initial_state = initial_state
        self.goal_state = goal_state

    def get_initial_state(self):
        """Return the initial state of the problem."""
        return self.initial_state

    def is_goal(self, state):
        """
        Test if the given state is a goal state.

        Args:
            state: The state to test

        Returns:
            bool: True if state is a goal state, False otherwise
        """
        return state == self.goal_state

    def get_actions(self, state):
        """
        Return a list of actions that can be executed in the given state.

        Args:
            state: The current state

        Returns:
            list: List of possible actions
        """
        raise NotImplementedError("Subclasses must implement get_actions")

    def get_result(self, state, action):
        """
        Return the state that results from executing the given action in the given state.

        Args:
            state: The current state
            action: The action to execute

        Returns:
            The resulting state after applying the action
        """
        raise NotImplementedError("Subclasses must implement get_result")

    def get_cost(self, state, action, next_state):
        """
        Return the cost of applying action in state to reach next_state.

        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state

        Returns:
            float: The cost of the action (default is 1)
        """
        return 1  # Default uniform cost
```

### 7.2.2 Example: Maze Problem Implementation

Let's implement a concrete maze problem using our framework:

```python
class MazeProblem(SearchProblem):
    """
    A maze navigation problem.
    The maze is represented as a 2D grid where:
    - 0 represents an open path
    - 1 represents a wall
    - Start and goal positions are given as (row, col) coordinates
    """

    def __init__(self, maze, start, goal):
        """
        Initialize the maze problem.

        Args:
            maze: 2D list representing the maze (0=open, 1=wall)
            start: Starting position as (row, col)
            goal: Goal position as (row, col)
        """
        super().__init__(start, goal)
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if maze else 0

    def is_valid_position(self, row, col):
        """
        Check if a position is valid (within bounds and not a wall).

        Args:
            row: Row coordinate
            col: Column coordinate

        Returns:
            bool: True if position is valid, False otherwise
        """
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.maze[row][col] == 0)

    def get_actions(self, state):
        """
        Get possible actions from current state.
        Actions are: 'UP', 'DOWN', 'LEFT', 'RIGHT'
        """
        actions = []
        row, col = state

        # Define possible moves
        moves = [
            ('UP', row - 1, col),
            ('DOWN', row + 1, col),
            ('LEFT', row, col - 1),
            ('RIGHT', row, col + 1)
        ]

        # Check which moves are valid
        for action, new_row, new_col in moves:
            if self.is_valid_position(new_row, new_col):
                actions.append(action)

        return actions

    def get_result(self, state, action):
        """
        Get the resulting state after applying an action.

        Args:
            state: Current position (row, col)
            action: Action to take ('UP', 'DOWN', 'LEFT', 'RIGHT')

        Returns:
            tuple: New position (row, col)
        """
        row, col = state

        if action == 'UP':
            return (row - 1, col)
        elif action == 'DOWN':
            return (row + 1, col)
        elif action == 'LEFT':
            return (row, col - 1)
        elif action == 'RIGHT':
            return (row, col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

    def print_maze(self, path=None):
        """
        Print the maze with optional path visualization.

        Args:
            path: List of positions representing the solution path
        """
        # Create a copy of the maze for display
        display_maze = [row[:] for row in self.maze]

        # Mark the path if provided
        if path:
            for i, (row, col) in enumerate(path):
                if i == 0:
                    display_maze[row][col] = 'S'  # Start
                elif i == len(path) - 1:
                    display_maze[row][col] = 'G'  # Goal
                else:
                    display_maze[row][col] = '.'  # Path
        else:
            # Just mark start and goal
            start_row, start_col = self.initial_state
            goal_row, goal_col = self.goal_state
            display_maze[start_row][start_col] = 'S'
            display_maze[goal_row][goal_col] = 'G'

        # Print the maze
        print("Maze:")
        for row in display_maze:
            print(' '.join(str(cell) for cell in row))
        print()
```

---

## 7.3 Depth-First Search (DFS)

**Depth-First Search** is a search algorithm that explores as far as possible along each branch before backtracking. It uses a **stack** data structure (or recursion) to keep track of the search frontier.

### 7.3.1 How DFS Works

1. Start with the initial state
2. Explore the first available action
3. Continue exploring deeper until you reach a dead end or find the goal
4. If you reach a dead end, backtrack to the most recent state with unexplored actions
5. Repeat until you find the goal or exhaust all possibilities

### 7.3.2 DFS Characteristics

- **Complete**: No (can get stuck in infinite loops)
- **Optimal**: No (doesn't guarantee shortest path)
- **Time Complexity**: O(b^m) where b is branching factor, m is maximum depth
- **Space Complexity**: O(bm) - only needs to store path to current node

### 7.3.3 DFS Implementation

```python
def depth_first_search(problem):
    """
    Implement Depth-First Search algorithm.

    Args:
        problem: A SearchProblem instance

    Returns:
        list: Solution path if found, None if no solution exists
    """
    # Initialize the frontier with the initial state
    # Each item in frontier is (state, path_to_state)
    frontier = [(problem.get_initial_state(), [problem.get_initial_state()])]

    # Keep track of visited states to avoid cycles
    visited = set()

    # Statistics for analysis
    nodes_expanded = 0

    print("Starting Depth-First Search...")

    while frontier:
        # Pop from the end (stack behavior - LIFO)
        current_state, path = frontier.pop()

        # Skip if we've already visited this state
        if current_state in visited:
            continue

        # Mark current state as visited
        visited.add(current_state)
        nodes_expanded += 1

        print(f"Exploring state: {current_state}")

        # Check if we've reached the goal
        if problem.is_goal(current_state):
            print(f"Goal found! Nodes expanded: {nodes_expanded}")
            return path

        # Get all possible actions from current state
        actions = problem.get_actions(current_state)

        # Add all successor states to frontier
        for action in actions:
            next_state = problem.get_result(current_state, action)

            # Only add if not visited
            if next_state not in visited:
                new_path = path + [next_state]
                frontier.append((next_state, new_path))

    print(f"No solution found. Nodes expanded: {nodes_expanded}")
    return None  # No solution found
```

### 7.3.4 DFS Example: Solving a Simple Maze

```python
# Define a simple maze
simple_maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]

# Create the maze problem
maze_problem = MazeProblem(simple_maze, start=(0, 0), goal=(4, 4))

# Print the initial maze
print("Initial Maze (S=Start, G=Goal, 1=Wall, 0=Open):")
maze_problem.print_maze()

# Solve using DFS
solution_path = depth_first_search(maze_problem)

if solution_path:
    print(f"Solution found with {len(solution_path)} steps:")
    print("Path:", solution_path)
    print("\nMaze with solution path:")
    maze_problem.print_maze(solution_path)
else:
    print("No solution exists!")
```

---

## 7.4 Breadth-First Search (BFS)

**Breadth-First Search** explores all states at the current depth before moving to states at the next depth level. It uses a **queue** data structure to manage the search frontier.

### 7.4.1 How BFS Works

1. Start with the initial state in a queue
2. Remove the first state from the queue
3. If it's the goal, we're done
4. Otherwise, add all its successors to the end of the queue
5. Repeat until queue is empty or goal is found

### 7.4.2 BFS Characteristics

- **Complete**: Yes (if solution exists and branching factor is finite)
- **Optimal**: Yes (finds shortest path in terms of number of steps)
- **Time Complexity**: O(b^d) where b is branching factor, d is depth of solution
- **Space Complexity**: O(b^d) - needs to store all nodes at current level

### 7.4.3 BFS Implementation

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

### 7.4.4 Comparing DFS and BFS

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

## 7.5 Practical Exercises

### 7.5.1 Exercise 1: Water Jug Problem

The classic water jug problem: You have two jugs, one that holds 4 gallons and another that holds 3 gallons. How can you measure exactly 2 gallons?

**Your Task**: Complete the `WaterJugProblem` class implementation.

```python
class WaterJugProblem(SearchProblem):
    """
    Water Jug Problem: Given two jugs with capacities jug1_capacity and jug2_capacity,
    find a sequence of operations to reach the goal state.

    State: (amount_in_jug1, amount_in_jug2)
    Actions: Fill jug1, Fill jug2, Empty jug1, Empty jug2,
             Pour jug1->jug2, Pour jug2->jug1
    """

    def __init__(self, jug1_capacity, jug2_capacity, goal_amount):
        """
        Initialize the water jug problem.

        Args:
            jug1_capacity: Capacity of first jug
            jug2_capacity: Capacity of second jug
            goal_amount: Target amount to measure
        """
        self.jug1_capacity = jug1_capacity
        self.jug2_capacity = jug2_capacity
        initial_state = (0, 0)  # Both jugs start empty
        self.goal_amount = goal_amount
        super().__init__(initial_state, None)

    def is_goal(self, state):
        """
        TODO: Check if either jug contains the goal amount.

        Args:
            state: Current state (jug1_amount, jug2_amount)

        Returns:
            bool: True if goal is reached, False otherwise
        """
        # Your code here
        pass

    def get_actions(self, state):
        """
        TODO: Get all possible actions from current state.

        Possible actions:
        - "FILL_JUG1": Fill jug 1 to capacity
        - "FILL_JUG2": Fill jug 2 to capacity
        - "EMPTY_JUG1": Empty jug 1
        - "EMPTY_JUG2": Empty jug 2
        - "POUR_JUG1_TO_JUG2": Pour from jug 1 to jug 2
        - "POUR_JUG2_TO_JUG1": Pour from jug 2 to jug 1

        Args:
            state: Current state (jug1_amount, jug2_amount)

        Returns:
            list: List of valid actions from this state
        """
        # Your code here
        pass

    def get_result(self, state, action):
        """
        TODO: Apply action to state and return new state.

        Args:
            state: Current state (jug1_amount, jug2_amount)
            action: Action to apply

        Returns:
            tuple: New state after applying action
        """
        # Your code here
        pass

# Test your implementation
water_jug = WaterJugProblem(jug1_capacity=4, jug2_capacity=3, goal_amount=2)
solution = breadth_first_search(water_jug)

if solution:
    print("Solution found!")
    for i, state in enumerate(solution):
        print(f"Step {i}: Jug1={state[0]}L, Jug2={state[1]}L")
else:
    print("No solution found.")
```

**Your solution:**

```
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
```

### 7.5.2 Exercise 2: 8-Puzzle Problem

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

**Your solution:**

```
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
```

### 7.5.3 Exercise 3: Compare Search Performance

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

**Your solution:**

```
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
____________________________________________________________________________
```

---

## 7.6 Key Takeaways

1. **Problem Formulation**: The key to solving AI problems is properly formulating them with states, actions, goal tests, and costs.

2. **Search Algorithms**: DFS and BFS are fundamental uninformed search techniques with different characteristics:

   - **DFS**: Memory efficient, not optimal, can get stuck
   - **BFS**: Finds optimal solutions, memory intensive, always complete

3. **Implementation Patterns**: Using classes and proper data structures makes search algorithms more maintainable and reusable.

4. **Real-World Applications**: Search problems appear everywhere in AI, from navigation to puzzle solving to planning.

5. **Trade-offs**: Different search algorithms have different trade-offs between optimality, completeness, time, and space complexity.

---

## Summary

In this chapter, we've learned how to:

- Formulate problems as search problems with states, actions, goals, and costs
- Implement a general framework for search problems in Python
- Understand and implement Depth-First Search and Breadth-First Search
- Apply these techniques to practical problems like maze navigation and the water jug problem
- Analyze the characteristics and trade-offs of different search strategies

These uninformed search techniques form the foundation for more advanced AI search algorithms. In the next chapter, we'll explore heuristic search techniques that use domain knowledge to search more efficiently!

---

> **Next Chapter Preview**: In Chapter 8, we'll learn about heuristic search techniques like A\* that use problem-specific knowledge to find solutions more efficiently than blind search methods.
