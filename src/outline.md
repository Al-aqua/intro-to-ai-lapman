# Course Title: Introduction to Artificial Intelligence - Practical Applications

## Course Description

This course is a hands-on introduction to Artificial Intelligence (AI), focusing on practical techniques for solving problems. You’ll learn the basics of AI concepts and algorithms by working with two programming languages: Prolog and Python. We’ll start with Prolog, a logic-based language commonly used in AI for tasks like knowledge representation and reasoning, providing a foundation in declarative programming and symbolic AI. Then, we’ll move on to Python, a powerful and widely-used language for AI development, where you’ll implement search algorithms, optimization techniques, constraint satisfaction problems, and strategies for game-playing.

By the end of the course, you’ll have a strong foundation in AI programming and problem formulation. You’ll understand key concepts like search methods, heuristic optimization, constraint satisfaction, and decision-making in games, and importantly, you’ll gain the skills to build AI systems and tackle real-world challenges through practical implementation.

**Prerequisites:** Familiarity with basic programming concepts (variables, loops, functions) in at least one high-level programming language is recommended.

## Course Outline

### Part 1: Introduction to Prolog (Chapters 1–3)

**Chapter 1: Getting Started with Prolog & Logic Programming**

- What is Prolog? Overview of logic programming and its role in AI (e.g., expert systems, knowledge representation).
- Installing and setting up Prolog.
- Basic syntax and structure of Prolog programs.
- Facts, rules, and queries.
- Simple examples: family relationships, basic reasoning puzzles.

**Chapter 2: Prolog Fundamentals**

- Variables, atoms, and terms.
- Pattern matching and unification: the core of Prolog's execution.
- Recursion in Prolog: building powerful predicates.
- Lists and list operations (e.g., member, append, reverse).
- Introduction to Arithmetic in Prolog.
- Practical exercises: building a knowledge base for a small domain.

**Chapter 3: Advanced Prolog Concepts & Problem Solving**

- Backtracking and implicit search in Prolog's execution model.
- Controlling search: Introduction to the Cut operator (`!`).
- Negation as Failure (`not` / `\+`).
- Defining and using predicates for more complex logic.
- Practical exercises: solving puzzles (e.g., N-Queens, Sudoku variants) and simple AI problems demonstrating logic and search.

### Part 2: Introduction to Python (Chapters 4–6)

**Chapter 4: Getting Started with Python**

- What is Python? Overview of its role in AI.
- Installing and setting up Python.
- Python basics: variables, data types, and operators.
- Input/output and basic control structures (if-else, loops).
- Writing and running Python scripts.

**Chapter 5: Python Data Structures and Functions**

- Lists, tuples, dictionaries, and sets.
- Functions: defining, calling, and passing arguments.
- Modules and libraries in Python.
- Practical exercises: basic data manipulation and function writing.

**Chapter 6: Object-Oriented Programming in Python**

- Classes and objects.
- Methods, attributes, and inheritance.
- Practical exercises: creating simple classes for AI-related tasks.

### Part 3: AI Techniques in Python (Chapters 7–10)

**Chapter 7: Uninformed Search Techniques & Problem Formulation**

- What is a search problem? States, actions, goal tests, path costs.
- Representing problems for search in Python.
- Depth-First Search (DFS) and its characteristics.
- Breadth-First Search (BFS) and its characteristics.
- Practical exercises: solving mazes, water jug problem.

**Chapter 8: Heuristic Search Techniques & Optimization**

- Introduction to heuristic functions: what they are and why they are useful.
- Properties of heuristics: Admissibility and Consistency.
- Hill Climbing and its variations (e.g., Steepest Ascent, Stochastic).
- Addressing local optima: Introduction to Simulated Annealing (optional, high-level).
- Best-First Search.
- A\* Search algorithm: combining cost and heuristic.
- Python implementation of heuristic search techniques.
- Practical exercises: advanced pathfinding (e.g., city shortest paths), optimization problems (e.g., 8-puzzle).

**Chapter 9: Constraint Satisfaction Problems (CSPs)**

- What are CSPs? Variables, domains, constraints. Examples and applications (scheduling, timetabling).
- Basic Backtracking for CSPs.
- Improving efficiency: Forward checking.
- Constraint propagation: Arc Consistency (AC-3 algorithm).
- Python implementation of CSP solvers.
- Practical exercises: solving Sudoku, N-Queens revisited, cryptarithmetic puzzles.

**Chapter 10: Game Playing and Adversarial Search**

- Introduction to game theory and AI in two-player, zero-sum games.
- Game trees and states.
- Minimax algorithm: finding optimal moves in perfect-information games.
- Alpha-beta pruning: improving Minimax efficiency.
- Designing effective evaluation functions (heuristics for games).
- Python implementation of game-playing strategies.
- Practical exercises: building AI for simple games (e.g., Tic-Tac-Toe, Connect Four, simple Othello).
