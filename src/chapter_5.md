# Lab 5: Python Data Structures and Functions

## Learning Objectives

By the end of this chapter, you will:

- Master Python's core data structures: lists, tuples, dictionaries, and sets.
- Understand when and how to use each data structure effectively.
- Create and use functions to organize and reuse code.
- Work with modules and libraries to extend Python's capabilities.
- Apply these concepts to solve AI-related problems.

---

## Requirements

- Completion of Lab 4 (Python basics)
- Python 3.8 or later installed
- A text editor or IDE for writing Python code

## 5.1 Introduction to Data Structures

**Data structures** are ways of organizing and storing data so that it can be accessed and modified efficiently. In AI, choosing the right data structure is crucial for performance and clarity. Python provides several built-in data structures that are perfect for AI applications.

### Why Data Structures Matter in AI:

- **Efficiency**: Proper data structures make algorithms faster
- **Organization**: Keep related data together (e.g., student records, game states)
- **Flexibility**: Different structures suit different AI problems
- **Memory Management**: Optimize how data is stored and accessed

---

## 5.2 Lists

**Lists** are ordered, mutable collections that can store different types of data. They're one of the most versatile data structures in Python.

### 5.2.1 Creating Lists

```python
# Empty list
empty_list = []
also_empty = list()

# List with initial values
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]
mixed = [1, "hello", 3.14, True]

# Lists can contain other lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

print(numbers)  # [1, 2, 3, 4, 5]
print(mixed)    # [1, 'hello', 3.14, True]
```

### 5.2.2 Accessing List Elements

```python
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

# Positive indexing (starts from 0)
print(fruits[0])    # apple (first element)
print(fruits[2])    # cherry (third element)

# Negative indexing (starts from -1)
print(fruits[-1])   # elderberry (last element)
print(fruits[-2])   # date (second to last)

# Slicing [start:end:step]
print(fruits[1:4])    # ['banana', 'cherry', 'date']
print(fruits[:3])     # ['apple', 'banana', 'cherry'] (first 3)
print(fruits[2:])     # ['cherry', 'date', 'elderberry'] (from index 2)
print(fruits[::2])    # ['apple', 'cherry', 'elderberry'] (every 2nd)
print(fruits[::-1])   # Reverse the list
```

### 5.2.3 Modifying Lists

```python
# Lists are mutable - you can change them
scores = [85, 92, 78, 96, 88]

# Change a single element
scores[2] = 82
print(scores)  # [85, 92, 82, 96, 88]

# Change multiple elements
scores[1:3] = [90, 85]
print(scores)  # [85, 90, 85, 96, 88]
```

### 5.2.4 List Methods

```python
cities = ["New York", "London", "Tokyo"]

# Adding elements
cities.append("Paris")           # Add to end
print(cities)  # ['New York', 'London', 'Tokyo', 'Paris']

cities.insert(1, "Berlin")      # Insert at specific position
print(cities)  # ['New York', 'Berlin', 'London', 'Tokyo', 'Paris']

cities.extend(["Sydney", "Cairo"])  # Add multiple elements
print(cities)  # ['New York', 'Berlin', 'London', 'Tokyo', 'Paris', 'Sydney', 'Cairo']

# Removing elements
cities.remove("Berlin")         # Remove first occurrence
print(cities)  # ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Cairo']

last_city = cities.pop()        # Remove and return last element
print(last_city)  # Cairo
print(cities)     # ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']

second_city = cities.pop(1)     # Remove and return element at index 1
print(second_city)  # London
print(cities)       # ['New York', 'Tokyo', 'Paris', 'Sydney']

# Other useful methods
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(numbers.count(1))         # 2 (count occurrences)
print(numbers.index(4))         # 2 (find index of first occurrence)

numbers.sort()                  # Sort in place
print(numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]

numbers.reverse()               # Reverse in place
print(numbers)  # [9, 6, 5, 4, 3, 2, 1, 1]
```

<!-- ### 5.2.5 List Comprehensions -->
<!---->
<!-- List comprehensions provide a concise way to create lists: -->
<!---->
<!-- ```python -->
<!-- # Traditional way -->
<!-- squares = [] -->
<!-- for x in range(10): -->
<!--     squares.append(x**2) -->
<!-- print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81] -->
<!---->
<!-- # List comprehension way -->
<!-- squares = [x**2 for x in range(10)] -->
<!-- print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81] -->
<!---->
<!-- # With condition -->
<!-- even_squares = [x**2 for x in range(10) if x % 2 == 0] -->
<!-- print(even_squares)  # [0, 4, 16, 36, 64] -->
<!---->
<!-- # Processing strings -->
<!-- words = ["hello", "world", "python", "ai"] -->
<!-- lengths = [len(word) for word in words] -->
<!-- print(lengths)  # [5, 5, 6, 2] -->
<!---->
<!-- # Nested list comprehension (for 2D data) -->
<!-- matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)] -->
<!-- print(matrix)  # [[1, 2, 3], [2, 4, 6], [3, 6, 9]] -->
<!-- ``` -->

---

## 5.3 Tuples

**Tuples** are ordered, immutable collections. Once created, you cannot change their contents.

### 5.3.1 Creating Tuples

```python
# Empty tuple
empty_tuple = ()
also_empty = tuple()

# Tuple with values
coordinates = (3, 4)
rgb_color = (255, 128, 0)
mixed_tuple = (1, "hello", 3.14)

# Single element tuple (note the comma!)
single = (42,)  # Without comma, it's just parentheses around a number
print(type(single))  # <class 'tuple'>

# Tuples without parentheses (tuple packing)
point = 10, 20
print(point)  # (10, 20)
print(type(point))  # <class 'tuple'>
```

### 5.3.2 Accessing Tuple Elements

```python
student = ("Alice", 20, "Computer Science", 3.8)

# Indexing (same as lists)
print(student[0])   # Alice
print(student[-1])  # 3.8

# Slicing (same as lists)
print(student[1:3])  # (20, 'Computer Science')

# Tuple unpacking
name, age, major, gpa = student
print(f"{name} is {age} years old, majoring in {major} with GPA {gpa}")
```

### 5.3.3 Tuple Methods

```python
numbers = (1, 2, 3, 2, 4, 2, 5)

# Count occurrences
print(numbers.count(2))  # 3

# Find index
print(numbers.index(4))  # 4

# Length
print(len(numbers))  # 7
```

### 5.3.4 When to Use Tuples vs Lists

**Use Tuples when:**

- Data shouldn't change (coordinates, RGB values, database records)
- You need a hashable type (for dictionary keys)
- Returning multiple values from functions
- Performance is critical (tuples are slightly faster)

**Use Lists when:**

- Data needs to be modified
- You need methods like append, remove, etc.
- Working with collections that grow or shrink

```python
# Good use of tuples
def get_name_age():
    return "Bob", 25  # Returns a tuple

name, age = get_name_age()  # Tuple unpacking

# Dictionary with tuple keys
locations = {
    (0, 0): "Origin",
    (1, 1): "Northeast",
    (-1, -1): "Southwest"
}
```

---

## 5.4 Dictionaries

**Dictionaries** are unordered collections of key-value pairs. They're perfect for storing related information and creating mappings.

### 5.4.1 Creating Dictionaries

```python
# Empty dictionary
empty_dict = {}
also_empty = dict()

# Dictionary with initial values
student = {
    "name": "Alice",
    "age": 20,
    "major": "AI",
    "gpa": 3.8
}

# Different ways to create dictionaries
grades = dict(math=95, science=88, english=92)
print(grades)  # {'math': 95, 'science': 88, 'english': 92}

# From list of tuples
pairs = [("a", 1), ("b", 2), ("c", 3)]
letter_numbers = dict(pairs)
print(letter_numbers)  # {'a': 1, 'b': 2, 'c': 3}
```

### 5.4.2 Accessing Dictionary Elements

```python
person = {
    "name": "John",
    "age": 30,
    "city": "New York",
    "occupation": "Engineer"
}

# Access by key
print(person["name"])  # John
print(person["age"])   # 30

# Safe access with get() method
print(person.get("name"))        # John
print(person.get("salary"))      # None (key doesn't exist)
print(person.get("salary", 0))   # 0 (default value)

# Check if key exists
if "age" in person:
    print(f"Age: {person['age']}")

# Get all keys, values, or items
print(person.keys())    # dict_keys(['name', 'age', 'city', 'occupation'])
print(person.values())  # dict_values(['John', 30, 'New York', 'Engineer'])
print(person.items())   # dict_items([('name', 'John'), ('age', 30), ...])
```

### 5.4.3 Modifying Dictionaries

```python
inventory = {"apples": 50, "bananas": 30, "oranges": 25}

# Add or update items
inventory["grapes"] = 40        # Add new item
inventory["apples"] = 45        # Update existing item
print(inventory)  # {'apples': 45, 'bananas': 30, 'oranges': 25, 'grapes': 40}

# Update multiple items
inventory.update({"bananas": 35, "pears": 20})
print(inventory)  # {'apples': 45, 'bananas': 35, 'oranges': 25, 'grapes': 40, 'pears': 20}

# Remove items
del inventory["oranges"]        # Remove specific key
print(inventory)  # {'apples': 45, 'bananas': 35, 'grapes': 40, 'pears': 20}

removed_value = inventory.pop("grapes")  # Remove and return value
print(removed_value)  # 40
print(inventory)      # {'apples': 45, 'bananas': 35, 'pears': 20}

# Remove and return arbitrary item
item = inventory.popitem()
print(item)       # ('pears', 20)
print(inventory)  # {'apples': 45, 'bananas': 35}

# Clear all items
inventory.clear()
print(inventory)  # {}
```

<!-- ### 5.4.4 Dictionary Comprehensions -->
<!---->
<!-- ```python -->
<!-- # Create dictionary from lists -->
<!-- names = ["Alice", "Bob", "Charlie"] -->
<!-- ages = [25, 30, 35] -->
<!-- people = {name: age for name, age in zip(names, ages)} -->
<!-- print(people)  # {'Alice': 25, 'Bob': 30, 'Charlie': 35} -->
<!---->
<!-- # Dictionary comprehension with condition -->
<!-- numbers = range(1, 11) -->
<!-- squares = {n: n**2 for n in numbers if n % 2 == 0} -->
<!-- print(squares)  # {2: 4, 4: 16, 6: 36, 8: 64, 10: 100} -->
<!---->
<!-- # Transform existing dictionary -->
<!-- original = {"a": 1, "b": 2, "c": 3} -->
<!-- doubled = {k: v*2 for k, v in original.items()} -->
<!-- print(doubled)  # {'a': 2, 'b': 4, 'c': 6} -->
<!-- ``` -->

---

## 5.5 Sets

**Sets** are unordered collections of unique elements. They're perfect for removing duplicates and performing mathematical set operations.

### 5.5.1 Creating Sets

```python
# Empty set (note: {} creates an empty dictionary, not set!)
empty_set = set()

# Set with initial values
colors = {"red", "green", "blue"}
numbers = {1, 2, 3, 4, 5}

# Create set from list (removes duplicates)
duplicates = [1, 2, 2, 3, 3, 3, 4]
unique_numbers = set(duplicates)
print(unique_numbers)  # {1, 2, 3, 4}

# Create set from string
letters = set("hello")
print(letters)  # {'h', 'e', 'l', 'o'} (note: only one 'l')
```

### 5.5.2 Set Operations

```python
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Add elements
set1.add(6)
print(set1)  # {1, 2, 3, 4, 5, 6}

# Remove elements
set1.remove(6)      # Raises error if element doesn't exist
set1.discard(10)    # Doesn't raise error if element doesn't exist

# Mathematical set operations
print(set1 | set2)  # Union: {1, 2, 3, 4, 5, 6, 7, 8}
print(set1 & set2)  # Intersection: {4, 5}
print(set1 - set2)  # Difference: {1, 2, 3}
print(set1 ^ set2)  # Symmetric difference: {1, 2, 3, 6, 7, 8}

# Alternative syntax
print(set1.union(set2))
print(set1.intersection(set2))
print(set1.difference(set2))
print(set1.symmetric_difference(set2))

# Set relationships
print(set1.issubset(set2))      # False
print(set1.issuperset(set2))    # False
print(set1.isdisjoint(set2))    # False (they share elements)
```

### 5.5.3 Practical Set Examples

```python
# Remove duplicates from a list
def remove_duplicates(lst):
    return list(set(lst))

numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique = remove_duplicates(numbers)
print(unique)  # [1, 2, 3, 4] (order may vary)

# Find common elements between lists
list1 = ["apple", "banana", "cherry"]
list2 = ["banana", "cherry", "date"]
common = list(set(list1) & set(list2))
print(common)  # ['banana', 'cherry'] (order may vary)

# Check membership (very fast for sets)
large_set = set(range(1000000))
print(999999 in large_set)  # True (very fast operation)
```

---

## 5.6 Functions

**Functions** are reusable blocks of code that perform specific tasks. They're essential for organizing code and avoiding repetition.

### 5.6.1 Defining Functions

```python
# Basic function definition
def greet():
    print("Hello, World!")

# Call the function
greet()  # Output: Hello, World!

# Function with parameters
def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice")  # Output: Hello, Alice!

# Function with multiple parameters
def add_numbers(a, b):
    result = a + b
    return result

sum_result = add_numbers(5, 3)
print(sum_result)  # Output: 8
```

### 5.6.2 Function Parameters

```python
# Default parameters
def greet_with_title(name, title="Mr./Ms."):
    print(f"Hello, {title} {name}!")

greet_with_title("Smith")           # Hello, Mr./Ms. Smith!
greet_with_title("Johnson", "Dr.")  # Hello, Dr. Johnson!

# Keyword arguments
def create_profile(name, age, city="Unknown", occupation="Student"):
    return {
        "name": name,
        "age": age,
        "city": city,
        "occupation": occupation
    }

# Different ways to call the function
profile1 = create_profile("Alice", 25)
profile2 = create_profile("Bob", 30, city="New York")
profile3 = create_profile(name="Charlie", age=35, occupation="Engineer", city="London")

print(profile1)  # {'name': 'Alice', 'age': 25, 'city': 'Unknown', 'occupation': 'Student'}
print(profile2)  # {'name': 'Bob', 'age': 30, 'city': 'New York', 'occupation': 'Student'}
print(profile3)  # {'name': 'Charlie', 'age': 35, 'city': 'London', 'occupation': 'Engineer'}
```

### 5.6.3 Variable-Length Arguments

```python
# *args for variable number of positional arguments
def sum_all(*numbers):
    total = 0
    for num in numbers:
        total += num
    return total

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs for variable number of keyword arguments
def print_info(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Boston")
# Output:
# name: Alice
# age: 25
# city: Boston

# Combining different parameter types
def complex_function(required, default="default", *args, **kwargs):
    print(f"Required: {required}")
    print(f"Default: {default}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

complex_function("must_have", "custom", 1, 2, 3, extra="info", more="data")
```

### 5.6.4 Return Values

```python
# Single return value
def square(x):
    return x * x

# Multiple return values (returns a tuple)
def divide_with_remainder(dividend, divisor):
    quotient = dividend // divisor
    remainder = dividend % divisor
    return quotient, remainder

q, r = divide_with_remainder(17, 5)
print(f"17 ÷ 5 = {q} remainder {r}")  # 17 ÷ 5 = 3 remainder 2

# Conditional returns
def absolute_value(x):
    if x >= 0:
        return x
    else:
        return -x

print(absolute_value(-5))  # 5
print(absolute_value(3))   # 3

# Early return
def find_first_even(numbers):
    for num in numbers:
        if num % 2 == 0:
            return num
    return None  # No even number found

result = find_first_even([1, 3, 5, 8, 9])
print(result)  # 8
```

### 5.6.5 Scope and Local vs Global Variables

```python
# Global variable
global_var = "I'm global"

def demonstrate_scope():
    # Local variable
    local_var = "I'm local"
    print(f"Inside function: {global_var}")  # Can access global
    print(f"Inside function: {local_var}")   # Can access local

demonstrate_scope()
print(f"Outside function: {global_var}")    # Can access global
# print(f"Outside function: {local_var}")   # Error! Can't access local

# Modifying global variables
counter = 0

def increment():
    global counter  # Declare that we want to modify the global variable
    counter += 1

increment()
increment()
print(counter)  # 2
```

<!-- ### 5.6.6 Lambda Functions -->
<!---->
<!-- Lambda functions are small, anonymous functions: -->
<!---->
<!-- ```python -->
<!-- # Regular function -->
<!-- def square(x): -->
<!--     return x * x -->
<!---->
<!-- # Lambda equivalent -->
<!-- square_lambda = lambda x: x * x -->
<!---->
<!-- print(square(5))        # 25 -->
<!-- print(square_lambda(5)) # 25 -->
<!---->
<!-- # Lambda with multiple arguments -->
<!-- add = lambda x, y: x + y -->
<!-- print(add(3, 4))  # 7 -->
<!---->
<!-- # Lambdas are often used with built-in functions -->
<!-- numbers = [1, 2, 3, 4, 5] -->
<!---->
<!-- # Using lambda with map() -->
<!-- squares = list(map(lambda x: x**2, numbers)) -->
<!-- print(squares)  # [1, 4, 9, 16, 25] -->
<!---->
<!-- # Using lambda with filter() -->
<!-- evens = list(filter(lambda x: x % 2 == 0, numbers)) -->
<!-- print(evens)  # [2, 4] -->
<!---->
<!-- # Using lambda with sorted() -->
<!-- students = [("Alice", 85), ("Bob", 90), ("Charlie", 78)] -->
<!-- sorted_by_grade = sorted(students, key=lambda student: student[1]) -->
<!-- print(sorted_by_grade)  # [('Charlie', 78), ('Alice', 85), ('Bob', 90)] -->
<!-- ``` -->

---

## 5.7 Modules and Libraries

**Modules** are files containing Python code that can be imported and used in other programs. **Libraries** are collections of modules.

### 5.7.1 Importing Modules

```python
# Import entire module
import math
print(math.pi)        # 3.141592653589793
print(math.sqrt(16))  # 4.0

# Import specific functions
from math import pi, sqrt, sin
print(pi)       # 3.141592653589793
print(sqrt(25)) # 5.0
print(sin(pi/2)) # 1.0

# Import with alias
import math as m
print(m.cos(0))  # 1.0

# Import all (generally not recommended)
from math import *
print(factorial(5))  # 120
```

### 5.7.2 Common Built-in Modules

```python
# random module
import random

# Generate random numbers
print(random.randint(1, 10))      # Random integer between 1 and 10
print(random.random())            # Random float between 0 and 1
print(random.uniform(1.5, 10.5))  # Random float between 1.5 and 10.5

# Random choices
colors = ["red", "green", "blue", "yellow"]
print(random.choice(colors))      # Random element from list
print(random.sample(colors, 2))   # Random sample of 2 elements

# Shuffle list
numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)
print(numbers)  # Shuffled list

# datetime module
from datetime import datetime, date, timedelta

now = datetime.now()
print(now)  # Current date and time

today = date.today()
print(today)  # Current date

# Date arithmetic
tomorrow = today + timedelta(days=1)
print(tomorrow)

# os module
import os

print(os.getcwd())  # Current working directory
# os.mkdir("new_folder")  # Create directory
# os.listdir(".")  # List files in current directory
```

### 5.7.3 Creating Your Own Modules

Create a file called `ai_utils.py`:

```python
# ai_utils.py
"""
Utility functions for AI applications
"""

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    if len(point1) != len(point2):
        raise ValueError("Points must have same dimensions")

    sum_squares = sum((a - b)**2 for a, b in zip(point1, point2))
    return sum_squares ** 0.5

def manhattan_distance(point1, point2):
    """Calculate Manhattan distance between two points."""
    if len(point1) != len(point2):
        raise ValueError("Points must have same dimensions")

    return sum(abs(a - b) for a, b in zip(point1, point2))

def normalize_data(data):
    """Normalize a list of numbers to range [0, 1]."""
    if not data:
        return []

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        return [0.5] * len(data)  # All values are the same

    return [(x - min_val) / (max_val - min_val) for x in data]

# Constants
GOLDEN_RATIO = 1.618033988749
PI = 3.141592653589793
```

Now use your module in another file:

```python
# main.py
import ai_utils

# Test distance functions
p1 = (0, 0)
p2 = (3, 4)

euclidean_dist = ai_utils.euclidean_distance(p1, p2)
manhattan_dist = ai_utils.manhattan_distance(p1, p2)

print(f"Euclidean distance: {euclidean_dist}")  # 5.0
print(f"Manhattan distance: {manhattan_dist}")  # 7

# Test normalization
scores = [85, 92, 78, 96, 88]
normalized = ai_utils.normalize_data(scores)
print(f"Original: {scores}")
print(f"Normalized: {normalized}")

# Use constants
print(f"Golden ratio: {ai_utils.GOLDEN_RATIO}")
```

---

## 5.8 Practical AI Examples

### 5.8.1 Simple Recommendation System

```python
def recommend_movies(user_ratings, all_movies, num_recommendations=3):
    """
    Simple movie recommendation based on user ratings.
    Uses collaborative filtering concept.
    """
    # Calculate average rating for movies the user has rated
    rated_movies = list(user_ratings.keys())
    avg_user_rating = sum(user_ratings.values()) / len(user_ratings)

    # Find movies the user hasn't rated
    unrated_movies = [movie for movie in all_movies if movie not in rated_movies]

    # Simple recommendation: suggest highest-rated movies they haven't seen
    # In a real system, this would be much more sophisticated
    movie_scores = {}
    for movie in unrated_movies:
        # Simulate getting average rating for this movie
        # In reality, this would come from a database
        simulated_rating = hash(movie) % 50 / 10 + 5  # Random rating 5.0-10.0
        movie_scores[movie] = simulated_rating

    # Sort by score and return top recommendations
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, score in sorted_movies[:num_recommendations]]

# Example usage
user_ratings = {
    "The Matrix": 9,
    "Inception": 8,
    "Interstellar": 9,
    "The Godfather": 10
}

all_movies = [
    "The Matrix", "Inception", "Interstellar", "The Godfather",
    "Pulp Fiction", "The Dark Knight", "Fight Club", "Goodfellas",
    "The Shawshank Redemption", "Forrest Gump"
]

recommendations = recommend_movies(user_ratings, all_movies)
print("Recommended movies for you:")
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")
```

### 5.8.2 Simple Data Analysis

```python
def analyze_student_data(students):
    """
    Analyze student performance data.
    Demonstrates data processing with Python data structures.
    """
    if not students:
        return "No student data provided"

    # Extract all grades
    all_grades = []
    for student in students:
        all_grades.extend(student["grades"])

    # Calculate statistics
    total_students = len(students)
    total_grades = len(all_grades)
    average_grade = sum(all_grades) / total_grades
    highest_grade = max(all_grades)
    lowest_grade = min(all_grades)

    # Find top performer
    student_averages = {}
    for student in students:
        student_avg = sum(student["grades"]) / len(student["grades"])
        student_averages[student["name"]] = student_avg

    top_student = max(student_averages.items(), key=lambda x: x[1])

    # Grade distribution
    grade_ranges = {"A (90-100)": 0, "B (80-89)": 0, "C (70-79)": 0, "D (60-69)": 0, "F (0-59)": 0}
    for grade in all_grades:
        if grade >= 90:
            grade_ranges["A (90-100)"] += 1
        elif grade >= 80:
            grade_ranges["B (80-89)"] += 1
        elif grade >= 70:
            grade_ranges["C (70-79)"] += 1
        elif grade >= 60:
            grade_ranges["D (60-69)"] += 1
        else:
            grade_ranges["F (0-59)"] += 1

    # Return analysis
    return {
        "total_students": total_students,
        "total_grades": total_grades,
        "average_grade": round(average_grade, 2),
        "highest_grade": highest_grade,
        "lowest_grade": lowest_grade,
        "top_student": top_student,
        "grade_distribution": grade_ranges
    }

# Example data
students_data = [
    {"name": "Alice", "grades": [85, 92, 78, 96, 88]},
    {"name": "Bob", "grades": [79, 85, 91, 87, 83]},
    {"name": "Charlie", "grades": [92, 88, 94, 90, 89]},
    {"name": "Diana", "grades": [76, 82, 79, 85, 81]},
    {"name": "Eve", "grades": [95, 97, 93, 98, 96]}
]

analysis = analyze_student_data(students_data)
print("Student Performance Analysis:")
print(f"Total Students: {analysis['total_students']}")
print(f"Average Grade: {analysis['average_grade']}")
print(f"Highest Grade: {analysis['highest_grade']}")
print(f"Lowest Grade: {analysis['lowest_grade']}")
print(f"Top Student: {analysis['top_student'][0]} (Average: {analysis['top_student'][1]:.2f})")
print("\nGrade Distribution:")
for grade_range, count in analysis['grade_distribution'].items():
    print(f"  {grade_range}: {count} grades")
```

### 5.8.3 Simple Search Algorithm with Data Structures

```python
def build_word_index(documents):
    """
    Build an inverted index for simple text search.
    Returns a dictionary mapping words to document IDs.
    """
    index = {}

    for doc_id, document in enumerate(documents):
        # Simple tokenization (split by spaces and remove punctuation)
        words = document.lower().replace(",", "").replace(".", "").split()

        for word in words:
            if word not in index:
                index[word] = set()
            index[word].add(doc_id)

    return index

def search_documents(query, index, documents):
    """
    Search for documents containing query terms.
    """
    query_words = query.lower().split()

    if not query_words:
        return []

    # Find documents containing all query words (AND search)
    result_docs = None

    for word in query_words:
        if word in index:
            word_docs = index[word]
            if result_docs is None:
                result_docs = word_docs.copy()
            else:
                result_docs = result_docs.intersection(word_docs)
        else:
            # Word not found, no results
            return []

    if result_docs is None:
        return []

    # Return the actual documents
    return [(doc_id, documents[doc_id]) for doc_id in sorted(result_docs)]

# Example usage
documents = [
    "Artificial intelligence is transforming the world of technology.",
    "Machine learning algorithms can recognize patterns in data.",
    "Deep learning uses neural networks to solve complex problems.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information.",
    "Robotics combines AI with mechanical engineering for automation."
]

# Build search index
search_index = build_word_index(documents)

# Perform searches
queries = ["artificial intelligence", "learning", "computers language"]

for query in queries:
    print(f"\nSearch results for '{query}':")
    results = search_documents(query, search_index, documents)

    if results:
        for doc_id, document in results:
            print(f"  Document {doc_id}: {document}")
    else:
        print("  No results found.")
```

---

## 5.9 Exercises

### Exercise 1: Contact Book

Create a simple contact book using dictionaries and lists:

```python
# Global dictionary to store contacts
contacts = {}

def add_contact(name, phone, email):
    """Add a new contact."""
    # Your code here: store contact info in contacts dictionary
    pass

def find_contact(name):
    """Find and return a contact by name."""
    # Your code here: return contact info or "Not found"
    pass

def list_all_contacts():
    """Print all contacts."""
    # Your code here: print all contacts nicely
    pass

# Test your functions
add_contact("Alice", "123-456-7890", "alice@email.com")
add_contact("Bob", "987-654-3210", "bob@email.com")

print("All contacts:")
list_all_contacts()

print("\nFinding Alice:")
print(find_contact("Alice"))
```

### Exercise 2: Word Counter

Create a function that counts words in a text:

```python
def analyze_text(text):
    """
    Analyze text and return word statistics.
    Complete this function.
    """
    # Your code here:
    # 1. Split text into words (convert to lowercase)
    # 2. Count how many times each word appears
    # 3. Find the most common word
    # 4. Return a dictionary with results

    pass

# Test your function
sample_text = "Python is great. Python is easy. Programming with Python is fun."
result = analyze_text(sample_text)
print(result)

# Expected output should show word counts and most common word
```

### Exercise 3: Simple Calculator Functions

Create calculator functions using what you've learned:

```python
def add_list(numbers):
    """Add all numbers in a list."""
    # Your code here
    pass

def find_average(numbers):
    """Find average of numbers in a list."""
    # Your code here
    pass

def find_max_min(numbers):
    """Return both max and min values as a tuple."""
    # Your code here
    pass

# Test your functions
test_numbers = [10, 5, 8, 20, 3, 15]

print(f"Numbers: {test_numbers}")
print(f"Sum: {add_list(test_numbers)}")
print(f"Average: {find_average(test_numbers)}")
print(f"Max and Min: {find_max_min(test_numbers)}")
```

---

## Key Takeaways

1. **Lists** are versatile, mutable sequences perfect for ordered data that changes.
2. **Tuples** are immutable sequences ideal for fixed data like coordinates or records.
3. **Dictionaries** provide fast key-value lookups, essential for mappings and structured data.
4. **Sets** efficiently handle unique elements and set operations.
5. **Functions** organize code into reusable, testable units.
6. **Modules** extend Python's capabilities and promote code reuse.
7. **Choosing the right data structure** significantly impacts performance and code clarity.
8. **List/dictionary comprehensions** provide concise ways to create and transform data.

---

> Next week, we'll explore Object-Oriented Programming in Python - the foundation for building complex AI systems with classes and objects!
