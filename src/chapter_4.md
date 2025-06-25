# Lab 4: Getting Started with Python

## Learning Objectives

By the end of this chapter, you will:

- Understand what Python is and why it's essential for AI development.
- Set up and run Python on your computer.
- Master Python basics: variables, data types, and operators.
- Use input/output operations and control structures effectively.
- Write and execute Python scripts confidently.

---

## Requirements

- A computer with a modern operating system (Windows, macOS, or Linux).
- A text editor or IDE (e.g., VS Code, PyCharm, or IDLE).
- Python 3.8 or later installed on your system.

## 4.1 What is Python?

Python is a **high-level, interpreted programming language** known for its simplicity and readability. Created by Guido van Rossum in 1991, Python has become one of the most popular programming languages in the world, especially in the field of **Artificial Intelligence**.

### Why Python for AI?

Python dominates AI development for several reasons:

- **Simplicity**: Clean, readable syntax that lets you focus on problem-solving rather than complex syntax.
- **Rich Ecosystem**: Extensive libraries for AI (NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch).
- **Versatility**: Suitable for data analysis, machine learning, web development, and more.
- **Community Support**: Large, active community with abundant resources and documentation.
- **Rapid Prototyping**: Quick to write and test ideas, essential for AI research and development.

### Python in AI Applications:

- **Machine Learning**: Building predictive models and pattern recognition systems.
- **Data Analysis**: Processing and analyzing large datasets.
- **Natural Language Processing**: Understanding and generating human language.
- **Computer Vision**: Image and video analysis.
- **Robotics**: Controlling and programming robotic systems.

---

## 4.2 Setting Up Python

### 4.2.1 Installing Python

<div style="display: flex; justify-content: start; align-items: end;">
<h4>Option 1: Official Python Installation</h4>
  <img src='./assets/python.png' height='48'>
</div>

1. **Download Python**:

   - Visit [python.org](https://www.python.org/downloads/).
   - Download Python 3.8 or later for your operating system.

2. **Install Python**:

   - **Windows**: Run the installer and **check "Add Python to PATH"**.
   - **macOS**: Run the installer package.
   - **Linux**: Python is usually pre-installed. If not, use your package manager:
     ```bash
     sudo apt update
     sudo apt install python3 python3-pip
     ```

3. **Verify Installation**:
   Open your terminal/command prompt and type:
   ```bash
   python --version
   ```
   or
   ```bash
   python3 --version
   ```
   You should see something like: `Python 3.11.5`

### 4.2.2 Python Development Environment

You have several options for writing Python code:

1. **IDLE** (comes with Python): Simple, good for beginners.
2. **VS Code**: Popular, feature-rich editor with Python extensions.
3. **PyCharm**: Professional IDE specifically for Python.
4. **Jupyter Notebook**: Great for data analysis and experimentation.

For this course, we recommend **VS Code** with the Python extension.

---

## 4.3 Python Basics: Variables, Data Types, and Operators

### 4.3.1 Your First Python Program

Create a file called `hello.py` and write:

```python
print("Hello, AI World!")
```

Run it by typing in your terminal:

```bash
python hello.py
```

### 4.3.2 Variables

Variables in Python are used to store data. Unlike some languages, you don't need to declare the type explicitly.

```python
# Variable assignment
name = "Alice"
age = 25
height = 5.6
is_student = True

print(name)    # Output: Alice
print(age)     # Output: 25
print(height)  # Output: 5.6
print(is_student)  # Output: True
```

#### Variable Naming Rules:

- Must start with a letter or underscore
- Can contain letters, numbers, and underscores
- Case-sensitive (`age` and `Age` are different)
- Cannot use Python keywords (`if`, `for`, `class`, etc.)

```python
# Good variable names
student_name = "John"
total_score = 95
is_valid = True

# Bad variable names (avoid these)
# 2name = "John"      # Starts with number
# class = "Math"      # Python keyword
# student-name = "John"  # Contains hyphen
```

### 4.3.3 Data Types

Python has several built-in data types:

#### 1. Numbers

```python
# Integers
count = 42
negative_num = -17

# Floats (decimal numbers)
pi = 3.14159
temperature = -2.5

# You can check the type
print(type(count))      # <class 'int'>
print(type(pi))         # <class 'float'>
```

#### 2. Strings

```python
# String creation
message = "Hello, World!"
name = 'Python'
multiline = """This is a
multiline string"""

# String operations
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name  # Concatenation
print(full_name)  # Output: John Doe

# String methods
text = "artificial intelligence"
print(text.upper())      # ARTIFICIAL INTELLIGENCE
print(text.capitalize()) # Artificial intelligence
print(len(text))         # 22 (length of string)
```

#### 3. Booleans

```python
is_raining = True
is_sunny = False

# Boolean operations
print(not is_raining)    # False
print(is_raining and is_sunny)  # False
print(is_raining or is_sunny)   # True
```

#### 4. None Type

```python
result = None  # Represents absence of value
print(result)  # Output: None
```

### 4.3.4 Operators

#### Arithmetic Operators

```python
a = 10
b = 3

print(a + b)   # Addition: 13
print(a - b)   # Subtraction: 7
print(a * b)   # Multiplication: 30
print(a / b)   # Division: 3.3333...
print(a // b)  # Floor division: 3
print(a % b)   # Modulus (remainder): 1
print(a ** b)  # Exponentiation: 1000
```

#### Comparison Operators

```python
x = 5
y = 10

print(x == y)  # Equal: False
print(x != y)  # Not equal: True
print(x < y)   # Less than: True
print(x > y)   # Greater than: False
print(x <= y)  # Less than or equal: True
print(x >= y)  # Greater than or equal: False
```

#### Logical Operators

```python
p = True
q = False

print(p and q)  # Logical AND: False
print(p or q)   # Logical OR: True
print(not p)    # Logical NOT: False
```

#### Assignment Operators

```python
score = 100

score += 10   # Same as: score = score + 10
print(score)  # 110

score -= 5    # Same as: score = score - 5
print(score)  # 105

score *= 2    # Same as: score = score * 2
print(score)  # 210
```

---

## 4.4 Input/Output and Basic Control Structures

### 4.4.1 Input and Output

#### Output with print()

```python
print("Hello, World!")
print("The answer is", 42)
print("Name:", "Alice", "Age:", 25)

# Formatted output
name = "Bob"
age = 30
print(f"My name is {name} and I am {age} years old.")
```

#### Input with input()

```python
# Getting user input (always returns a string)
name = input("Enter your name: ")
print(f"Hello, {name}!")

# Converting input to numbers
age_str = input("Enter your age: ")
age = int(age_str)  # Convert string to integer
print(f"You are {age} years old.")

# Or in one line
height = float(input("Enter your height in meters: "))
print(f"Your height is {height} meters.")
```

### 4.4.2 Control Structures

#### If-Else Statements

```python
# Basic if statement
temperature = 25

if temperature > 30:
    print("It's hot!")
elif temperature > 20:
    print("It's warm.")
elif temperature > 10:
    print("It's cool.")
else:
    print("It's cold!")
```

#### Practical Example: Grade Calculator

```python
score = int(input("Enter your score (0-100): "))

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Your grade is: {grade}")
```

#### Loops

##### For Loops

```python
# Loop through a range of numbers
for i in range(5):
    print(f"Count: {i}")
# Output: Count: 0, Count: 1, Count: 2, Count: 3, Count: 4

# Loop through a range with start and end
for i in range(1, 6):
    print(f"Number: {i}")
# Output: Number: 1, Number: 2, Number: 3, Number: 4, Number: 5

# Loop through a string
word = "Python"
for letter in word:
    print(letter)
# Output: P, y, t, h, o, n (each on a new line)
```

##### While Loops

```python
# Basic while loop
count = 0
while count < 5:
    print(f"Count is: {count}")
    count += 1

# User input validation
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input.lower() == 'quit':
        break
    print(f"You entered: {user_input}")
```

---

## 4.5 Writing and Running Python Scripts

### 4.5.1 Creating Your First AI-Related Script

Let's create a simple script that demonstrates basic AI concepts:

**File: `simple_ai_demo.py`**

```python
"""
Simple AI Demo - Basic Decision Making
This script demonstrates basic decision-making logic
that could be used in AI systems.
"""

def analyze_weather(temperature, humidity, wind_speed):
    """
    Analyze weather conditions and provide recommendations.
    This is a simple rule-based system similar to expert systems in AI.
    """
    print("=== Weather Analysis System ===")
    print(f"Temperature: {temperature}°C")
    print(f"Humidity: {humidity}%")
    print(f"Wind Speed: {wind_speed} km/h")
    print()

    # Decision logic
    recommendations = []

    if temperature > 30:
        recommendations.append("It's hot - stay hydrated!")
    elif temperature < 10:
        recommendations.append("It's cold - dress warmly!")

    if humidity > 80:
        recommendations.append("High humidity - expect muggy conditions")
    elif humidity < 30:
        recommendations.append("Low humidity - moisturize your skin")

    if wind_speed > 25:
        recommendations.append("Windy conditions - secure loose objects")

    # Output recommendations
    if recommendations:
        print("Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("Weather conditions are normal - enjoy your day!")

def main():
    """Main function to run the weather analysis."""
    print("Welcome to the AI Weather Advisor!")
    print()

    try:
        # Get user input
        temp = float(input("Enter temperature (°C): "))
        humidity = float(input("Enter humidity (%): "))
        wind = float(input("Enter wind speed (km/h): "))

        # Validate input
        if not (0 <= humidity <= 100):
            print("Warning: Humidity should be between 0-100%")

        # Analyze weather
        analyze_weather(temp, humidity, wind)

    except ValueError:
        print("Error: Please enter valid numbers!")

# Run the program
if __name__ == "__main__":
    main()
```

### 4.5.2 Running Python Scripts

There are several ways to run Python scripts:

1. **Command Line**:

   ```bash
   python simple_ai_demo.py
   ```

2. **In VS Code**: Press F5 or use the Run button.

3. **In IDLE**: Open the file and press F5.

### 4.5.3 Python Interactive Mode

You can also run Python interactively:

```bash
python
```

This opens the Python interpreter where you can type commands directly:

```python
>>> print("Hello, AI!")
Hello, AI!
>>> x = 5
>>> y = 10
>>> print(x + y)
15
>>> exit()
```

---

## 4.6 Practical Examples

### 4.6.1 Simple Pattern Recognition

```python
def recognize_pattern(sequence):
    """
    Simple pattern recognition - detect if a sequence is arithmetic.
    This demonstrates basic pattern analysis used in AI.
    """
    if len(sequence) < 3:
        return "Sequence too short to determine pattern"

    # Check for arithmetic progression
    diff = sequence[1] - sequence[0]
    for i in range(2, len(sequence)):
        if sequence[i] - sequence[i-1] != diff:
            return "No arithmetic pattern detected"

    return f"Arithmetic sequence with common difference: {diff}"

# Test the function
numbers = [2, 5, 8, 11, 14]
result = recognize_pattern(numbers)
print(f"Sequence: {numbers}")
print(f"Pattern: {result}")
```

### 4.6.2 Basic Search Algorithm

```python
def linear_search(data, target):
    """
    Linear search algorithm - fundamental to many AI search techniques.
    Returns the index of the target if found, -1 otherwise.
    """
    for i in range(len(data)):
        if data[i] == target:
            return i
    return -1

# Test the search
cities = ["New York", "London", "Tokyo", "Paris", "Sydney"]
search_city = "Tokyo"
index = linear_search(cities, search_city)

if index != -1:
    print(f"Found '{search_city}' at position {index}")
else:
    print(f"'{search_city}' not found")
```

---

## 4.7 Exercises

### Exercise : Simple calculator

Create a simple calculator that can perform basic arithmetic operations:

```python
def calculator():
    """
    Simple AI assistant for basic calculations.
    Complete this function to handle different operations.
    """
    print("Hello! I'm your AI assistant.")
    print("I can help you with basic math operations.")

    while True:
        print("\nWhat would you like to do?")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '5':
            print("Goodbye!")
            break
        elif choice in ['1', '2', '3', '4']:
            # Your code here: get two numbers and perform the operation
            pass
        else:
            print("Invalid choice. Please try again.")

# Test your assistant
calculator()
```

---

## Key Takeaways

1. **Python's simplicity** makes it ideal for AI development and rapid prototyping.
2. **Variables and data types** are the building blocks of any program.
3. **Control structures** enable decision-making and repetition in your programs.
4. **Input/output operations** allow interaction with users and external systems.
5. **Practice is essential** - the more you code, the more comfortable you'll become.

---

> Next week, we'll dive deeper into Python's data structures and functions - the tools that will power your AI applications!
