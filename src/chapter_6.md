# Chapter 6: Object-Oriented Programming in Python

## Learning Objectives

By the end of this chapter, you will:

- Understand the fundamental concepts of Object-Oriented Programming (OOP).
- Create and use classes and objects in Python.
- Implement methods and attributes effectively.
- Apply inheritance to create hierarchical relationships between classes.
- Build practical AI-related classes for real-world applications.

---

## Requirements

- Completion of Chapters 4 and 5 (Python basics and data structures).
- Python 3.8 or later installed on your system.
- A text editor or IDE for writing Python code.

## 6.1 What is Object-Oriented Programming?

**Object-Oriented Programming (OOP)** is a programming paradigm that organizes code around **objects** rather than functions. Think of objects as real-world entities that have:

- **Attributes** (characteristics or properties)
- **Methods** (actions or behaviors)

### Why OOP for AI?

OOP is particularly valuable in AI development because:

- **Modularity**: Break complex AI systems into manageable components.
- **Reusability**: Create reusable AI components (agents, neural networks, search algorithms).
- **Organization**: Structure large AI projects with clear relationships between components.
- **Abstraction**: Hide complex implementation details behind simple interfaces.

### Real-World Analogy

Think of a **car**:

- **Attributes**: color, model, speed, fuel_level
- **Methods**: start(), stop(), accelerate(), brake()

In AI, we might have an **AI Agent**:

- **Attributes**: position, knowledge_base, goals
- **Methods**: perceive(), think(), act()

---

## 6.2 Classes and Objects

### 6.2.1 Understanding Classes

A **class** is a blueprint or template for creating objects. It defines what attributes and methods objects of that type will have.

```python
# Basic class definition
class Car:
    pass  # Empty class for now

# Creating objects (instances) from the class
my_car = Car()
your_car = Car()

print(type(my_car))  # <class '__main__.Car'>
print(my_car)        # <__main__.Car object at 0x...>
```

### 6.2.2 Adding Attributes

Let's add some attributes to our Car class:

```python
class Car:
    def __init__(self, make, model, year):
        """
        Constructor method - called when creating a new object.
        'self' refers to the instance being created.
        """
        self.make = make      # Instance attribute
        self.model = model    # Instance attribute
        self.year = year      # Instance attribute
        self.speed = 0        # Default value
        self.is_running = False

# Creating car objects with specific attributes
car1 = Car("Toyota", "Camry", 2022)
car2 = Car("Honda", "Civic", 2021)

# Accessing attributes
print(f"Car 1: {car1.year} {car1.make} {car1.model}")
print(f"Car 2: {car2.year} {car2.make} {car2.model}")
print(f"Car 1 speed: {car1.speed}")
```

**Output:**

```
Car 1: 2022 Toyota Camry
Car 2: 2021 Honda Civic
Car 1 speed: 0
```

### 6.2.3 The `__init__` Method

The `__init__` method is a special method called a **constructor**. It's automatically called when you create a new object.

```python
class Student:
    def __init__(self, name, age, major):
        self.name = name
        self.age = age
        self.major = major
        self.grades = []  # Empty list to store grades
        self.gpa = 0.0

# Creating student objects
alice = Student("Alice Johnson", 20, "Computer Science")
bob = Student("Bob Smith", 19, "Mathematics")

print(f"Student: {alice.name}, Age: {alice.age}, Major: {alice.major}")
```

---

## 6.3 Methods and Attributes

### 6.3.1 Instance Methods

Methods are functions that belong to a class and can access the object's attributes.

```python
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.speed = 0
        self.is_running = False

    def start(self):
        """Start the car."""
        if not self.is_running:
            self.is_running = True
            print(f"The {self.make} {self.model} is now running.")
        else:
            print(f"The {self.make} {self.model} is already running.")

    def stop(self):
        """Stop the car."""
        if self.is_running:
            self.is_running = False
            self.speed = 0
            print(f"The {self.make} {self.model} has stopped.")
        else:
            print(f"The {self.make} {self.model} is already stopped.")

    def accelerate(self, amount):
        """Increase the car's speed."""
        if self.is_running:
            self.speed += amount
            print(f"Accelerating... Current speed: {self.speed} km/h")
        else:
            print("Cannot accelerate. Car is not running.")

    def get_info(self):
        """Return information about the car."""
        status = "running" if self.is_running else "stopped"
        return f"{self.year} {self.make} {self.model} - Status: {status}, Speed: {self.speed} km/h"

# Using the car
my_car = Car("Tesla", "Model 3", 2023)
print(my_car.get_info())

my_car.start()
my_car.accelerate(50)
my_car.accelerate(30)
print(my_car.get_info())
my_car.stop()
```

**Output:**

```
2023 Tesla Model 3 - Status: stopped, Speed: 0 km/h
The Tesla Model 3 is now running.
Accelerating... Current speed: 50 km/h
Accelerating... Current speed: 80 km/h
2023 Tesla Model 3 - Status: running, Speed: 80 km/h
The Tesla Model 3 has stopped.
```

### 6.3.2 AI Example: Simple Agent Class

Let's create a basic AI agent class:

```python
class AIAgent:
    def __init__(self, name, position_x=0, position_y=0):
        self.name = name
        self.position_x = position_x
        self.position_y = position_y
        self.energy = 100
        self.knowledge = []

    def move(self, dx, dy):
        """Move the agent by dx, dy units."""
        if self.energy >= 10:
            self.position_x += dx
            self.position_y += dy
            self.energy -= 10
            print(f"{self.name} moved to ({self.position_x}, {self.position_y})")
            print(f"Energy remaining: {self.energy}")
        else:
            print(f"{self.name} is too tired to move!")

    def learn(self, fact):
        """Add new knowledge to the agent."""
        self.knowledge.append(fact)
        print(f"{self.name} learned: {fact}")

    def rest(self):
        """Restore energy."""
        self.energy = min(100, self.energy + 20)
        print(f"{self.name} rested. Energy: {self.energy}")

    def get_status(self):
        """Return the agent's current status."""
        return {
            'name': self.name,
            'position': (self.position_x, self.position_y),
            'energy': self.energy,
            'knowledge_count': len(self.knowledge)
        }

# Create and use an AI agent
robot = AIAgent("R2D2", 0, 0)
print(robot.get_status())

robot.move(3, 4)
robot.learn("The sky is blue")
robot.learn("Water boils at 100°C")
robot.move(2, -1)
robot.rest()
print(robot.get_status())
```

### 6.3.3 Class Attributes vs Instance Attributes

```python
class Robot:
    # Class attribute - shared by all instances
    robot_count = 0

    def __init__(self, name):
        # Instance attributes - unique to each instance
        self.name = name
        self.battery = 100
        Robot.robot_count += 1  # Increment class attribute

    def get_robot_count(self):
        """Return the total number of robots created."""
        return Robot.robot_count

# Create robots
robot1 = Robot("Alpha")
robot2 = Robot("Beta")
robot3 = Robot("Gamma")

print(f"Total robots created: {Robot.robot_count}")
print(f"Robot 1 name: {robot1.name}")
print(f"Robot count from instance: {robot1.get_robot_count()}")
```

**Output:**

```
Total robots created: 3
Robot 1 name: Alpha
Robot count from instance: 3
```

---

## 6.4 Inheritance

**Inheritance** allows you to create new classes based on existing classes. The new class (child/subclass) inherits attributes and methods from the parent class (superclass).

### 6.4.1 Basic Inheritance

```python
# Parent class (Base class)
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.energy = 100

    def eat(self):
        self.energy += 20
        print(f"{self.name} is eating. Energy: {self.energy}")

    def sleep(self):
        self.energy += 30
        print(f"{self.name} is sleeping. Energy: {self.energy}")

    def make_sound(self):
        print(f"{self.name} makes a sound.")

# Child class (Derived class)
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Canine")  # Call parent constructor
        self.breed = breed
        self.loyalty = 100

    def make_sound(self):  # Override parent method
        print(f"{self.name} barks: Woof! Woof!")

    def fetch(self):  # New method specific to Dog
        if self.energy >= 15:
            self.energy -= 15
            self.loyalty += 5
            print(f"{self.name} fetches the ball! Loyalty: {self.loyalty}")
        else:
            print(f"{self.name} is too tired to fetch.")

class Cat(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Feline")
        self.breed = breed
        self.independence = 80

    def make_sound(self):  # Override parent method
        print(f"{self.name} meows: Meow!")

    def hunt(self):  # New method specific to Cat
        if self.energy >= 20:
            self.energy -= 20
            print(f"{self.name} is hunting mice.")
        else:
            print(f"{self.name} is too tired to hunt.")

# Using inheritance
my_dog = Dog("Buddy", "Golden Retriever")
my_cat = Cat("Whiskers", "Persian")

# Methods inherited from Animal
my_dog.eat()
my_cat.sleep()

# Overridden methods
my_dog.make_sound()  # Calls Dog's version
my_cat.make_sound()  # Calls Cat's version

# Class-specific methods
my_dog.fetch()
my_cat.hunt()
```

---

## 6.5 Exercises

### Smart Home Device

Create a class hierarchy for smart home devices:

```python
class SmartDevice:
    """Base class for smart home devices."""

    def __init__(self, name, location):
        # Your code here
        pass

    def turn_on(self):
        # Your code here
        pass

    def turn_off(self):
        # Your code here
        pass

    def get_status(self):
        # Your code here
        pass

class SmartLight(SmartDevice):
    """Smart light that can change brightness and color."""

    def __init__(self, name, location):
        # Your code here - call parent constructor
        # Add brightness (0-100) and color attributes
        pass

    def set_brightness(self, level):
        # Your code here - validate level is 0-100
        pass

    def set_color(self, color):
        # Your code here
        pass

class SmartThermostat(SmartDevice):
    """Smart thermostat for temperature control."""

    def __init__(self, name, location):
        # Your code here - call parent constructor
        # Add current_temp and target_temp attributes
        pass

    def set_temperature(self, temp):
        # Your code here
        pass

    def get_temperature_info(self):
        # Your code here - return current and target temps
        pass

# Test your classes
living_room_light = SmartLight("Main Light", "Living Room")
bedroom_thermostat = SmartThermostat("Bedroom Thermostat", "Bedroom")

# Test the functionality
living_room_light.turn_on()
living_room_light.set_brightness(75)
living_room_light.set_color("warm white")
print(living_room_light.get_status())
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

## Key Takeaways

1. **Classes are blueprints** for creating objects with shared attributes and methods.
2. **Objects are instances** of classes with their own specific attribute values.
3. **Inheritance allows code reuse** and creates hierarchical relationships between classes.
4. **Method overriding** lets subclasses provide specific implementations of parent methods.
5. **OOP organizes complex AI systems** into manageable, reusable components.
6. **The `self` parameter** refers to the current instance and is required in all instance methods.
7. **Constructor (`__init__`)** initializes new objects with their starting state.

---

> Next week, we'll use these OOP concepts to build sophisticated AI search algorithms and problem-solving systems!
