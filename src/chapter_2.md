# Chapter 2: Prolog Fundamentals

## Learning Objectives

By the end of this chapter, you will:

- Understand the basic building blocks of Prolog: variables, atoms, and terms.
- Learn how Prolog matches patterns and performs unification.
- Use recursion to define reusable rules.
- Work with lists and perform common list operations.
- Perform basic arithmetic in Prolog.
- Build a small knowledge base and solve simple problems.

---

## 2.1 Variables, Atoms, and Terms

Prolog programs are built using **facts**, **rules**, and **queries**. To understand how Prolog works, you need to know its basic building blocks: **atoms**, **variables**, and **terms**.

---

### 2.1.1 Atoms

Atoms are **constants** in Prolog. They represent fixed values like names, labels, or symbols. Atoms are used to represent things that don’t change, such as names of people, objects, or relationships.

#### Examples of Atoms:

- Lowercase words: `john`, `mary`, `cat`.
- Words enclosed in single quotes: `'New York'`, `'hello world'`.
- Special characters: `+`, `-`, `@`, `#`.

#### Example:

```prolog
likes(john, pizza). % "john" and "pizza" are atoms
```

Here, `likes` is a relationship, and `john` and `pizza` are constants (atoms).

---

### 2.1.2 Variables

Variables are **placeholders** that can represent any value. In Prolog, variables always start with an **uppercase letter** or an **underscore** (`_`).

#### Examples of Variables:

- `X`, `Y`, `Person`, `_Result`.

#### Example:

```prolog
likes(john, X). % X is a variable that can match any value
```

Here, `X` can represent anything that `john` likes.

---

### 2.1.3 Terms

A **term** is the most general building block in Prolog. It can be:

1. An **atom** (e.g., `john`).
2. A **variable** (e.g., `X`).
3. A **compound term**: a structure with a functor and arguments (e.g., `parent(john, mary)`).

#### Example of a Compound Term:

```prolog
parent(john, mary). % "parent" is the functor, "john" and "mary" are arguments
```

Here, `parent` is the relationship (functor), and `john` and `mary` are the arguments.

---

## 2.2 Pattern Matching and Unification

Prolog’s core mechanism is **unification**, which matches terms to find solutions. This is how Prolog answers queries.

---

### 2.2.1 How Unification Works

Unification is the process of matching two terms. Here’s how it works:

1. **Two atoms unify** if they are the same.
2. **A variable unifies** with any term and takes its value.
3. **Two compound terms unify** if their functors and arguments match.

#### Examples:

```prolog
?- X = john. % A variable unifies with an atom
X = john.

?- parent(john, mary) = parent(john, Y). % A variable unifies with a term
Y = mary.

?- likes(john, pizza) = likes(john, X). % Matching compound terms
X = pizza.
```

---

### 2.2.2 When Unification Fails

Unification fails when:

1. Atoms are different.
2. Compound terms have different functors or a different number of arguments.

#### Example:

```prolog
?- parent(john, mary) = parent(john, alice).
false. % Different arguments

?- parent(john, mary) = likes(john, mary).
false. % Different functors
```

---

## 2.3 Recursion in Prolog

Recursion is a way to define rules that apply repeatedly. In Prolog, recursion is used to process lists, traverse relationships, and solve problems.

---

### Example: Defining Ancestors

Let’s define a rule for finding ancestors. An ancestor is someone who is either a parent or a parent of an ancestor.

#### Rule:

```prolog
% Base case: A parent is an ancestor
ancestor(X, Y) :- parent(X, Y).

% Recursive case: If X is a parent of Z, and Z is an ancestor of Y, then X is an ancestor of Y
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

#### Query:

```prolog
?- ancestor(john, alice).
true.
```

Explanation:

1. Prolog checks if `john` is a parent of `alice`. If not, it checks if `john` is a parent of someone (`Z`) who is an ancestor of `alice`.
2. This process repeats until Prolog finds a match or determines there is no solution.

---

## 2.4 Lists and List Operations

Lists are a fundamental data structure in Prolog. They are used to store multiple items in a single structure.

---

### 2.4.1 List Syntax

- An empty list: `[]`.
- A list with elements: `[a, b, c]`.
- A list with a head and tail: `[Head | Tail]`.

#### Example:

```prolog
?- [H | T] = [1, 2, 3].
H = 1, % The first element (head)
T = [2, 3]. % The rest of the list (tail)
```

---

### 2.4.2 Common List Operations

#### 1. **Check Membership** (`member/2`)

The `member/2` predicate checks if an element is in a list.

```prolog
?- member(2, [1, 2, 3]).
true.
```

#### 2. **Concatenate Lists** (`append/3`)

The `append/3` predicate combines two lists into one.

```prolog
?- append([1, 2], [3, 4], Result).
Result = [1, 2, 3, 4].
```

#### 3. **Reverse a List** (`reverse/2`)

The `reverse/2` predicate reverses the order of elements in a list.

```prolog
?- reverse([1, 2, 3], Result).
Result = [3, 2, 1].
```

---

## 2.5 Arithmetic in Prolog

Prolog can perform basic arithmetic operations like addition, subtraction, multiplication, and division.

---

### 2.5.1 Arithmetic Operators

- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Modulus: `mod`

#### Example:

```prolog
?- X is 2 + 3.
X = 5.

?- Y is 10 / 2.
Y = 5.0.
```

---

### 2.5.2 Using Arithmetic in Rules

You can use arithmetic to define rules.

#### Example: Calculating the Square of a Number

```prolog
square(X, Y) :- Y is X * X.
```

#### Query:

```prolog
?- square(4, Result).
Result = 16.
```

---

## 2.6 Building a Small Knowledge Base

Let’s create a small knowledge base to represent Animals Kingdom relationships and solve simple problems.

### Example Knowledge Base

Create a new file called animals.pl and add the following facts and rules:

```prolog
% Facts about animals
animal(dog).
animal(cat).
animal(bird).
animal(fish).
animal(elephant).

% Facts about what animals eat
eats(dog, meat).
eats(cat, meat).
eats(bird, seeds).
eats(fish, plankton).
eats(elephant, plants).

% Facts about where animals live
lives_in(dog, house).
lives_in(cat, house).
lives_in(bird, tree).
lives_in(fish, water).
lives_in(elephant, savanna).

% Facts about animal sizes
size(dog, medium).
size(cat, small).
size(bird, small).
size(fish, small).
size(elephant, large).
```

Now write rules to answer these questions:

1. **Write a rule for "carnivore"** (animals that eat meat):

```
____________________________________________________________________________
```

Test it:

```
____________________________________________________________________________
```

2. **Write a rule for "pet"** (animals that live in a house):

```
____________________________________________________________________________
```

Test it:

```
____________________________________________________________________________
```

---

> **Next Chapter Preview:** In Chapter 3, we’ll explore advanced Prolog concepts like backtracking, the Cut operator, and solving more complex problems like puzzles and games.
