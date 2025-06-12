# Chapter 1: Getting Started with Prolog

## Learning Objectives

By the end of this chapter, you will:

- Understand what Prolog is and why it’s used in AI.
- Set up and run Prolog on your computer.
- Write simple Prolog programs using facts, rules, and queries.
- Solve basic reasoning problems using Prolog.

---

## Requirements

- A computer with a modern operating system (Windows, macOS, or Linux).
- A text editor (e.g., Notepad, VS Code).
- A Prolog interpreter (e.g., SWI-Prolog, GNU Prolog).

## 1.1 What is Prolog?

Prolog (short for **PROgramming in LOGic**) is a **declarative programming language**. Unlike traditional programming languages like Python or Java, where you tell the computer _how_ to solve a problem step by step, in Prolog, you describe _what_ the problem is and let the system figure out the solution.

Prolog is widely used in **Artificial Intelligence** for tasks like:

- **Knowledge Representation**: Storing facts and relationships about the world.
- **Reasoning**: Answering questions based on those facts and relationships.
- **Problem Solving**: Solving puzzles, games, and logical problems.

### Key Features of Prolog:

- **Logic-based**: Prolog is based on formal logic, specifically predicate logic.
- **Declarative**: You define facts and rules, and Prolog uses them to infer answers.
- **Backtracking**: Prolog automatically explores different possibilities to find solutions.

---

## 1.2 Setting Up Prolog

To start using Prolog, you need to install a Prolog interpreter. The most popular options are:

- **SWI-Prolog** (recommended for beginners): Free and widely used.
- **GNU Prolog**: Another free option.

### Installation Steps (SWI-Prolog):

1. **Download SWI-Prolog**:

   - Visit [SWI-Prolog's website](https://www.swi-prolog.org/).
   - Download the version for your operating system (Windows, macOS, or Linux).

2. **Install SWI-Prolog**:

   - Follow the installation instructions for your operating system.

3. **Run SWI-Prolog**:
   - Open the Prolog interpreter by typing `swipl` in your terminal or launching the application.

---

## 1.3 Basic Syntax and Structure of Prolog Programs

Prolog programs consist of **facts**, **rules**, and **queries**.

### 1.3.1 Facts

Facts represent **things that are true**. They are the simplest building blocks of a Prolog program.

#### Syntax:

```prolog
fact_name(argument1, argument2, ...).
```

#### Example:

```prolog
parent(john, mary).  % John is a parent of Mary
parent(mary, alice). % Mary is a parent of Alice
```

Here, `parent/2` is a fact with two arguments: the parent and the child.

---

### 1.3.2 Rules

Rules define **relationships** between facts. They allow Prolog to infer new information.

#### Syntax:

```prolog
rule_name(argument1, argument2, ...) :- condition1, condition2, ... .
```

The `:-` symbol means "if." A rule is true if its conditions are true.

#### Example:

```prolog
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
```

This rule says: "X is a grandparent of Y if X is a parent of Z and Z is a parent of Y."

---

### 1.3.3 Queries

Queries are **questions** you ask Prolog. Prolog tries to answer them based on the facts and rules in your program.

#### Syntax:

```prolog
?- query_name(argument1, argument2, ...).
```

#### Example:

```prolog
?- parent(john, mary).
```

Prolog will respond:

```
true.
```

You can also ask more complex questions:

```prolog
?- grandparent(john, alice).
```

Prolog will respond:

```
true.
```

---

## 1.4 Writing and Running Your First Prolog Program

Let’s create a simple Prolog program to represent family relationships and use it to answer questions.

### Step 1: Create a File

1. Open a text editor (e.g., Notepad, VS Code).
2. Save the file as `family.pl`.

### Step 2: Add Facts and Rules

Write the following code in the file:

```prolog
% Facts
parent(john, mary).
parent(mary, alice).
parent(mary, bob).
parent(bob, charlie).

% Rules
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
```

Here’s what this program represents:

- **Facts**: John is Mary’s parent, Mary is Alice’s and Bob’s parent, and Bob is Charlie’s parent.
- **Rule**: X is a grandparent of Y if X is a parent of Z and Z is a parent of Y.

### Step 3: Load the Program in Prolog

1. Open SWI-Prolog.
2. Load your program by typing:
   ```prolog
   ?- [family].
   ```
   If the program loads successfully, Prolog will respond with:
   ```
   true.
   ```

### Step 4: Ask Questions (Queries)

Now that your program is loaded, you can ask Prolog questions about the relationships.

#### Example Queries:

1. Is John a parent of Mary?

   ```prolog
   ?- parent(john, mary).
   ```

   Prolog will respond:

   ```
   true.
   ```

2. Who are John’s grandchildren?

   ```prolog
   ?- grandparent(john, X).
   ```

   Prolog will respond:

   ```
   X = alice ;
   X = charlie.
   ```

3. Who are Mary’s children?

   ```prolog
   ?- parent(mary, X).
   ```

   Prolog will respond:

   ```
   X = alice ;
   X = bob.
   ```

4. Is Mary a grandparent of Charlie?
   ```prolog
   ?- grandparent(mary, charlie).
   ```
   Prolog will respond:
   ```
   false.
   ```

---

### Step 5: Solve a Reasoning Problem

Let’s solve a reasoning problem using the same program.

#### Problem: Family Tree

Imagine the following family tree:

```
John → Mary → Alice
John → Mary → Bob → Charlie
```

Using the facts and rules in your program, you can answer questions like:

1. Who are Bob’s siblings?
   Add this rule to your program:

   ```prolog
   sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.
   ```

   Then ask:

   ```prolog
   ?- sibling(bob, X).
   ```

   Prolog will respond:

   ```
   X = alice.
   ```

2. Who are Charlie’s cousins?
   Add this rule to your program:
   ```prolog
   cousin(X, Y) :- parent(A, X), parent(B, Y), sibling(A, B).
   ```
   Then ask:
   ```prolog
   ?- cousin(charlie, X).
   ```
   Prolog will respond:
   ```
   X = alice.
   ```

---

## 1.5 Exercises

Try these exercises to practice Prolog:

1. Add a new fact: `parent(alice, dave).` Then ask:

   - Who are Alice’s children?
   - Who are John’s great-grandchildren?

2. Write a rule for "ancestor" (someone who is a parent, grandparent, great-grandparent, etc.):

   ```prolog
   ancestor(X, Y) :- parent(X, Y).
   ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
   ```

   Then ask:

   - Who are John’s descendants?

3. Write a rule for "uncle" (a parent’s sibling):
   ```prolog
   uncle(X, Y) :- sibling(X, Z), parent(Z, Y).
   ```
   Then ask:
   - Who is Charlie’s uncle?

---

## 1.6 Summary

In this chapter, you learned:

- What Prolog is and why it’s used in AI.
- How to set up and run Prolog.
- How to write simple Prolog programs using facts, rules, and queries.
- How to solve basic reasoning problems using Prolog.
