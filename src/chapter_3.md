# Lab 3: Advanced Prolog Concepts & Problem Solving

## Learning Objectives

By the end of this chapter, you will:

- Understand how Prolog uses **backtracking** to explore solutions.
- Learn how to control Prolog’s search behavior using the **Cut operator**.
- Understand **Negation as Failure** and how it works in Prolog.
- Write and use more complex predicates to solve real-world problems.
- Solve practical problems like puzzles and reasoning tasks using Prolog.

---

## 3.1 Backtracking in Prolog

### What is Backtracking?

Backtracking is the process Prolog uses to explore all possible solutions to a query. When Prolog encounters a choice point (multiple possible solutions), it tries one option. If that option fails, it **backtracks** to the previous choice point and tries the next option.

### Example: Backtracking in Action

Consider the following facts:

```prolog
likes(john, pizza).
likes(john, pasta).
likes(mary, sushi).
likes(mary, pizza).
```

If we ask:

```prolog
?- likes(john, X).
```

Prolog will:

1. Find the first solution: `X = pizza`.
2. If we ask for more solutions (using `;`), Prolog will backtrack and find \( X = \text{pasta} \).
3. If there are no more solutions, Prolog will stop.

---

## 3.2 Controlling Backtracking with the Cut Operator (`!`)

The **Cut operator** (`!`) is used to control Prolog’s backtracking behavior. When Prolog encounters a cut, it **commits** to the choices made so far and will not backtrack past the cut.

### Syntax:

```prolog
rule_name :- condition1, condition2, !, condition3.
```

### Example: Using the Cut Operator

```prolog
max(X, Y, X) :- X >= Y, !.
max(X, Y, Y).
```

This rule finds the maximum of two numbers. Here’s how it works:

1. If \( X \geq Y \), Prolog commits to the first rule and does not backtrack.
2. If \( X < Y \), Prolog skips the first rule and uses the second rule.

#### Query:

```prolog
?- max(5, 3, M).
```

Prolog will respond:

```
M = 5.
```

---

## 3.3 Negation as Failure (`\+`)

In Prolog, **negation as failure** means that something is considered false if Prolog cannot prove it to be true. This is represented by the operator `\+`.

### Example: Negation as Failure

```prolog
likes(john, pizza).
likes(mary, sushi).

dislikes(Person, Food) :- \+ likes(Person, Food).
```

#### Query:

```prolog
?- dislikes(john, sushi).
```

Prolog will respond:

```
true.
```

Here, Prolog cannot prove that `john` likes `sushi`, so it concludes that `john` dislikes `sushi`.

---

## 3.4 Writing Complex Predicates

### Example 1: Sibling Relationship

Let’s define a rule to find siblings:

```prolog
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.
```

This rule says that \( X \) and \( Y \) are siblings if:

1. They share the same parent \( Z \).
2. \( X \) and \( Y \) are not the same person (`X \= Y`).

#### Query:

```prolog
?- sibling(alice, bob).
```

Prolog will respond:

```
true.
```

---

### Example 2: Ancestor Relationship

Let’s define a rule to find ancestors:

```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

This rule says that \( X \) is an ancestor of \( Y \) if:

1. \( X \) is a parent of \( Y \), or
2. \( X \) is a parent of \( Z \), and \( Z \) is an ancestor of \( Y \).

#### Query:

```prolog
?- ancestor(john, sarah).
```

Prolog will respond:

```
true.
```

---

## 3.5 Solving Practical Problems: Choosing a Pet

Let’s solve a simple problem using Prolog: **choosing the best pet based on your preferences**. This example demonstrates how Prolog can use facts, rules, and user input to make decisions.

---

### Problem: Choosing a Pet

Imagine you want to choose a pet, but you’re unsure which one is best for your lifestyle. Prolog can help by asking you questions about your preferences and recommending a pet based on your answers.

---

### Prolog Code

Here’s the Prolog program for choosing a pet:

```prolog
% Rules for choosing a pet
pet(dog) :- likes_outdoors, has_space.
pet(cat) :- likes_indoors, busy_schedule.
pet(fish) :- likes_peace, low_maintenance.

% Facts based on user preferences
likes_outdoors :- ask('Do you enjoy outdoor activities?').
has_space :- ask('Do you have a spacious home?').
likes_indoors :- ask('Do you prefer staying indoors?').
busy_schedule :- ask('Do you have a busy schedule?').
likes_peace :- ask('Do you enjoy peaceful environments?').
low_maintenance :- ask('Do you prefer low-maintenance pets?').

% Asking the user questions
ask(Question) :-
    format('~w (yes/no): ', [Question]),
    read(Response),
    Response = yes.
```

---

### How It Works

1. **Rules for Pets**:

   - The program defines rules for recommending a pet based on your preferences:
     - A **dog** is recommended if you enjoy outdoor activities and have a spacious home.
     - A **cat** is recommended if you prefer staying indoors and have a busy schedule.
     - A **fish** is recommended if you enjoy peaceful environments and prefer low-maintenance pets.

2. **User Input**:

   - The program asks you questions about your preferences (e.g., "Do you enjoy outdoor activities?").
   - You respond with `yes` or `no`.

3. **Decision Making**:
   - Based on your answers, Prolog evaluates the rules and recommends a pet.

---

### Example Interaction

Here’s an example of how the program works:

#### Query:

```prolog
?- pet(X).
```

#### Interaction:

Prolog will ask you questions:

```
Do you enjoy outdoor activities? (yes/no): yes.
Do you have a spacious home? (yes/no): yes.
```

Based on your answers, Prolog will respond:

```
X = dog.
```

If you answer differently:

```
Do you enjoy outdoor activities? (yes/no): no.
Do you prefer staying indoors? (yes/no): yes.
Do you have a busy schedule? (yes/no): yes.
```

Prolog will respond:

```
X = cat.
```

---

### Explanation of Key Concepts

1. **Rules**:

   - Each rule (`pet(dog)`, `pet(cat)`, `pet(fish)`) defines the conditions under which a specific pet is recommended.

2. **Facts**:

   - Facts like `likes_outdoors` and `has_space` are determined dynamically based on user input.

3. **Dynamic Input**:

   - The `ask/1` predicate asks the user a question and reads their response. If the response is `yes`, the fact is considered true.

4. **Decision Process**:
   - Prolog evaluates the rules in order. If the conditions for a rule are met, Prolog recommends the corresponding pet.

---

## 3.6 Exercises

Try modifying the program to add more pets or preferences. For example:

- Add a rule for recommending a **bird**:

  ```prolog
  pet(bird) :- likes_outdoors, low_maintenance, likes_music.
  ```

- Add a new preference question:
  ```prolog
  likes_music :- ask('Do you enjoy musical sounds?').
  ```

Then test the program by running:

```prolog
?- pet(X).
```

---

> **Tip**: Practice is key to mastering Prolog. Try writing your own rules and solving small puzzles to build confidence!
