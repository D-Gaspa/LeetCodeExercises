# Week 2: July 8th - July 14th
from collections import deque


# <------------------------------------------------- July 8th, 2024 ------------------------------------------------->
# 1823. Find the Winner of the Circular Game

# There are `n` friends playing a game where they sit in a clockwise circle and are numbered from `1` to `n`.
# Starting at the `1st` friend, count the next `k` friends (wrapping around if necessary), and the last counted friend
# leaves the circle.
# Repeat until one friend remains, who is the winner.
# Given `n` and `k`, return the winner.


def findTheWinner1(n: int, k: int) -> int:
    # Initialize the queue with friends numbered from 0 to n-1
    active_players = deque(range(n))

    # Simulate the game until only one player remains
    while len(active_players) > 1:
        # Move k-1 players to the end of the queue
        for _ in range(k - 1):
            active_players.append(active_players.popleft())

        # Remove the k-th player
        active_players.popleft()

    # Convert to 1-based indexing for the result
    return active_players[0] + 1


def findTheWinner2(n: int, k: int) -> int:
    def simulate_game(players: int, step: int) -> int:
        # Base case: if only one player remains, they are at position 0
        if players == 1:
            return 0

        # Recursive case: calculate the survivor's position
        # for n-1 players and adjust it for n players
        survivor_position = simulate_game(players - 1, step)
        return (survivor_position + step) % players

    # Convert the result to 1-based indexing
    return simulate_game(n, k) + 1


def findTheWinner3(n: int, k: int) -> int:
    survivor_position = 0

    # Iterate from 2 to n, simulating the game for increasing circle sizes
    for circle_size in range(2, n + 1):
        # Calculate the new position of the survivor for the current circle size
        survivor_position = (survivor_position + k) % circle_size

    # Convert the result to 1-based indexing
    return survivor_position + 1


# <------------------------------------------------- July 9th, 2024 ------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <------------------------------------------------- July 10th, 2024 ------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <------------------------------------------------- July 11th, 2024 ------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <------------------------------------------------- July 12th, 2024 ------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <------------------------------------------------- July 13th, 2024 ------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <------------------------------------------------- July 14th, 2024 ------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for July 8th, 2024
# Expected output: 1
# findTheWinner2(n=6, k=5)

# Test cases for July 9th, 2024

# Test cases for July 10th, 2024

# Test cases for July 11th, 2024

# Test cases for July 12th, 2024

# Test cases for July 13th, 2024

# Test cases for July 14th, 2024
