# Week 2: July 8th - July 14th
from collections import deque

from tabulate import tabulate


# <------------------------------------------------- July 8th, 2024 ------------------------------------------------->
# 1823. Find the Winner of the Circular Game

# There are `n` friends playing a game where they sit in a clockwise circle and are numbered from `1` to `n`.
# Starting at the `1st` friend, count the next `k` friends (wrapping around if necessary), and the last counted friend
# leaves the circle.
# Repeat until one friend remains, who is the winner.
# Given `n` and `k`, return the winner.


def findTheWinner1(n: int, k: int) -> int:
    print("\n--- Input Parameters ---")
    print(f"\tn = {n}")
    print(f"\tk = {k}")

    print("\n--- Initialization ---")
    active_players = deque(range(n))
    print(f"\tInitial active_players: {list(active_players)}")

    iteration_data = []
    round_number = 1

    print("\n--- Main Loop ---")
    while len(active_players) > 1:
        print(f"\n--- Round {round_number} ---")
        print(f"\tCurrent active players: {list(active_players)}")
        print(f"\tRemoving player at position {k}")

        print("\tRotating players:")
        for i in range(k - 1):
            moved_player = active_players.popleft()
            active_players.append(moved_player)
            print(f"\t\tMove {i + 1}: Moved player {moved_player} to the end")

        removed_player = active_players.popleft()
        print(f"\tRemoved player: {removed_player}")

        iteration_data.append([round_number, list(active_players), removed_player])
        round_number += 1

    print("\n--- Iteration Summary ---")
    headers = ["Round", "Active Players", "Removed Player"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    winner = active_players[0] + 1
    print(f"\tLast remaining player (0-indexed): {active_players[0]}")
    print(f"\tConverting to 1-indexed: {active_players[0]} + 1 = {winner}")
    return winner


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

    # Simulate the game for increasing circle sizes
    for circle_size in range(2, n + 1):
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
findTheWinner1(n=6, k=5)
# findTheWinner2(n=6, k=5)
# findTheWinner3(n=6, k=5)

# Test cases for July 9th, 2024

# Test cases for July 10th, 2024

# Test cases for July 11th, 2024

# Test cases for July 12th, 2024

# Test cases for July 13th, 2024

# Test cases for July 14th, 2024
