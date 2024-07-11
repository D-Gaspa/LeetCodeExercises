# Week 2: July 8th - July 14th
from collections import deque
from typing import List

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
    print("\n--- Input Parameters ---")
    print(f"\tn = {n}")
    print(f"\tk = {k}")

    print("\n--- Initialization ---")
    print("\tStarting recursive simulation")

    iteration_data = []

    def simulate_game(players: int, step: int) -> int:
        print(f"\n--- Recursive Call: simulate_game({players}, {step}) ---")
        print(f"\tCurrent state: {players} players, step size {step}")

        if players == 1:
            print("\tBase case reached: Only one player remains")
            print("\tReturning position 0")
            iteration_data.append([players, step, "Base case", 0])
            return 0

        print(f"\tMaking recursive call: simulate_game({players - 1}, {step})")
        survivor_position = simulate_game(players - 1, step)

        print("\n\tCalculating new survivor position:")
        new_position = (survivor_position + step) % players
        print(f"\t\t(survivor_position + step) % players")
        print(f"\t\t({survivor_position} + {step}) % {players} = {new_position}")

        iteration_data.append([players, step, "Recursive case", new_position])

        print(f"\tReturning new position: {new_position}")
        return new_position

    print("\n--- Starting Main Recursive Function ---")
    result = simulate_game(n, k)

    print("\n--- Recursive Calls Summary ---")
    headers = ["Players", "Step", "Case", "Survivor Position"]
    print(tabulate(iteration_data[::-1], headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    final_result = result + 1
    print(f"\tAdjusting result to 1-based indexing:")
    print(f"\t\tFinal Result = {result} + 1 = {final_result}")

    return final_result


def findTheWinner3(n: int, k: int) -> int:
    print("\n--- Input Parameters ---")
    print(f"\tn = {n}")
    print(f"\tk = {k}")

    print("\n--- Initialization ---")
    survivor_position = 0
    print(f"\tInitial survivor_position: {survivor_position}")

    iteration_data = []

    print("\n--- Main Loop ---")
    for circle_size in range(2, n + 1):
        print(f"\n--- Iteration for circle size {circle_size} ---")
        print(f"\tCurrent state:")
        print(f"\t\tsurvivor_position = {survivor_position}")
        print(f"\t\tcircle_size = {circle_size}")

        print("\tCalculation:")
        new_position = (survivor_position + k) % circle_size
        print(f"\t\tnew_position = (survivor_position + k) % circle_size")
        print(f"\t\tnew_position = ({survivor_position} + {k}) % {circle_size} = {new_position}")

        survivor_position = new_position
        print(f"\tUpdated survivor_position: {survivor_position}")

        iteration_data.append([circle_size, survivor_position])

    print("\n--- Iteration Summary ---")
    headers = ["Circle Size", "Survivor Position"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    final_result = survivor_position + 1
    print(f"\tAdjusting result to 1-based indexing:")
    print(f"\t\tFinal Result = {survivor_position} + 1 = {final_result}")

    return final_result


# <------------------------------------------------- July 9th, 2024 ------------------------------------------------->
# 1701. Average Waiting Time

# Given a restaurant with a single chef and an array `customers`, where each element `[arrival_i, time_i]` represents
# the arrival time and preparation time for each customer, return the average waiting time, assuming the chef prepares
# orders one at a time in the given order.


def averageWaitingTime1(customers: List[List[int]]) -> float:
    pass


def averageWaitingTime2(customers: List[List[int]]) -> float:
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
# findTheWinner1(n=6, k=5)
# findTheWinner2(n=6, k=5)
# findTheWinner3(n=6, k=5)

# Test cases for July 9th, 2024
# Expected output: 5.0000
# averageWaitingTime1(customers=[[1, 2], [2, 5], [4, 3]])
# averageWaitingTime2(customers=[[1, 2], [2, 5], [4, 3]])

# Expected output: 3.2500
# averageWaitingTime1(customers=[[5, 2], [5, 4], [10, 3], [20, 1]])
# averageWaitingTime2(customers=[[5, 2], [5, 4], [10, 3], [20, 1]])

# Test cases for July 10th, 2024

# Test cases for July 11th, 2024

# Test cases for July 12th, 2024

# Test cases for July 13th, 2024

# Test cases for July 14th, 2024
