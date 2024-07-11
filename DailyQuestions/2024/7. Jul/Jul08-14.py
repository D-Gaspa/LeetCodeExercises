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
    print("\n--- Input Parameters ---")
    print(f"\tcustomers = {customers}")

    print("\n--- Initialization ---")
    order_finish_time, total_customer_wait_time = 0, 0
    print(f"\torder_finish_time = {order_finish_time}")
    print(f"\ttotal_customer_wait_time = {total_customer_wait_time}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for i, (arrival_time, prep_time) in enumerate(customers):
        print(f"\n--- Customer {i + 1}/{len(customers)} ---")
        print(f"\tCurrent state:")
        print(f"\t\torder_finish_time = {order_finish_time}")
        print(f"\t\ttotal_customer_wait_time = {total_customer_wait_time}")
        print(f"\t\tarrival_time = {arrival_time}")
        print(f"\t\tprep_time = {prep_time}")

        print("\tCalculating new order_finish_time:")
        new_order_finish_time = max(arrival_time, order_finish_time) + prep_time
        print(f"\t\tmax({arrival_time}, {order_finish_time}) + {prep_time} = {new_order_finish_time}")

        print("\tCalculating customer wait time:")
        customer_wait_time = new_order_finish_time - arrival_time
        print(f"\t\t{new_order_finish_time} - {arrival_time} = {customer_wait_time}")

        print("\tUpdating variables:")
        print(f"\t\torder_finish_time: {order_finish_time} -> {new_order_finish_time}")
        order_finish_time = new_order_finish_time
        print(f"\t\ttotal_customer_wait_time: {total_customer_wait_time} -> "
              f"{total_customer_wait_time + customer_wait_time}")
        total_customer_wait_time += customer_wait_time

        iteration_data.append([i + 1, arrival_time, prep_time, order_finish_time, customer_wait_time,
                               total_customer_wait_time])

    print("\n--- Iteration Summary ---")
    headers = ["Customer", "Arrival Time", "Prep Time", "Order Finish Time", "Wait Time", "Total Wait Time"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    average_wait_time = total_customer_wait_time / len(customers)
    print(f"\tCalculating average wait time: {total_customer_wait_time} / {len(customers)} = {average_wait_time}")
    print(f"\tFinal Result: {average_wait_time}")
    return average_wait_time


# <------------------------------------------------- July 10th, 2024 ------------------------------------------------->
# 1598. Crawler Log Folder

# The LeetCode file system logs user folder operations: "../" moves to parent folder, "./" remains in the same folder,
# and "x/" moves to child folder x.
# Given a list of strings 'logs' representing these operations starting from the main folder, return the minimum number
# of operations needed to go back to the main folder.


def minOperations1(logs: List[str]) -> int:
    pass


def minOperations2(logs: List[str]) -> int:
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
# Expected output: 3.2500
# averageWaitingTime1(customers=[[5, 2], [5, 4], [10, 3], [20, 1]])

# Test cases for July 10th, 2024
# Expected output: 2
minOperations1(logs=["d1/", "d2/", "../", "d21/", "./"])
minOperations2(logs=["d1/", "d2/", "../", "d21/", "./"])

# Expected output: 3
minOperations1(logs=["d1/", "d2/", "./", "d3/", "../", "d31/"])
minOperations2(logs=["d1/", "d2/", "./", "d3/", "../", "d31/"])

# Expected output: 0
minOperations1(logs=["d1/", "../", "../", "../"])
minOperations2(logs=["d1/", "../", "../", "../"])

# Test cases for July 11th, 2024

# Test cases for July 12th, 2024

# Test cases for July 13th, 2024

# Test cases for July 14th, 2024
