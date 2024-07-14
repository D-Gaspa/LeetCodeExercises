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
    print("\n--- Input Parameters ---")
    print(f"\tlogs = {logs}")

    print("\n--- Initialization ---")
    folder_depth = 0
    print(f"\tfolder_depth = {folder_depth}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for i, operation in enumerate(logs):
        print(f"\n--- Operation {i + 1}/{len(logs)} ---")
        print(f"\tCurrent folder_depth = {folder_depth}")
        print(f"\tCurrent operation: '{operation}'")

        print("\tProcessing operation:")
        if operation == "./":
            print("\t\tOperation is './': Stay in the same folder")
            print("\t\tAction: No change to folder_depth")
        elif operation == "../":
            print("\t\tOperation is '../': Move to parent folder")
            print(f"\t\tCalculation: max(0, {folder_depth} - 1)")
            folder_depth = max(0, folder_depth - 1)
            print(f"\t\tAction: Updated folder_depth to {folder_depth}")
        else:
            print(f"\t\tOperation is '{operation}': Move to child folder")
            folder_depth += 1
            print(f"\t\tAction: Incremented folder_depth to {folder_depth}")

        iteration_data.append([i + 1, operation, folder_depth])

    print("\n--- Iteration Summary ---")
    headers = ["Operation #", "Command", "Resulting Depth"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal folder_depth: {folder_depth}")
    print(f"\tThis represents the minimum number of '../' operations needed to return to the main folder.")
    return folder_depth


def minOperations2(logs: List[str]) -> int:
    print("\n--- Input Parameters ---")
    print(f"\tlogs = {logs}")

    print("\n--- Initialization ---")
    folder_stack = []
    print(f"\tfolder_stack = {folder_stack}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for i, operation in enumerate(logs):
        print(f"\n--- Operation {i + 1}/{len(logs)} ---")
        print(f"\tCurrent folder_stack = {folder_stack}")
        print(f"\tCurrent operation: '{operation}'")

        print("\tProcessing operation:")
        if operation == "./":
            print("\t\tOperation is './': Stay in the same folder")
            print("\t\tAction: No change to folder_stack")
        elif operation == "../":
            print("\t\tOperation is '../': Move to parent folder")
            if folder_stack:
                removed_folder = folder_stack.pop()
                print(f"\t\tAction: Removed '{removed_folder}' from folder_stack")
            else:
                print("\t\tAction: Already in main folder, no change to folder_stack")
        else:
            print(f"\t\tOperation is '{operation}': Move to child folder")
            folder_stack.append(operation)
            print(f"\t\tAction: Added '{operation}' to folder_stack")

        iteration_data.append([i + 1, operation, len(folder_stack), folder_stack.copy()])

    print("\n--- Iteration Summary ---")
    headers = ["Operation #", "Command", "Stack Depth", "Current Stack"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    result = len(folder_stack)
    print(f"\tFinal folder_stack: {folder_stack}")
    print(f"\tLength of folder_stack: {result}")
    print(f"\tThis represents the minimum number of '../' operations needed to return to the main folder.")
    return result


# <------------------------------------------------- July 11th, 2024 ------------------------------------------------->
# 1190. Reverse Substrings Between Each Pair of Parentheses

# You are given a string `s` that consists of lower case English letters and brackets.
# Reverse the strings in each pair of matching parentheses, starting from the innermost one.


def reverseParentheses1(s: str) -> str:
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")

    print("\n--- Initialization ---")
    stack = []
    print(f"\tstack = {stack}")

    print("\n--- Main Loop: Iterating through string ---")
    iteration_data = []
    for i, char in enumerate(s):
        print(f"\n--- Character {i + 1}/{len(s)}: '{char}' ---")
        print(f"\tCurrent stack: {stack}")

        if char == ')':
            print("\tEncountered closing parenthesis ')'")
            reversed_substring = []
            print("\tReversing substring:")
            while stack and stack[-1] != '(':
                popped_char = stack.pop()
                reversed_substring.append(popped_char)
                print(f"\t\tPopped '{popped_char}' from stack")
                print(f"\t\tCurrent reversed_substring: {reversed_substring}")
                print(f"\t\tUpdated stack: {stack}")

            print("\tRemoving opening parenthesis '(':")
            if stack:
                stack.pop()
                print(f"\t\tRemoved '(' from stack")
                print(f"\t\tUpdated stack: {stack}")

            print("\tAdding reversed substring back to stack:")
            stack.extend(reversed_substring)
            print(f"\t\tAdded {reversed_substring} to stack")
            print(f"\t\tUpdated stack: {stack}")
        else:
            print(f"\tAppending '{char}' to stack")
            stack.append(char)
            print(f"\t\tUpdated stack: {stack}")

        iteration_data.append([i + 1, char, "".join(stack)])

    print("\n--- Iteration Summary ---")
    headers = ["Step", "Character", "Stack Content"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    result = "".join(stack)
    print(f"\tJoining stack elements: {''.join(stack)}")
    print(f"\tFinal Result: {result}")
    return result


def reverseParentheses2(s: str) -> str:
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")

    print("\n--- Initialization ---")
    parentheses_pairs = {}
    opening_parentheses = []
    print(f"\tparentheses_pairs = {parentheses_pairs}")
    print(f"\topening_parentheses = {opening_parentheses}")

    print("\n--- First Pass: Pairing Parentheses ---")
    for index, char in enumerate(s):
        print(f"\n--- Character {index + 1}/{len(s)}: '{char}' ---")
        if char == '(':
            print(f"\tEncountered opening parenthesis '(' at index {index}")
            opening_parentheses.append(index)
            print(f"\t\tAdded {index} to opening_parentheses")
            print(f"\t\tUpdated opening_parentheses: {opening_parentheses}")
        elif char == ')':
            print(f"\tEncountered closing parenthesis ')' at index {index}")
            if opening_parentheses:
                opening_index = opening_parentheses.pop()
                print(f"\t\tPopped opening index {opening_index} from opening_parentheses")
                parentheses_pairs[index] = opening_index
                parentheses_pairs[opening_index] = index
                print(f"\t\tPaired: ({opening_index}, {index})")
                print(f"\t\tUpdated parentheses_pairs: {parentheses_pairs}")
            else:
                print("\t\tNo matching opening parenthesis found")

    print("\n--- Parentheses Pairs ---")
    print(tabulate(parentheses_pairs.items(), headers=["Index", "Matching Index"], tablefmt="fancy_grid"))

    print("\n--- Second Pass: Building Result ---")
    result = []
    current_index = 0
    direction = 1
    print(f"\tInitial state: result = {result}, current_index = {current_index}, direction = {direction}")

    iteration_data = []
    while current_index < len(s):
        print(f"\n--- Processing index {current_index} ---")
        print(f"\tCurrent character: '{s[current_index]}'")
        print(f"\tCurrent direction: {'forward' if direction == 1 else 'backward'}")

        if s[current_index] in '()':
            print(f"\tEncountered parenthesis at index {current_index}")
            new_index = parentheses_pairs[current_index]
            print(f"\t\tJumping to matching parenthesis at index {new_index}")
            current_index = new_index
            direction = -direction
            print(f"\t\tReversed direction: {'forward' if direction == 1 else 'backward'}")
        else:
            print(f"\tAppending '{s[current_index]}' to result")
            result.append(s[current_index])

        current_index += direction
        print(f"\tUpdated: result = {''.join(result)}, current_index = {current_index}, direction = {direction}")
        iteration_data.append([current_index - direction, s[current_index - direction], ''.join(result), direction])

    print("\n--- Result Building Summary ---")
    headers = ["Processed Index", "Character", "Current Result", "Direction"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    final_result = ''.join(result)
    print(f"\tJoining result: {''.join(result)}")
    print(f"\tFinal Result: {final_result}")
    return final_result


# <------------------------------------------------- July 12th, 2024 ------------------------------------------------->
# 1717. Maximum Score From Removing Substrings

# Given a string `s` and two integers `x` and `y`, return the maximum points you can gain by repeatedly removing the
# substring "ab" for `x` points and "ba" for `y` points.


def maximumGain1(s: str, x: int, y: int) -> int:
    pass


def maximumGain2(s: str, x: int, y: int) -> int:
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
# Expected output: 3
# minOperations1(logs=["d1/", "d2/", "./", "d3/", "../", "d31/"])
# minOperations2(logs=["d1/", "d2/", "./", "d3/", "../", "d31/"])

# Test cases for July 11th, 2024
# Expected output: "leetcode"
# reverseParentheses1(s="(el(a(pm))xe)")
# reverseParentheses2(s="(el(a(pm))xe)")

# Test cases for July 12th, 2024
# Expected output: 19
print(maximumGain1(s="cdbcbbaaabab", x=4, y=5))
# Expected output: 20
print(maximumGain1(s="aabbaaxybbaabb", x=5, y=4))

# Test cases for July 13th, 2024

# Test cases for July 14th, 2024
