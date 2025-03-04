# Week 2: July 8th - July 14th
import re
from collections import deque, defaultdict
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
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")
    print(f"\tx = {x}")
    print(f"\ty = {y}")

    print("\n--- Initialization ---")
    high_value_seq, low_value_seq = ('ab', x), ('ba', y)
    if x < y:
        print(f"\tSince x < y, swapping high and low value sequences")
        high_value_seq, low_value_seq = low_value_seq, high_value_seq
    print(f"\thigh_value_seq = {high_value_seq}")
    print(f"\tlow_value_seq = {low_value_seq}")
    total_points = 0
    print(f"\ttotal_points = {total_points}")

    iteration_data = []
    for seq_index, ((first_char, second_char), points) in enumerate([high_value_seq, low_value_seq]):
        print(
            f"\n--- Processing {'High' if seq_index == 0 else 'Low'} Value Sequence: "
            f"'{first_char}{second_char}' (Points: {points}) ---")
        stack = []
        print(f"\tInitial stack: {stack}")

        for char_index, char in enumerate(s):
            print(f"\n\t--- Character {char_index + 1}/{len(s)}: '{char}' ---")
            print(f"\tCurrent stack: {stack}")

            if stack and stack[-1] == first_char and char == second_char:
                popped_char = stack.pop()
                total_points += points
                print(f"\t\tFound sequence '{popped_char}{char}'. Removing from stack.")
                print(f"\t\tAdding {points} points. Total points: {total_points}")
            else:
                stack.append(char)
                print(f"\t\tAppending '{char}' to stack.")

            print(f"\tUpdated stack: {stack}")

            iteration_data.append([seq_index + 1, char_index + 1, char, ''.join(stack), total_points])

        s = stack
        print(f"\tRemaining string after processing: {''.join(s)}")

    print("\n--- Iteration Summary ---")
    headers = ["Sequence", "Character Index", "Current Char", "Stack", "Total Points"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Total Points: {total_points}")
    return total_points


def maximumGain2(s: str, x: int, y: int) -> int:
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")
    print(f"\tx = {x}")
    print(f"\ty = {y}")

    print("\n--- Initialization ---")
    high_value_char, low_value_char = 'a', 'b'
    if x < y:
        print(f"\tSince x < y, swapping high and low value chars")
        x, y = y, x
        high_value_char, low_value_char = low_value_char, high_value_char
    print(f"\thigh_value_char = '{high_value_char}', points = {x}")
    print(f"\tlow_value_char = '{low_value_char}', points = {y}")

    total_points = 0
    unpaired_high_count, unpaired_low_count = 0, 0
    print(f"\ttotal_points = {total_points}")
    print(f"\tunpaired_high_count = {unpaired_high_count}")
    print(f"\tunpaired_low_count = {unpaired_low_count}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for char_index, char in enumerate(s):
        print(f"\n--- Character {char_index + 1}/{len(s)}: '{char}' ---")
        print(
            f"\tCurrent state: total_points = {total_points}, unpaired_high_count ="
            f" {unpaired_high_count}, unpaired_low_count = {unpaired_low_count}")

        if char == high_value_char:
            unpaired_high_count += 1
            print(
                f"\tFound high-value char '{high_value_char}'. "
                f"Incrementing unpaired_high_count to {unpaired_high_count}")
        elif char == low_value_char:
            if unpaired_high_count > 0:
                unpaired_high_count -= 1
                total_points += x
                print(f"\tFound low-value char '{low_value_char}' and unpaired_high_count > 0.")
                print(f"\tForming high-value pair. Decrementing unpaired_high_count to {unpaired_high_count}")
                print(f"\tAdding {x} points. Total points now {total_points}")
            else:
                unpaired_low_count += 1
                print(f"\tFound low-value char '{low_value_char}' but no unpaired high-value char.")
                print(f"\tIncrementing unpaired_low_count to {unpaired_low_count}")
        else:
            print(f"\tFound non-pair char '{char}'. Checking for potential low-value pairs.")
            if unpaired_high_count:
                print("\tFound unpaired high-value chars. Checking for low-value pairs.")
                if unpaired_low_count:
                    pairs_to_remove = min(unpaired_high_count, unpaired_low_count)
                    points_to_add = pairs_to_remove * y
                    total_points += points_to_add
                    print(f"\tFound {pairs_to_remove} low-value pairs to remove.")
                    print(f"\tAdding {points_to_add} points. Total points now {total_points}")
                    unpaired_high_count, unpaired_low_count = 0, 0
                    print("\tResetting unpaired counts to 0")
                else:
                    print("\tNo low-value chars for pairing. Resetting unpaired_high_count to 0")
                    unpaired_high_count = 0
            elif unpaired_low_count:
                print("\tNo high-value chars for pairing. Resetting unpaired_low_count to 0")
                unpaired_low_count = 0

        iteration_data.append([char_index + 1, char, unpaired_high_count, unpaired_low_count, total_points])

    print("\n--- Final Check for Remaining Pairs ---")
    if unpaired_high_count and unpaired_low_count:
        pairs_to_remove = min(unpaired_high_count, unpaired_low_count)
        points_to_add = pairs_to_remove * y
        total_points += points_to_add
        print(f"\tFound {pairs_to_remove} remaining low-value pairs to remove.")
        print(f"\tAdding {points_to_add} points. Total points now {total_points}")

    iteration_data.append(["Final", "N/A", unpaired_high_count, unpaired_low_count, total_points])

    print("\n--- Iteration Summary ---")
    headers = ["Character Index", "Current Char", "Unpaired High Count", "Unpaired Low Count", "Total Points"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Total Points: {total_points}")
    return total_points


# <------------------------------------------------- July 13th, 2024 ------------------------------------------------->
# 2751. Robot Collisions

# There are `n` 1-indexed robots on a line, each with a position, health, and direction ('L' for left, 'R' for right).
# Robots move simultaneously at the same speed, and collide if they share a position.
# In a collision, the robot with lower health is removed, and the survivor's health decreases by one
# (both are removed if health is equal).
# The task is to determine the final health of surviving robots in their original order after all collisions.


def survivedRobotsHealths1(positions: List[int], healths: List[int], directions: str) -> List[int]:
    print("\n--- Input Parameters ---")
    print(f"\tpositions = {positions}")
    print(f"\thealths = {healths}")
    print(f"\tdirections = {directions}")

    print("\n--- Initialization ---")
    sorted_robot_indices = sorted(range(len(positions)), key=lambda x: positions[x])
    right_moving_robot_stack = []
    print(f"\tsorted_robot_indices = {sorted_robot_indices}")
    print(f"\tright_moving_robot_stack = {right_moving_robot_stack}")

    def handle_collision(left_robot_index, right_robot_index):
        print(f"\n\t--- Handling Collision between Robot {left_robot_index + 1} and Robot {right_robot_index + 1} ---")
        print(
            f"\t\tBefore collision: Robot {left_robot_index + 1} health = {healths[left_robot_index]}, "
            f"Robot {right_robot_index + 1} health = {healths[right_robot_index]}")

        health_diff = healths[left_robot_index] - healths[right_robot_index]
        print(f"\t\tHealth difference: {health_diff}")

        if health_diff < 0:
            print(f"\t\tRobot {left_robot_index + 1} is destroyed, Robot {right_robot_index + 1} loses 1 health")
            healths[left_robot_index] = 0
            healths[right_robot_index] -= 1
            right_moving_robot_stack.pop()
        elif health_diff > 0:
            print(f"\t\tRobot {right_robot_index + 1} is destroyed, Robot {left_robot_index + 1} loses 1 health")
            healths[right_robot_index] = 0
            healths[left_robot_index] -= 1
        else:
            print(f"\t\tBoth robots are destroyed")
            healths[left_robot_index], healths[right_robot_index] = 0, 0
            right_moving_robot_stack.pop()

        print(
            f"\t\tAfter collision: Robot {left_robot_index + 1} health = {healths[left_robot_index]}, "
            f"Robot {right_robot_index + 1} health = {healths[right_robot_index]}")

    print("\n--- Main Simulation Loop ---")
    iteration_data = []
    for i, current_robot_index in enumerate(sorted_robot_indices):
        print(f"\n--- Processing Robot {current_robot_index + 1} (Iteration {i + 1}/{len(sorted_robot_indices)}) ---")
        print(
            f"\tPosition: {positions[current_robot_index]}, Health: {healths[current_robot_index]}, "
            f"Direction: {directions[current_robot_index]}")

        if directions[current_robot_index] == 'R':
            print(f"\tRobot {current_robot_index + 1} is moving right, adding to stack")
            right_moving_robot_stack.append(current_robot_index)
            print(f"\tUpdated right_moving_robot_stack: {right_moving_robot_stack}")
        else:
            print(f"\tRobot {current_robot_index + 1} is moving left, checking for collisions")
            collisions = 0
            while right_moving_robot_stack and healths[current_robot_index] > 0:
                colliding_robot_index = right_moving_robot_stack[-1]
                print(f"\t\tPotential collision with Robot {colliding_robot_index + 1}")
                handle_collision(colliding_robot_index, current_robot_index)
                collisions += 1

            print(f"\tTotal collisions for Robot {current_robot_index + 1}: {collisions}")

        iteration_data.append([
            i + 1,
            current_robot_index + 1,
            positions[current_robot_index],
            healths[current_robot_index],
            directions[current_robot_index],
            len(right_moving_robot_stack)
        ])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Robot", "Position", "Health", "Direction", "Right-moving Stack Size"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    surviving_healths = [health for health in healths if health > 0]
    print(f"\tSurviving robot healths: {surviving_healths}")

    return surviving_healths


# <------------------------------------------------- July 14th, 2024 ------------------------------------------------->
# 726. Number of Atoms

# Given a string formula representing a chemical formula, return the count of each atom.
# Atoms start with an uppercase letter followed by optional lowercase letters, and may have a count greater than 1
# indicated by digits.
# The formula can include concatenated formulas, parentheses, and nested counts, and the output should be a string with
# atom names in sorted order followed by their counts if greater than 1.

def countOfAtoms1(formula: str) -> str:
    print("\n--- Input Parameters ---")
    print(f"\tformula = {formula}")

    print("\n--- Initialization ---")
    formula_length = len(formula)
    atoms_stack = [defaultdict(int)]
    print(f"\tformula_length = {formula_length}")
    print(f"\tatoms_stack = {atoms_stack}")

    index = 0

    print("\n--- Main Loop ---")
    while index < formula_length:
        print(f"\n--- While Loop Iteration: while {index} < {formula_length} ---")
        print(f"\tCurrent character: '{formula[index]}' at index {index}")
        print(f"\tCurrent atoms_stack: {atoms_stack}")

        if formula[index] == '(':
            print("\tFound opening parenthesis. Starting a new nested formula.")
            atoms_stack.append(defaultdict(int))
            index += 1
        elif formula[index] == ')':
            print("\tFound closing parenthesis. Processing nested formula.")
            index += 1
            start_index = index
            while index < formula_length and formula[index].isdigit():
                index += 1
            count_multiplier = int(formula[start_index:index] or 1)
            print(f"\tNested formula multiplier: {count_multiplier}")

            print("\tMultiplying and merging nested atom counts with parent formula")
            nested_atom_counts = atoms_stack.pop()
            for atom, count in nested_atom_counts.items():
                atoms_stack[-1][atom] += count * count_multiplier
                print(f"\t\tUpdating {atom}: {atoms_stack[-1][atom] - count * count_multiplier} + "
                      f"{count} * {count_multiplier} = {atoms_stack[-1][atom]}")

        elif formula[index].isupper():
            print("\tFound uppercase letter. Processing new atom.")
            start_index = index
            index += 1
            while index < formula_length and formula[index].islower():
                index += 1
            atom_name = formula[start_index:index]
            print(f"\tAtom name: {atom_name}")

            start_index = index
            while (index < formula_length and
                   formula[index].isdigit()):
                index += 1
            count = int(formula[start_index:index] or 1)
            print(f"\tAtom count: {count}")

            print(f"\tAdding atom count to current formula level")
            atoms_stack[-1][atom_name] += count
            print(f"\t\tUpdated {atom_name} count: {atoms_stack[-1][atom_name]}")

    print("\n--- Function Returning ---")
    print("\tFormatting the final result")
    formatted_atom_counts = ""
    for atom, count in sorted(atoms_stack[0].items()):
        print(f"\t\t{atom}: {count}")
        formatted_atom_counts += atom + (str(count) if count > 1 else "")

    print(f"\tFinal Result: {formatted_atom_counts}")

    return formatted_atom_counts


def countOfAtoms2(formula: str) -> str:
    print("\n--- Input Parameters ---")
    print(f"\tformula = {formula}")

    print("\n--- Initialization ---")
    formula_length = len(formula)
    multiplier_stack = [1]
    multiplier_at_index = [1] * formula_length
    current_multiplier = 1
    current_count_str = ""
    print(f"\tformula_length = {formula_length}")
    print(f"\tmultiplier_stack = {multiplier_stack}")
    print(f"\tmultiplier_at_index = {multiplier_at_index}")
    print(f"\tcurrent_multiplier = {current_multiplier}")
    print(f"\tcurrent_count_str = '{current_count_str}'")

    print("\n--- First Pass: Calculate multipliers ---")
    index = formula_length - 1
    first_pass_data = []
    while index >= 0:
        print(f"\n--- Iteration {formula_length - index}/{formula_length} ---")
        current_char = formula[index]
        print(f"\tCurrent character: '{current_char}' at index {index}")
        print(f"\tBefore processing: current_multiplier = {current_multiplier}, "
              f"current_count_str = '{current_count_str}'")

        if current_char.isdigit():
            current_count_str += current_char
            print(f"\tDigit found. Updated current_count_str = '{current_count_str}'")
        elif current_char.isalpha():
            current_count_str = ""
            print(f"\tAlphabet found. Reset current_count_str = '{current_count_str}'")
        elif current_char == ')':
            print("\tClosing parenthesis found. Start of a nested formula (reading right to left)")
            new_multiplier = int(current_count_str[::-1] or 1)
            multiplier_stack.append(new_multiplier)
            current_multiplier *= new_multiplier
            current_count_str = ""
            print(f"\t\tNew multiplier: {new_multiplier}")
            print(f"\t\tUpdated multiplier_stack: {multiplier_stack}")
            print(f"\t\tUpdated current_multiplier: {current_multiplier}")
        elif current_char == '(':
            print("\tOpening parenthesis found. End of a nested formula (reading right to left)")
            current_multiplier //= multiplier_stack.pop()
            print(f"\t\tUpdated multiplier_stack: {multiplier_stack}")
            print(f"\t\tUpdated current_multiplier: {current_multiplier}")

        multiplier_at_index[index] = current_multiplier
        print(f"\tStored current_multiplier {current_multiplier} at index {index}")

        first_pass_data.append([formula_length - index, current_char, current_multiplier, current_count_str])
        index -= 1

    print("\n--- First Pass Summary ---")
    headers = ["Iteration", "Character", "Current Multiplier", "Current Count String"]
    print(tabulate(first_pass_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Second Pass: Count atoms ---")
    atom_map = defaultdict(int)
    index = 0
    second_pass_data = []
    while index < formula_length:
        print(f"\n--- While Loop Iteration: while {index} < {formula_length} ---")
        print(f"\tCurrent character: '{formula[index]}' at index {index}")

        if formula[index].isupper():
            print("\tUppercase letter found. Processing new atom.")
            start_index = index
            index += 1
            if index < formula_length and formula[index].islower():
                index += 1
            atom_name = formula[start_index:index]
            print(f"\tAtom name: {atom_name}")

            start_index = index
            while index < formula_length and formula[index].isdigit():
                index += 1
            atom_count = int(formula[start_index:index] or 1)
            print(f"\tAtom count: {atom_count}")

            final_count = atom_count * multiplier_at_index[index - 1]
            atom_map[atom_name] += final_count
            print(f"\tApplied multiplier: {multiplier_at_index[index - 1]}")
            print(f"\tFinal count for {atom_name}: {final_count}")
            print(f"\tUpdated atom_map: {dict(atom_map)}")

            second_pass_data.append([atom_name, atom_count, multiplier_at_index[index - 1], final_count])
        else:
            index += 1

    print("\n--- Second Pass Summary ---")
    headers = ["Atom", "Initial Count", "Multiplier", "Final Count"]
    print(tabulate(second_pass_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("\tFormatting the final result")
    formatted_atom_counts = ""
    for atom, count in sorted(atom_map.items()):
        print(f"\t\t{atom}: {count}")
        formatted_atom_counts += atom + (str(count) if count > 1 else "")

    print(f"\tFinal Result: {formatted_atom_counts}")

    return formatted_atom_counts


def countOfAtoms3(formula: str) -> str:
    print("\n--- Input Parameters ---")
    print(f"\tformula = {formula}")

    print("\n--- Initialization ---")
    multiplier_stack = [1]
    current_multiplier = 1
    atom_map = defaultdict(int)
    print(f"\tmultiplier_stack = {multiplier_stack}")
    print(f"\tcurrent_multiplier = {current_multiplier}")
    print(f"\tatom_map = {dict(atom_map)}")

    print("\n--- Regex Pattern Matching ---")
    pattern = r"([A-Z][a-z]*)(\d*)|(\()|(\))(\d*)"
    print(f"\tPattern: {pattern}")
    formula_components = re.findall(pattern, formula)
    print(f"\tMatched components: {formula_components}")
    formula_components.reverse()
    print(f"\tReversed components: {formula_components}")

    print("\n--- Main Loop: Processing Formula Components ---")
    iteration_data = []
    for i, (atom, count, left_parenthesis, right_parenthesis, multiplicity) in enumerate(formula_components, 1):
        print(f"\n--- Iteration {i}/{len(formula_components)} ---")
        print(f"\tCurrent component: {(atom, count, left_parenthesis, right_parenthesis, multiplicity)}")
        print(f"\tBefore processing: current_multiplier = {current_multiplier}, multiplier_stack = {multiplier_stack}")

        if atom:
            print(f"\tProcessing atom: {atom}")
            atom_count = int(count) if count else 1
            print(f"\t\tAtom count: {atom_count}")
            atom_map[atom] += current_multiplier * atom_count
            print(f"\t\tUpdated atom_map[{atom}] = {atom_map[atom]}")

        elif right_parenthesis:
            print("\tProcessing closing parenthesis")
            multiplicity = int(multiplicity) if multiplicity else 1
            print(f"\t\tMultiplicity: {multiplicity}")
            multiplier_stack.append(multiplicity)
            current_multiplier *= multiplicity
            print(f"\t\tUpdated multiplier_stack: {multiplier_stack}")
            print(f"\t\tUpdated current_multiplier: {current_multiplier}")

        elif left_parenthesis:
            print("\tProcessing opening parenthesis (end of a group when processing in reverse)")
            popped_multiplier = multiplier_stack.pop()
            current_multiplier //= popped_multiplier
            print(f"\t\tPopped multiplier: {popped_multiplier}")
            print(f"\t\tUpdated multiplier_stack: {multiplier_stack}")
            print(f"\t\tUpdated current_multiplier: {current_multiplier}")

        iteration_data.append([i, atom or left_parenthesis or right_parenthesis,
                               count or multiplicity, current_multiplier, dict(atom_map)])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Component", "Count/Multiplicity", "Current Multiplier", "Atom Map"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("\tFormatting the final result")
    formatted_atom_counts = ""
    for atom, count in sorted(atom_map.items()):
        print(f"\t\t{atom}: {count}")
        formatted_atom_counts += (atom +
                                  (str(count) if count > 1 else ""))

    print(f"\tFinal Result: {formatted_atom_counts}")

    return formatted_atom_counts


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
# maximumGain1(s="cdbcbbaaabab", x=4, y=5)
# maximumGain2(s="cdbcbbaaabab", x=4, y=5)

# Test cases for July 13th, 2024
# Expected output: [14]
# survivedRobotsHealths1(positions=[3, 5, 2, 6], healths=[10, 12, 15, 12], directions="RLRL")

# Test cases for July 14th, 2024
# Expected output: "K32N2O62S20"
# countOfAtoms1(formula="K32(ON(SO3)10)2")
# countOfAtoms2(formula="K32(ON(SO3)10)2")
# countOfAtoms3(formula="K32(ON(SO3)10)2")
