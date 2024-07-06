# Week 1: July 1st - July 7th
import heapq
from collections import Counter
from typing import List, Optional

from tabulate import tabulate

from Utils.linked_list_utils import ListNode, LinkedListVisualizer


# <-------------------------------------------------- July 1st, 2024 -------------------------------------------------->
# 1550. Three Consecutive Odds

# Given an integer `arr`, return `true` if there are three consecutive odd numbers in the array.
# Otherwise, return `false`.


def threeConsecutiveOdds1(arr):
    print("\n--- Input Parameters ---")
    print(f"\tarr = {arr}")

    print("\n--- Initialization ---")
    consecutive_odds = 0
    print(f"\tconsecutive_odds = {consecutive_odds}")

    print("\n--- Input Validation ---")
    if len(arr) < 3:
        print(f"\tArray length ({len(arr)}) is less than 3")
        print("\tReturning False")
        return False
    else:
        print(f"\tArray length ({len(arr)}) is valid")

    print("\n--- Main Loop ---")
    iteration_data = []
    process_finished = False
    for i, num in enumerate(arr):
        print(f"\n--- Element {i + 1}/{len(arr)} ---")
        print(f"\tCurrent number: {num}")
        print(f"\tconsecutive_odds at start: {consecutive_odds}")

        print("\tChecking if number is odd:")
        if num % 2:
            print(f"\t\t{num} is odd (remainder when divided by 2 is {num % 2})")
            consecutive_odds += 1
            print(f"\t\tIncrementing consecutive_odds to {consecutive_odds}")
        else:
            print(f"\t\t{num} is even (remainder when divided by 2 is {num % 2})")
            consecutive_odds = 0
            print(f"\t\tResetting consecutive_odds to {consecutive_odds}")

        print("\tChecking if three consecutive odds found:")
        if consecutive_odds == 3:
            print("\t\tThree consecutive odds found!")
            print("\t\tReturning True")
            process_finished = True
            iteration_data.append([i + 1, num, "Odd" if num % 2 else "Even", consecutive_odds])
            break
        else:
            print(f"\t\tNot yet three consecutive odds (current count: {consecutive_odds})")

        iteration_data.append([i + 1, num, "Odd" if num % 2 else "Even", consecutive_odds])

    print("\n--- Iteration Summary ---")
    headers = ["Element", "Number", "Odd/Even", "Consecutive Odds"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    if process_finished:
        return True

    print("\n--- Function Returning ---")
    print("\tThree consecutive odds not found in the entire array")
    print("\tReturning False")
    return False


# <-------------------------------------------------- July 2nd, 2024 -------------------------------------------------->
# 350. Intersection of Two Arrays II

# Given two integer arrays `nums1` and `nums2`, return an array of their intersection.
# Each element in the result must appear as many times as it shows in both arrays,
# and you may return the result in any order.


def intersect1(nums1: List[int], nums2: List[int]) -> List[int]:
    print("\n--- Input Parameters ---")
    print(f"\tnums1 = {nums1}")
    print(f"\tnums2 = {nums2}")

    print("\n--- Initialization ---")
    if len(nums1) > len(nums2):
        print("\tSwapping nums1 and nums2 to ensure nums1 is shorter")
        nums1, nums2 = nums2, nums1
    print(f"\tAfter potential swap: nums1 = {nums1}, nums2 = {nums2}")

    print("\nSorting both arrays:")
    nums1.sort()
    nums2.sort()
    print(f"\tSorted nums1 = {nums1}")
    print(f"\tSorted nums2 = {nums2}")

    intersection = []
    index1, index2 = 0, 0
    print(f"\nInitial state: index1 = {index1}, index2 = {index2}, intersection = {intersection}")

    print("\n--- Main Loop ---")
    iteration_data = []
    while index1 < len(nums1) and index2 < len(nums2):
        print(f"\n--- Iteration {len(iteration_data) + 1} ---")
        print(f"While loop condition: index1 ({index1}) < len(nums1) ({len(nums1)}) "
              f"and index2 ({index2}) < len(nums2) ({len(nums2)})")
        print(f"\tCurrent state: index1 = {index1}, index2 = {index2}")
        print(f"\tComparing nums1[{index1}] = {nums1[index1]} and nums2[{index2}] = {nums2[index2]}")

        if nums1[index1] < nums2[index2]:
            print(f"\t\tnums1[{index1}] < nums2[{index2}], incrementing index1")
            index1 += 1
            action = "Increment index1"
        elif nums1[index1] > nums2[index2]:
            print(f"\t\tnums1[{index1}] > nums2[{index2}], incrementing index2")
            index2 += 1
            action = "Increment index2"
        else:
            print(f"\t\tnums1[{index1}] == nums2[{index2}], adding to intersection and incrementing both indices")
            intersection.append(nums1[index1])
            index1 += 1
            index2 += 1
            action = "Add to intersection, increment both"

        iteration_data.append([len(iteration_data) + 1, index1, index2, nums1[index1 - 1] if index1 > 0 else None,
                               nums2[index2 - 1] if index2 > 0 else None, action, intersection.copy()])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Index1", "Index2", "nums1 value", "nums2 value", "Action", "Current Intersection"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal intersection: {intersection}")
    return intersection


def intersect2(nums1: List[int], nums2: List[int]) -> List[int]:
    print("\n--- Input Parameters ---")
    print(f"\tnums1 = {nums1}")
    print(f"\tnums2 = {nums2}")

    print("\n--- Initialization ---")
    if len(nums1) > len(nums2):
        print("\tSwapping nums1 and nums2 to"
              " ensure nums1 is shorter")
        nums1, nums2 = nums2, nums1
    print(f"\tAfter potential swap: nums1 = {nums1}, nums2 = {nums2}")

    count_nums1 = Counter(nums1)
    print("\nCreating frequency map of nums1:")
    print(tabulate(count_nums1.items(), headers=["Element", "Frequency"], tablefmt="fancy_grid"))

    intersection = []
    print(f"\nInitial state: intersection = {intersection}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for i, num in enumerate(nums2):
        print(f"\n--- Iteration {i + 1}/{len(nums2)} ---")
        print(f"\tProcessing num = {num}")
        print(f"\tCurrent count in frequency map: {count_nums1[num]}")

        if count_nums1[num] > 0:
            print(f"\t\tnum {num} found in frequency map with count > 0")
            intersection.append(num)
            count_nums1[num] -= 1
            action = f"Add {num} to intersection, decrement count"
        else:
            print(f"\t\tnum {num} not found in frequency map or count is 0")
            action = "No action"

        iteration_data.append([i + 1, num, count_nums1[num], action, intersection.copy()])

        print(f"\tUpdated frequency map for {num}: {count_nums1[num]}")
        print(f"\tCurrent intersection: {intersection}")

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Number", "Count after action", "Action", "Current Intersection"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal intersection: {intersection}")
    return intersection


# <-------------------------------------------------- July 3rd, 2024 -------------------------------------------------->
# 1509. Minimum Difference Between Largest and Smallest Value in Three Moves

# Return the minimum difference between the largest and smallest value of `nums` after performing at most three moves.
# In one move, you can choose one element of `nums` and change it to any value.


def minDifference1(nums: List[int]) -> int:
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")

    print("\n--- Initialization ---")
    n = len(nums)
    print(f"\tLength of nums (n) = {n}")

    print("\n--- Edge Case Check ---")
    if n <= 4:
        print("\tEdge case: n <= 4")
        print("\tAll elements can be made equal")
        print("\tReturning 0")
        return 0

    print("\n--- Extracting Smallest and Largest Elements ---")
    smallest_four = heapq.nsmallest(4, nums)
    largest_four = heapq.nlargest(4, nums)
    print(f"\tFour smallest elements: {smallest_four}")
    print(f"\tFour largest elements: {largest_four}")

    print("\n--- Calculating Possible Differences ---")
    differences = [
        (largest_four[0] - smallest_four[3], "Change 3 smallest"),
        (largest_four[1] - smallest_four[2], "Change 2 smallest, 1 largest"),
        (largest_four[2] - smallest_four[1], "Change 1 smallest, 2 largest"),
        (largest_four[3] - smallest_four[0], "Change 3 largest")
    ]

    print("\n--- Difference Calculations ---")
    for i, (diff, strategy) in enumerate(differences, 1):
        print(f"\n--- Calculation {i}/4 ---")
        print(f"\tStrategy: {strategy}")
        print(f"\tLargest: {largest_four[i - 1]}, Smallest: {smallest_four[4 - i]}")
        print(f"\tDifference: {largest_four[i - 1]} - {smallest_four[4 - i]} = {diff}")

    print("\n--- Difference Summary ---")
    headers = ["Strategy", "Largest", "Smallest", "Difference"]
    table_data = [
        [strategy, largest_four[i - 1], smallest_four[4 - i], diff]
        for i, (diff, strategy) in enumerate(differences, 1)
    ]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Finding Minimum Difference ---")
    min_diff = min(diff for diff, _ in differences)
    min_strategy = next(strategy for diff, strategy in differences if diff == min_diff)
    print(f"\tMinimum difference: {min_diff}")
    print(f"\tCorresponding strategy: {min_strategy}")

    print("\n--- Function Returning ---")
    print(f"\tReturning minimum difference: {min_diff}")
    return min_diff


# <-------------------------------------------------- July 4th, 2024 -------------------------------------------------->
# 2181. Merge Nodes in Between Zeros

# Given the `head` of a linked list where integers are separated by `0`s (with `0` at the beginning and end), merge
# nodes between every two consecutive `0`s into a single node with a value equal to their sum,
# and return the modified list without any `0`s.


def mergeNodes1(head: Optional[ListNode]) -> Optional[ListNode]:
    print("\n--- Input Parameters ---")
    print(f"\thead = {LinkedListVisualizer.visualize(head, file_name='mergeNodes1-Input')}")

    print("\n--- Initialization ---")
    result_node = head
    traversal_node = head.next_node
    current_sum = 0
    print(f"\tresult_node = Node with value {result_node.val}")
    print(f"\ttraversal_node = Node with value {traversal_node.val}")
    print(f"\tcurrent_sum = {current_sum}")

    print("\n--- Main Loop ---")
    iteration_data = []
    iteration = 0
    while traversal_node:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"\tCurrent state:")
        print(f"\t\tresult_node.val = {result_node.val}")
        print(f"\t\ttraversal_node.val = {traversal_node.val}")
        print(f"\t\tcurrent_sum = {current_sum}")

        print("\tDecision Point:")
        if traversal_node.val == 0:
            print(f"\t\tCondition true: traversal_node.val == 0")
            print(f"\t\tActions:")
            print(f"\t\t\t1. Moving result_node to next node")
            result_node = result_node.next_node
            print(f"\t\t\t2. Setting result_node.val to current_sum ({current_sum})")
            result_node.val = current_sum
            print(f"\t\t\t3. Resetting current_sum to 0")
            current_sum = 0
            action = "Zero found, sum applied"
            print(f"\tUpdated List State: "
                  f"{LinkedListVisualizer.visualize(head, file_name=f'mergeNodes1-Iteration{iteration}-End')}")
        else:
            print(f"\t\tCondition false: traversal_node.val != 0")
            print(f"\t\tAction: Adding traversal_node.val to current_sum")
            current_sum += traversal_node.val
            action = f"Added {traversal_node.val} to sum"

        print("\tMoving traversal_node to next node")
        traversal_node = traversal_node.next_node

        iteration_data.append([iteration, result_node.val,
                               traversal_node.val if traversal_node else "None", current_sum, action])

    print("\n--- Final List Adjustment ---")
    print(f"\tSetting result_node.next_node to None")
    result_node.next_node = None

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Result Node Val", "Traversal Node Val", "Current Sum", "Action"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    return_value = head.next_node
    print(f"\tReturning head.next_node: {LinkedListVisualizer.visualize(return_value, file_name='mergeNodes1-Output')}")
    return return_value


# <-------------------------------------------------- July 5th, 2024 -------------------------------------------------->
# 2058. Find the Minimum and Maximum Number of Nodes Between Critical Points

# Given a linked list `head`, return an array `[minDistance, maxDistance]` where `minDistance` is the minimum distance
# and `maxDistance` is the maximum distance between any two distinct critical points.
# A critical point is defined as a node that is a local maxima or minima, with values strictly greater or smaller than
# both its adjacent nodes.
# If there are fewer than two critical points, return `[-1, -1]`.


def nodesBetweenCriticalPoints1(head: Optional[ListNode]) -> List[int]:
    print("\n--- Input Parameters ---")
    input_visualization = LinkedListVisualizer.visualize(head, file_name='nodesBetweenCriticalPoints1-Input')
    print(f"\thead = {input_visualization}")

    print("\n--- Initialization ---")
    if not head.next_node.next_node:
        print("\tInsufficient nodes in the list. Returning [-1, -1]")
        return [-1, -1]

    prev_node = head
    current_node = head.next_node
    first_critical_index, last_critical_index = None, None
    min_distance = float('inf')
    current_index = 1

    print(f"\tprev_node = {prev_node.val}")
    print(f"\tcurrent_node = {current_node.val}")
    print(f"\tfirst_critical_index = {first_critical_index}")
    print(f"\tlast_critical_index = {last_critical_index}")
    print(f"\tmin_distance = {min_distance}")
    print(f"\tcurrent_index = {current_index}")

    print("\n--- Main Loop ---")
    iteration_data = []
    critical_points = []
    while current_node.next_node:
        print(f"\n--- Iteration {current_index} ---")
        current_index += 1
        print(f"\tCurrent state: prev_node = {prev_node.val}, current_node = {current_node.val}, "
              f"next_node = {current_node.next_node.val}")

        print("\tChecking for critical point:")
        is_local_maxima = prev_node.val < current_node.val > current_node.next_node.val
        is_local_minima = prev_node.val > current_node.val < current_node.next_node.val
        is_critical_point = is_local_maxima or is_local_minima

        if is_critical_point:
            print(f"\t\tCritical point found at index {current_index}")
            critical_points.append(current_index)
            if is_local_maxima:
                print(f"\t\t\tLocal maxima: {prev_node.val} < {current_node.val} > {current_node.next_node.val}")
            else:
                print(f"\t\t\tLocal minima: {prev_node.val} > {current_node.val} < {current_node.next_node.val}")

            if not first_critical_index:
                print(f"\t\tSetting first_critical_index to {current_index}")
                first_critical_index = current_index
            else:
                print(f"\t\tUpdating min_distance:")
                print(f"\t\t\tCurrent min_distance: {min_distance}")
                print(f"\t\t\tNew distance: {current_index - last_critical_index}")
                min_distance = min(min_distance, current_index - last_critical_index)
                print(f"\t\t\tUpdated min_distance: {min_distance}")

            print(f"\t\tUpdating last_critical_index to {current_index}")
            last_critical_index = current_index
        else:
            print("\t\tNot a critical point")

        print("\tUpdating nodes for next iteration:")
        print(f"\t\tprev_node: {prev_node.val} -> {current_node.val}")
        print(f"\t\tcurrent_node: {current_node.val} -> {current_node.next_node.val}")

        iteration_data.append([current_index, prev_node.val, current_node.val,
                               current_node.next_node.val if current_node.next_node else "None",
                               is_critical_point, first_critical_index, last_critical_index, min_distance])

        prev_node = current_node
        current_node = current_node.next_node

    print("\n--- Iteration Summary ---")
    headers = ["Index", "Prev", "Current", "Next", "Is Critical", "First Critical", "Last Critical", "Min Distance"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Critical Points Visualization ---")
    critical_points_visualization = LinkedListVisualizer.visualize(
        head, file_name='nodesBetweenCriticalPoints1-CriticalPoints')

    print(f"\tCritical Points: {critical_points}")
    print(f"\tVisualization: {critical_points_visualization}")

    print("\n--- Function Returning ---")
    if min_distance != float('inf'):
        max_distance = last_critical_index - first_critical_index
        print(f"\tCalculating max_distance: {last_critical_index} - {first_critical_index} = {max_distance}")
        print(f"\tFinal Result: [{min_distance}, {max_distance}]")
        return [min_distance, max_distance]
    else:
        print("\tLess than two critical points found. Returning [-1, -1]")
        return [-1, -1]


# <-------------------------------------------------- July 6th, 2024 -------------------------------------------------->
# 2582. Pass the Pillow

# Given `n` people that stand in a line from 1 to `n` passing a pillow to the next person every second, determine which
# person is holding the pillow after `time` seconds.
# When the pillow reaches either end, the direction reverses; return the index of the person holding the pillow.


def passThePillow1(n: int, time: int) -> int:
    time %= (n - 1) * 2

    if time < n:
        return time + 1
    else:
        offset = time - n + 1
        return n - offset


# <-------------------------------------------------- July 7th, 2024 -------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for July 1st, 2024
# Expected output: True
# threeConsecutiveOdds1(arr=[1, 2, 34, 3, 4, 5, 7, 23, 12])

# Test cases for July 2nd, 2024
# Expected output: [4, 9]
# intersect1(nums1=[4, 9, 5], nums2=[9, 4, 9, 8, 4])
# intersect2(nums1=[4, 9, 5], nums2=[9, 4, 9, 8, 4])

# Test cases for July 3rd, 2024
# Expected output: 4
# minDifference1(nums=[6, 5, 0, 7, 10, 4, 8, 21])

# Test cases for July 4th, 2024
# Expected output: [4, 11] -> ListNode(val=4, next_node=ListNode(val=11))
# mergeNodes1(head=ListNode(next_node=ListNode(val=3, next_node=ListNode(val=1, next_node=ListNode(next_node=ListNode(
#     val=4, next_node=ListNode(val=5, next_node=ListNode(val=2, next_node=ListNode()))))))))

# Test cases for July 5th, 2024
# Expected output: [1, 3]
# nodesBetweenCriticalPoints1(head=ListNode(val=5, next_node=ListNode(val=3, next_node=ListNode(
#     val=1, next_node=ListNode(val=2, next_node=ListNode(val=5, next_node=ListNode(
#         val=1, next_node=ListNode(val=2))))))))

# Test cases for July 6th, 2024
# Expected output: 2
passThePillow1(n=3, time=7)

# Test cases for July 7th, 2024
