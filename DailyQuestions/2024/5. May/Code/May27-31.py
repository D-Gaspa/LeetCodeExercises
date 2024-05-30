from collections import defaultdict
from pprint import pprint
from typing import List

from tabulate import tabulate


# Week 5: May 27th - May 31st

# <-------------------------------------------------- May 27th, 2024 -------------------------------------------------->
# 1608. Special Array With X Elements Greater Than or Equal X

# Given a non-negative integer array nums, find if it is special; that is, find a unique number 'x' where 'x' equals the
# count of elements in nums that are greater than or equal to 'x'.
# If such 'x' exists, return it, else return -1.


def specialArray1(nums: List[int]) -> int:
    """
    Finds a special number 'x' in the given integer array.

    This solution uses a brute-force approach to find the special number 'x'.
    The time complexity of this solution is O(n log(n)), where 'n' is the number of elements in the input array.
    """
    nums.sort()
    n = len(nums)

    for potential_special_x in range(n + 1):
        # Count how many numbers are greater than or equal to the potential special number 'x'
        numbers_greater_or_equal = sum(1 for num in nums if num >= potential_special_x)

        if numbers_greater_or_equal == potential_special_x:
            return potential_special_x

    return -1


# <-------------------------------------------------- May 28th, 2024 -------------------------------------------------->
# 1208. Get Equal Substrings Within a Budget

# Given two strings `s` and `t` of the same length and a budget `max_cost`, find the length of the longest substring
# of `s` you can transform into the corresponding substring of `t` by changing characters, where the cost of changing
# each character is the absolute difference of their ASCII values, and the total cost cannot exceed `max_cost`.


def equalSubstring1(s: str, t: str, max_cost: int) -> int:
    """
    Finds the length of the longest substring of 's' that can be transformed into the corresponding substring of 't'
    within the given budget 'max_cost'.

    This solution uses a sliding window approach to find the longest substring.
    The time complexity of this solution is O(n), where 'n' is the length of the input strings.
    """
    # Print the input parameters for debugging
    print(f"Input Parameters: s = {s}, t = {t}, max_cost = {max_cost}")

    n = len(s)
    cost = [abs(ord(s[i]) - ord(t[i])) for i in range(n)]

    print("\n--- Initialization ---")
    print("String Length (n):", n)
    print("Cost Array:")
    pprint(cost)

    start_index = 0
    total_cost = 0
    max_length = 0

    print("\n--- Sliding Window Iterations ---")
    table_data = []
    for end_index in range(n):
        total_cost += cost[end_index]

        current_substring_s = s[start_index: end_index + 1]
        current_substring_t = t[start_index: end_index + 1]

        table_data.append([end_index, total_cost, start_index, max_length, current_substring_s, current_substring_t])

        while total_cost > max_cost:
            total_cost -= cost[start_index]
            start_index += 1
            table_data[-1][2] = start_index
            current_substring_s = s[start_index: end_index + 1]
            current_substring_t = t[start_index: end_index + 1]
            table_data[-1][4] = current_substring_s
            table_data[-1][5] = current_substring_t

        max_length = max(max_length, end_index - start_index + 1)
        table_data[-1][3] = max_length

        # Print table for each iteration
        print(tabulate(table_data, headers=["End Index", "Total Cost", "Start Index", "Max Length", "Substring (s)",
                                            "Substring (t)"], tablefmt="fancy_grid"))

    print("\n--- Final Result ---")
    print("Maximum Substring Length:", max_length)

    return max_length


# <-------------------------------------------------- May 29th, 2024 -------------------------------------------------->
# 1404. Number of Steps to Reduce a Number in Binary Representation to One

# Given a string `s` representing a binary number, determine the number of steps needed to reduce it to the value of 1
# by repeatedly dividing even numbers by 2 and adding 1 to odd numbers.


def numSteps1(s: str) -> int:
    """
    Determines the number of steps needed to reduce the binary number 's' to 1.

    This solution converts the binary number to an integer and simulates the process of reducing it to 1.
    The time complexity of this solution is O(n), where 'n' is the length of the binary number.
    """
    # Print the input parameter for debugging
    print(f"Input Parameter: s = {s}")

    num = int(s, 2)
    steps = 0

    print(f"\nInitial Binary: {s}  (Decimal: {num})")
    print("\n--- Reduction Steps ---\n")

    table_data = []

    while num != 1:
        operation = "Divide by 2" if num % 2 == 0 else "Add 1"
        next_num = num // 2 if num % 2 == 0 else num + 1

        binary_num = bin(num)[2:]
        binary_next_num = bin(next_num)[2:]

        table_data.append([steps, binary_num, num, operation, binary_next_num, next_num])

        num = next_num
        steps += 1

    print(tabulate(table_data, headers=["Step", "Binary", "Decimal", "Operation", "New Binary", "New Decimal"],
                   tablefmt="fancy_grid"))

    print(f"\nFinal Result: {steps} steps\n")
    return steps


def numSteps2(s: str) -> int:
    """
    Determines the number of steps needed to reduce the binary number 's' to 1.

    This approach simulates the process by working with bits in reverse order.
    The time complexity of this solution is O(n), where 'n' is the length of the binary number.
    """
    # Print the input parameter for debugging
    print(f"Input Parameter: s = {s}")

    steps = 0
    carry = 0
    table_data = []

    print("\n--- Bit-by-Bit Iterations (Right-to-Left, Skipping Last Bit) ---")
    for index in range(len(s) - 1, 0, -1):
        print(f"\nProcessing Bit at Index {index} ('{s[index]}'):")
        steps_this_iteration = 1  # Initial step for processing the bit

        if s[index] == '1':
            print("  Odd Number Detected")
            if carry == 0:
                carry = 1
                print("  Carry Generated:", carry)
                steps_this_iteration += 1
            else:
                print("  Existing Carry Maintained:", carry)
        else:
            print("  Even Number Detected")
            if carry == 1:
                print("  Carry Consumed:", carry)
                steps_this_iteration += 1
            else:
                print("  No Carry to Consume")

        steps += steps_this_iteration  # Update total steps

        # Add current state to table
        table_data.append([index, s[index], carry, steps_this_iteration, steps])

    print("\n--- Iteration Summary ---")
    print(tabulate(table_data, headers=["Index", "Bit", "Carry", "Steps (Iteration)", "Total Steps"],
                   tablefmt="fancy_grid"))

    if carry == 1:
        print(f"\n--- Final Carry Requires One More Step ---")
        steps += 1

    print(f"\n--- Total Steps to Reduce '{s}' to '1' ---")
    print(steps)

    return steps


# <-------------------------------------------------- May 30th, 2024 -------------------------------------------------->
# 1442. Count Triplets That Can Form Two Arrays of Equal XOR

# Given an integer array `arr`, count the number of triplets of indices `(i, j, k)` where `0 <= i < j <= k <
# arr.length` and the bitwise XOR of elements from `i` to `j-1` equals the bitwise XOR from `j` to `k`.


def countTriplets1(arr: List[int]) -> int:
    """
    Counts the number of triplets that can form two arrays of equal XOR.

    This solution uses a prefix XOR array to calculate the XOR values efficiently.
    If we consider the XOR from index 0 to indices i and j respectively (where i < j),
    and find that those XORs are equal, then the XOR from index i+1 to j must be 0.
    This is because the XOR of a number with itself is 0. We can deduce it follows that
    the array can be sliced at index i+1 to form two subarrays with equal XORs.

    The time complexity of this solution is O(n^2), where 'n' is the number of elements in the array.
    This is because for each starting index, we are looping through the rest of the array.
    """
    # Generate prefix xor array
    prefix_xor = [0]
    for num in arr:
        prefix_xor.append(prefix_xor[-1] ^ num)  # Calculate XOR value and append to prefix XOR array

    triplet_count = 0
    n = len(prefix_xor)

    # Iterate over potential triplets
    for start_index in range(n):
        for end_index in range(start_index + 1, n):
            if prefix_xor[start_index] == prefix_xor[end_index]:
                # Triplet count is incremented by the number of elements between start_index and end_index.
                # This is because the subarray from start_index+1 to end_index can be partitioned in multiple ways.
                triplet_count += end_index - start_index - 1

    return triplet_count


def countTriplets2(arr: List[int]) -> int:
    """
    Counts the number of triplets that can form two arrays of equal XOR.

    This solution adopts a more efficient approach than `countTriplets1` by computing a prefix XOR array in a single
    pass, keeping track of the number of occurrences and cumulative indices of each XOR value encountered.
    However, with `countTriplets2`, we don't explicitly calculate every triplet.
    Instead, for each XOR value, we calculate how many new triplets are added based on already encountered XORs.

    Specifically, for each previously encountered 'current_xor', we can form (index - 1) new triplets.
    This is because the longer the array with the same XOR,
    the more divisions there are that can split the array into triplets with equal XORs.
    Since we also have to account for over-counting of previous indices, we subtract the sum of indices where
    the current XOR value occurred.

    The time complexity of this solution is O(n), where 'n' is the number of elements in the input array.
    This is because we are iterating through the array only once and using maps to store the XOR values.
    """
    # Generate prefix xor array
    prefix_xor = [0]
    for num in arr:
        prefix_xor.append(prefix_xor[-1] ^ num)  # Calculate XOR value and append to prefix XOR array

    triplet_count = 0
    n = len(prefix_xor)

    xor_count = defaultdict(int)  # Frequency of each prefix XOR value
    xor_index_sum = defaultdict(int)  # Sum of indices where each prefix XOR value occurred

    for index in range(n):
        current_xor = prefix_xor[index]

        # Calculate triplet count for current XOR value
        triplet_count += xor_count[current_xor] * (index - 1) - xor_index_sum[current_xor]

        # Update maps for the current XOR
        xor_index_sum[current_xor] += index
        xor_count[current_xor] += 1

    return triplet_count


def countTriplets3(arr: List[int]) -> int:
    pass


# <-------------------------------------------------- May 31st, 2024 -------------------------------------------------->


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for May 27th, 2024
nums = [0, 4, 3, 0, 4]
# specialArray1(nums)  # Expected output: 3

# Test cases for May 28th, 2024
s = "abcd"
t = "bcdf"
max_cost = 3
# equalSubstring1("s", "t", max_cost)  # Expected output: 3

# Test cases for May 29th, 2024
s_2 = "1101"
# numSteps1(s_2)  # Expected output: 6
# numSteps2(s_2)  # Expected output: 6

# Test cases for May 30th, 2024
arr = [2, 3, 1, 6, 7]
arr_2 = [1, 1, 1, 1, 1]
# countTriplets2(arr)  # Expected output: 4
# countTriplets2(arr_2)  # Expected output: 10
