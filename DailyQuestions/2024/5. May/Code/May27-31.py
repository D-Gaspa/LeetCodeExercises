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
    steps = 0
    carry = 0

    # Process bits from the second-to-last (most significant bit) to the first (least significant bit).
    # We skip the last bit as it doesn't affect carry propagation when adding 1.
    for index in range(len(s) - 1, 0, -1):
        steps += 1  # Every bit operation (division or addition) requires a step

        if s[index] == '1':  # Odd number
            # If there's no carry, adding 1 will result in a carry
            if carry == 0:
                carry = 1
                steps += 1  # Extra step due to carry propagation
        else:  # Even number
            # Only need an extra step if there was a carry from the previous (less significant) bit.
            if carry == 1:
                steps += 1

    # If there's a final carry, it represents an extra step
    return steps + carry


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
numSteps1(s_2)  # Expected output: 6
# numSteps2(s_2)  # Expected output: 6
