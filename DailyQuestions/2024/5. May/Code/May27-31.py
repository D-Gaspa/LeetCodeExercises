import bisect
from collections import defaultdict
from pprint import pprint
from typing import List

from tabulate import tabulate


# Week 5: May 27th - May 31st, 2024

# <-------------------------------------------------- May 27th, 2024 -------------------------------------------------->
# 1608. Special Array With X Elements Greater Than or Equal X

# Given a non-negative integer array nums, find if it is special; that is, find a unique number 'x' where 'x' equals the
# count of elements in nums that are greater than or equal to 'x'.
# If such 'x' exists, return it, else return -1.


def specialArray1(nums: List[int]) -> int:
    """
    Finds a special number 'x' in a non-negative integer array 'nums'
    such that 'x' equals the count of elements in 'nums' that are greater than or equal to 'x'.

    This solution uses a brute-force approach by sorting the array first.
    Then, it iterates through possible values of 'x' from 0 to the length of 'nums'.
    For each value, it counts the elements greater or equal to it and checks if that count equals 'x'.
    If found, it returns 'x'; otherwise, -1.

    The time complexity of this solution is O(n^2) due to the nested linear search within range(n+1)
    The space complexity is O(n) due to the sorting operation in Python that uses Timsort.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")  # Display the input list
    n = len(nums)

    print("\n--- Preprocessing (Sorting) ---")
    nums.sort()
    print(f"\tSorted nums = {nums}")

    print("\n--- Main Loop (Testing Potential Special Values) ---")
    for potential_special_x in range(n + 1):
        print(f"\n\tIteration {potential_special_x + 1}:")
        print(f"\t\tTesting potential_special_x = {potential_special_x}")

        # Count elements >= potential_special_x
        numbers_greater_or_equal = sum(1 for num in nums if num >= potential_special_x)
        print(f"\t\tCount of numbers >= {potential_special_x}: {numbers_greater_or_equal}")

        # Decision Point
        print(f"\t\tDecision: Is {numbers_greater_or_equal} == {potential_special_x}?")
        if numbers_greater_or_equal == potential_special_x:
            print("\t\t\tYes! Special number found.")
            print("\n--- Function Returning ---")
            print(f"Result: {potential_special_x}")
            return potential_special_x

    print("\n--- Function Returning ---")
    print("Result: -1 (No special number found)")
    return -1


def specialArray2(nums: List[int]) -> int:
    """
    Finds a special number 'x' in a non-negative integer array 'nums'
    where 'x' equals the count of elements in 'nums' that are greater than or equal to 'x'.

    The function sorts the input array 'nums' and then uses a binary search
    algorithm to find the special number 'x'.

    During the binary search, for each potential 'x' (represented by 'mid_index'), it calculates the number of items
    in 'nums' that are greater than or equal to 'x'.
    If this count equals 'x', then 'x' is returned immediately.
    If the count is more than 'x',
    the function narrows the search to the right half of the current range, else it narrows it to the left half.

    The time complexity of this solution is O(n log n), due to sorting the array and the later binary search.
    The space complexity is O(1), as there are no extra space-demanding data structures used.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")

    print("\n--- Sorting Array ---")
    nums.sort()
    print(f"\tSorted nums = {nums}")

    left_index = 0
    right_index = len(nums)
    mid_index = -1  # Initialize mid_index
    print(f"\tInitial search range: left_index = {left_index}, right_index = {right_index}")

    print("\n--- Main Loop (Binary Search for Special Number) ---")
    iteration_data = []  # To store data for iteration summary table
    while left_index <= right_index:
        mid_index = (left_index + right_index) // 2
        print(f"\n\tIteration: left_index = {left_index}, right_index = {right_index}, mid_index = {mid_index}")

        numbers_greater_or_equal = len(nums) - bisect.bisect_left(nums, mid_index)
        print(f"\t\tElements >= mid_index: {numbers_greater_or_equal}")

        # Decision Point
        print(f"\t\tDecision: numbers_greater_or_equal ({numbers_greater_or_equal}) vs. mid_index ({mid_index})")
        if numbers_greater_or_equal == mid_index:
            print("\t\tSpecial number found!")
            iteration_data.append([left_index, right_index, mid_index, numbers_greater_or_equal, "Found"])
            break  # Found the special number, exit loop
        elif numbers_greater_or_equal > mid_index:
            left_index = mid_index + 1
            print("\t\t\tSearching right half")
            iteration_data.append([left_index, right_index, mid_index, numbers_greater_or_equal, "Right"])
        else:
            right_index = mid_index - 1
            print("\t\t\tSearching left half")
            iteration_data.append([left_index, right_index, mid_index, numbers_greater_or_equal, "Left"])

    print("\n--- Iteration Summary (Binary Search Steps) ---")
    headers = ["Left Index", "Right Index", "Mid Index", "Elements >=", "Decision"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    if left_index > right_index:
        print(f"\tNo special number found, returning -1")
        return -1
    else:
        print(f"\tReturning special number: {mid_index}")
        return mid_index


def specialArray3(nums: List[int]) -> int:
    """
    Finds a special number 'x' in a non-negative integer array 'nums'
    such that 'x' equals the count of elements in 'nums' that are greater than or equal to 'x'.

    This function employs a frequency count array for optimization, which diminishes the need for a nested search.
    The function first finds the frequency of each number in 'nums' up to its length 'n'.
    Following that, it iteratively checks the potential values for 'x' in reverse order (from 'n' down to 1).

    For each 'x', it keeps a running total of the number of elements that are greater than or equal to 'x'
    (by adding the count of 'x' itself to the total).
    If this total equals 'x' at some point, 'x' is returned.
    If no such 'x' can be found, the function yields -1.

    The time complexity of this solution is O(n) due to the single pass through the input array.
    The space complexity is O(n) due to the frequency count array.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    n = len(nums)

    print("\n--- Building Frequency Count Array ---")
    frequency_count = [0] * (n + 1)  # Create an array to store frequencies

    for i, num in enumerate(nums):
        print(f"\n\tIteration {i + 1}: Processing num = {num}")
        # Store frequency at index min(num, n) to avoid going out of bounds
        frequency_count[min(num, n)] += 1
        print(f"\t\tUpdated frequency_count = {frequency_count}")

    print("\n--- Frequency Count Array (Final) ---")
    print(tabulate(enumerate(frequency_count), headers=["Number", "Frequency"], tablefmt="fancy_grid"))

    print("\n--- Searching for Special Number in Reverse ---")
    numbers_greater_or_equal = 0
    iteration_data = []  # To collect data for iteration summary
    for potential_special_x in range(n, 0, -1):
        print(f"\n\tIteration: potential_special_x = {potential_special_x}")
        numbers_greater_or_equal += frequency_count[potential_special_x]
        print(f"\t\tNumbers >= {potential_special_x}: {numbers_greater_or_equal}")

        # Decision Point:
        if numbers_greater_or_equal == potential_special_x:
            print(f"\t\t--- Special Number Found! ---")
            return potential_special_x

        # Store iteration data for summary
        iteration_data.append([potential_special_x, numbers_greater_or_equal])

    print("\n--- Iteration Summary (Reverse Search) ---")
    headers = ["Potential Special X", "Numbers >= X"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("No special number found.")
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
    # Print the input parameter for debugging
    print(f"Input Parameter: arr = {arr}")

    prefix_xor = [0]
    print("\nBuilding Prefix XOR Array:")
    table_data = [["Index", "Number", "Prefix XOR"]]  # Headers for the table
    for i, num in enumerate(arr):
        prefix_xor.append(prefix_xor[-1] ^ num)
        table_data.append([i, num, prefix_xor[-1]])  # Add data row to table
    print(tabulate(table_data, headers='firstrow', tablefmt='fancy_grid'))

    triplet_count = 0
    n = len(prefix_xor)

    print("\nChecking for Triplets:")
    for start_index in range(n):
        for end_index in range(start_index + 1, n):
            if prefix_xor[start_index] == prefix_xor[end_index]:
                count_increment = end_index - start_index - 1
                print(f"\t- Found triplet at indices ({start_index}, {end_index}):")
                print(f"\t\tPrefix XOR values match: {prefix_xor[start_index]}")
                print(f"\t\tIncrementing count by: {count_increment}")
                triplet_count += count_increment

    print("\nFinal Triplet Count:")
    pprint(triplet_count)  # Use pprint for clean output
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
    # Print the input parameter for debugging
    print(f"Input Parameter: arr = {arr}")

    triplet_count = 0
    xor_count = defaultdict(int)
    xor_index_sum = defaultdict(int)

    print("\nBuilding Prefix XOR Array:")
    prefix_xor = [0]

    table_data = []
    for i, num in enumerate(arr):
        prefix_xor.append(prefix_xor[-1] ^ num)
        table_data.append([i, num, prefix_xor[-1]])
    print(tabulate(table_data, headers=["Index", "Number", "Prefix XOR"], tablefmt="fancy_grid"))

    print("\nIterating and Counting Triplets:")
    table_data = [["Index", "Current XOR", "XOR Count", "Index Sum", "New Triplets", "Total Triplets"]]
    for index, current_xor in enumerate(prefix_xor):
        new_triplets = xor_count[current_xor] * (index - 1) - xor_index_sum[current_xor]
        triplet_count += new_triplets

        table_data.append(
            [index, current_xor, xor_count[current_xor], xor_index_sum[current_xor], new_triplets, triplet_count])

        xor_index_sum[current_xor] += index
        xor_count[current_xor] += 1

    print(tabulate(table_data, headers='firstrow', tablefmt='fancy_grid'))

    print("\nFinal Triplet Count:")
    pprint(triplet_count)
    return triplet_count


def countTriplets3(arr: List[int]) -> int:
    """
    Counts the number of triplets that can form two arrays with equal XOR.

    This solution optimizes `countTriplets2` by combining prefix XOR calculation and triplet count calculation
    in just a single pass through the array.
    It maintains a running prefix variable (`prefix_xor`) that stores the XOR
    of elements up to the current index, and gets updated with each iteration by XORing it with the current element.
    This eliminates the need for a separate prefix XOR computation step, making this approach more efficient.

    The time complexity of this solution is O(n), where 'n' is the number of elements in the input array.
    This is because we only iterate through the array once and use maps to store the XOR values.
    """
    # Print the input parameter for debugging
    print(f"Input Parameter: arr = {arr}")

    triplet_count = 0
    prefix_xor = 0
    xor_count = defaultdict(int, {0: 1})
    xor_index_sum = defaultdict(int)

    print("\nIterating and Counting Triplets:")
    table_data = [["Index", "Number", "Prefix XOR", "XOR Count", "Index Sum", "New Triplets", "Total Triplets"]]
    for index, num in enumerate(arr):
        prefix_xor ^= num
        new_triplets = xor_count[prefix_xor] * index - xor_index_sum[prefix_xor]
        triplet_count += new_triplets

        table_data.append(
            [index, num, prefix_xor, xor_count[prefix_xor], xor_index_sum[prefix_xor], new_triplets, triplet_count])

        xor_index_sum[prefix_xor] += index + 1
        xor_count[prefix_xor] += 1

    print(tabulate(table_data, headers='firstrow', tablefmt='fancy_grid'))

    print("\nFinal Triplet Count:")
    pprint(triplet_count)
    return triplet_count


# <-------------------------------------------------- May 31st, 2024 -------------------------------------------------->
# 260. Single Number III

# Given an integer array `nums`, in which exactly two elements appear only once and all the other elements appear
# exactly twice, return the two elements (in any order) that appear only once.

def singleNumber1(nums: List[int]) -> List[int]:
    """
    Finds the two elements that appear only once in the given integer array.

    This solution uses a dictionary to store the count of each number in the input array.
    It then filters the dictionary to find the elements that have a count of 1, which are the unique elements.

    The time complexity of this solution is O(n), where 'n' is the number of elements in the input array.
    This is because we only iterate through the array and the dictionary once, hence the time complexity is linear.
    The space complexity is also O(n) because the dictionary can store up to n unique numbers.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums: {nums}")

    element_counts = {}
    iteration_data = []

    print("\n--- Main Loop (Building Count Dictionary) ---")
    for idx, num in enumerate(nums):
        print(f"\tIteration {idx + 1}:")
        print(f"\t\tNumber: {num}")

        if num in element_counts:
            element_counts[num] += 1
            print(f"\t\tIncremented count for {num}: {element_counts[num]}")
        else:
            element_counts[num] = 1
            print(f"\t\tAdded new element {num}: {element_counts[num]}")

        # Data for iteration table
        iteration_data.append([idx + 1, num, element_counts[num]])

    print("\n--- Iteration Summary (Element Counts) ---")
    headers = ["Iteration", "Number", "Count"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Filtering for Unique Elements ---")
    print(tabulate(element_counts.items(), headers=["Element", "Count"], tablefmt="fancy_grid"))
    unique_elements = [num for num, count in element_counts.items() if count == 1]

    print("\n--- Function Returning ---")
    print(f"\tUnique Elements: {unique_elements}")

    return unique_elements


def singleNumber2(nums: List[int]) -> List[int]:
    """
    Finds the two elements that appear only once in the given integer array.

    This solution uses XOR properties to isolate two unique elements in an array.
    By XORing all numbers, duplicates cancel out (a ^ a = 0), leaving the XOR of the two unique numbers.
    This result has at least one set bit, differentiating the unique values.

    Isolating this rightmost-set bit using bitwise operations (differentiating_bit = combined_xor & -combined_xor)
    allows partitioning the array into two groups: numbers with this bit set and numbers without.
    This works because the unique numbers have different values at this bit.
    Within each group, XORing the elements eliminates duplicates, revealing the unique number within that group.

    The time complexity of this solution is O(n), where 'n' is the number of elements in the input array.
    This is because the function iterates through the array twice - once to find the cumulative XOR
    and once to separate the numbers into two groups.
    We improve the space complexity to O(1) by not using any additional data structures.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")  # Display the input array

    print("\n--- Main Loop (Finding cumulative XOR) ---")
    combined_xor = 0
    iteration_data_xor = []
    for i, num in enumerate(nums):
        combined_xor ^= num
        iteration_data_xor.append([i + 1, f"{num} ({num:b})", f"{combined_xor:b}"])
        print(f"\nIteration {i + 1}: XORing {num} ({num:b})")
        # Get the binary representation of the number
        print(f"\tCombined XOR At This Point: {combined_xor:b}")

    print("\n--- Iteration Summary (Cumulative XOR at each step) ---")
    headers = ["Iteration", "Number", "Cumulative XOR (Binary)"]
    print(tabulate(iteration_data_xor, headers=headers, tablefmt="fancy_grid"))

    differentiating_bit = combined_xor & -combined_xor
    print(f"\nDifferentiating Bit: {differentiating_bit:b}")

    num1, num2 = 0, 0
    iteration_data = []
    print("\n--- Second Loop (Separating numbers into two groups) ---")
    for i, num in enumerate(nums):
        if num & differentiating_bit:
            num1 ^= num
            group = "Group 1"
        else:
            num2 ^= num
            group = "Group 2"
        iteration_data.append([i + 1, f"{num} ({num:b})", group, f"{num1} ({num1:b})", f"{num2} ({num2:b})"])
        print(f"\nIteration {i + 1}: Separating {num} ({num:b}) into {group}")
        print(f"\tNum1 and Num2 At This Point: {num1} ({num1:b}), {num2} ({num2:b})")

    print("\n--- Iteration Summary (Numbers in Groups) ---")
    headers = ["Iteration", "Number", "Group", "Num1", "Num2"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    result = [num1, num2]
    print("\n--- Function Returning ---")
    print(f"Function Return Value: {result}")
    return result


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for May 27th, 2024
nums = [0, 4, 3, 0, 4]
# specialArray1(nums)  # Expected output: 3
# specialArray2(nums)  # Expected output: 3
# specialArray3(nums)  # Expected output: 3

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
# countTriplets1(arr)  # Expected output: 4
# countTriplets2(arr)  # Expected output: 4
# countTriplets3(arr)  # Expected output: 4

# Test cases for May 31st, 2024
nums_2 = [1, 2, 1, 3, 2, 5]
# singleNumber1(nums_2)  # Expected output: [3, 5]
# singleNumber2(nums_2)  # Expected output: [3, 5]
