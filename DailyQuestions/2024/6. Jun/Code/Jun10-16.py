import heapq
from collections import defaultdict
from typing import List

from tabulate import tabulate


# June 2024, Week 3: June 10th - June 16th

# <------------------------------------------------- June 10th, 2024 ------------------------------------------------->
# 1051. Height Checker

# Given an integer array heights representing the current order of students' heights in line,
# return the number of indices where heights[i] does not match the expected non-decreasing order of heights.

def heightChecker1(heights: List[int]) -> int:
    print("\n--- Input Parameters ---")
    print(f"\theights = {heights}")

    print("\n--- Initialization ---")
    expected_heights = heights[:]
    print(f"\texpected_heights (copy of heights) = {expected_heights}")

    print("\n--- Counting Sort Process ---")
    frequency_map = defaultdict(int)
    print("\tCreating frequency map:")
    for height in expected_heights:
        frequency_map[height] += 1
        print(f"\t\tAdded height {height}, frequency_map[{height}] = {frequency_map[height]}")

    print(tabulate(frequency_map.items(), headers=["Height", "Frequency"], tablefmt="fancy_grid"))

    min_height, max_height = min(expected_heights), max(expected_heights)
    print(f"\tmin_height = {min_height}, max_height = {max_height}")

    sorted_index = 0
    print(f"\n\tSorting process from {min_height} to {max_height}:")
    for height in range(min_height, max_height + 1):
        print(f"\n\t--- Processing height {height} ---")
        print(f"\tThe frequency of {height} in heights: {frequency_map[height]}")
        for _ in range(frequency_map[height]):
            print(f"\t\tPlacing {height} at index {sorted_index}")
            expected_heights[sorted_index] = height
            sorted_index += 1
        print(f"\t\tCurrent expected_heights: {expected_heights}")

    print("\n--- Comparison and Mismatch Counting ---")
    mismatch_count = 0
    comparison_data = []
    for i, (current_height, expected_height) in enumerate(zip(heights, expected_heights)):
        print(f"\n\t--- Comparing index {i} ---")
        print(f"\tCurrent height: {current_height}, Expected height: {expected_height}")
        if current_height != expected_height:
            mismatch_count += 1
            print(f"\t\tMismatch found! Mismatch count: {mismatch_count}")
        else:
            print("\t\tHeights match.")
        comparison_data.append([i, current_height, expected_height, "Mismatch" if current_height != expected_height else "Match"])

    print("\n--- Comparison Summary ---")
    headers = ["Index", "Current Height", "Expected Height", "Status"]
    print(tabulate(comparison_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal mismatch count: {mismatch_count}")
    return mismatch_count


# <------------------------------------------------- June 11th, 2024 ------------------------------------------------->
# 1122. Relative Sort Array

# Given two arrays `arr1` and `arr2`, where `arr2` elements are distinct, and all elements in `arr2` are also in `arr1`.
# Sort the elements of `arr1` such that the relative ordering of items in `arr1` is the same as in `arr2`.
# Elements that do not appear in `arr2` should be placed at the end of `arr1` in ascending order.


def relativeSortArray1(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Sorts arr1 such that elements are ordered as in arr2, with remaining elements in ascending order.

    This function uses a defaultdict to store the frequency of each element in arr1.
    It iterates through arr2, appending elements to the result list based on their frequency.
    Remaining elements not found in arr2 are sorted and appended at the end.

    The time complexity of this solution is O(n + m + r log r), where n is the length of arr1, m is the length of arr2,
    and r is the number of elements in arr1 that are not present in arr2.
    This is because we iterate through arr1 to count frequencies O(n), then iterate through arr2 to add elements
    O(m), and finally sort the remaining elements O(r log r).
    Here, r can vary from 0 to n, so the worst-case time complexity is O(n log n).
    The space complexity is O(n) to store the frequency counts in the hashmap and the result list.
    """
    print("\n--- Input Parameters ---")
    print(f"\tarr1 = {arr1}")
    print(f"\tarr2 = {arr2}")

    print("\n--- Building Frequency Dictionary ---")
    counts_dict = defaultdict(int)
    for num in arr1:
        counts_dict[num] += 1
        print(f"\tProcessed {num}, counts_dict: {dict(counts_dict)}")  # Show updated dictionary

    print("\n--- Appending Elements from arr2 ---")
    result = []
    for num in arr2:
        print(f"\n\tProcessing num {num} with count: {counts_dict[num]}")
        while counts_dict[num] > 0:
            result.append(num)
            counts_dict[num] -= 1
            print(f"\t\tAppended {num}, updated result: {result}, counts_dict[{num}]: {counts_dict[num]}")
    print("\n\tResult after processing arr2:", result)

    print("\n--- Collecting Remaining Elements Not in arr2 ---")
    remaining = []
    for num, count in counts_dict.items():
        if count > 0:  # Check if the count is greater than 0
            remaining.extend([num] * count)  # Append as many times as its count
            print(f"\t\tAdded {num} (count: {count}) to remaining: {remaining}")

    print("\n--- Sorting Remaining Elements ---")
    remaining.sort()  # In-place sort
    print(f"\tSorted remaining: {remaining}")
    result.extend(remaining)
    print(f"\tFinal Result: {result}")

    print("\n--- Function Returning ---")
    return result


def relativeSortArray2(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Sorts arr1 such that elements are ordered as in arr2, with remaining elements in ascending order.

    This function employs a counting sort strategy.
    It first determines the maximum value in arr1 to create a count array that can store frequencies of all elements
    After counting element occurrences in arr1,
    it constructs the sorted result by iterating over arr2 and appending elements based on their frequencies.
    Remaining elements not in arr2 are then added in ascending order.

    The time complexity of this solution is O(n + m + k), where n is the length of arr1, m is the length of arr2,
    and k is the range of values in arr1 (max value + 1).
    Finding the maximum value takes O(n) time, and counting frequencies takes O(n).
    Adding elements based on arr2 takes O(n + m) time, and adding remaining elements takes O(n + k) time.
    Hence, the overall time complexity is O(n + m + k).

    The space complexity is O(k) to store the count array.
    """
    print("\n--- Input Parameters ---")
    print(f"\tarr1 = {arr1}")
    print(f"\tarr2 = {arr2}")

    print("\n--- Finding Maximum Value in arr1 ---")
    max_val = max(arr1)
    print(f"\tMax value: {max_val}")

    print("\n--- Building Count Array ---")
    count = [0] * (max_val + 1)
    for num in arr1:
        count[num] += 1
        print(f"\tProcessed {num}, count: {count[num]}")  # Show an updated count array

    print("\n\tCount Array:", count)

    print("\n--- Appending Elements from arr2 Based on Count ---")
    result = []
    for num in arr2:
        print(f"\n\tProcessing num {num}")
        print(f"\t\tCount of {num}: {count[num]}")
        while count[num] > 0:
            result.append(num)
            count[num] -= 1
            print(f"\t\tAppended {num}, updated result: {result}, count[{num}]: {count[num]}")
    print("\n\tResult after processing arr2:", result)

    print("\n--- Appending Remaining Elements in Ascending Order ---")
    for num in range(max_val + 1):
        while count[num] > 0:
            result.append(num)
            count[num] -= 1
            print(f"\t\tAppended {num}, updated result: {result}, count[{num}]: {count[num]}")
    print(f"\n\tFinal Result: {result}")

    print("\n--- Function Returning ---")
    return result


# <------------------------------------------------- June 12th, 2024 ------------------------------------------------->
# 75. Sort Colors

# Given an array `nums` with `n` objects colored red, white, or blue, sort them in-place (without using built-in sort)
# so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
# We will use the integers `0`, `1`, and `2` to represent the color red, white, and blue, respectively.


def sortColors1(nums: List[int]) -> None:
    """
    Sorts an array of integers containing only 0, 1, and 2 in-place.

    The function uses an algorithm similar to counting sort to do in-place sorting.
    It first counts the frequency of each color (0, 1, and 2 representing red, white, and blue respectively)
    and stores it in the 'color_counts' list.
    Subsequently, it rebuilds the `nums` list by placing the correct number of each color in the appropriate order.
    This approach avoids comparisons and leverages the limited value range for an efficient sorting process.

    The time complexity of this solution is O(n) due to two linear iterations over the input list `nums`:
    one for counting and another for reconstruction.
    The space complexity is O(1) as it uses a fixed-size list `color_counts` to store counts of three colors.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")  # Display the input list

    print("\n--- Main Loop (Counting Colors) ---")
    color_counts = [0, 0, 0]  # Initialize color counts
    iteration_data = []  # Collect data for iteration summary

    for i, num in enumerate(nums):
        print(f"\n\tIteration {i + 1}:")
        print(f"\t\tNumber: {num}")
        color_counts[num] += 1
        print(f"\t\tColor Counts (0=Red, 1=White, 2=Blue): {color_counts}")
        iteration_data.append([i + 1, num, color_counts.copy()])

    print("\n--- Iteration Summary (Color Counts) ---")
    headers = ["Iteration", "Number", "Color Counts"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Main Loop (Reconstructing Array) ---")
    index = 0
    for color in range(3):
        print(f"\n\tColor {color}:")
        for _ in range(color_counts[color]):
            print(f"\t\tPlacing color {color} at index {index}")
            nums[index] = color
            index += 1
            print(f"\t\tUpdated nums: {nums}")

    print("\n--- Function Returning ---")
    print(f"\tSorted nums: {nums}")


def sortColors2(nums: List[int]) -> None:
    """
    Sorts an array of integers containing only 0, 1, and 2 in-place.

    The function uses a variation of the three-way partitioning quicksort algorithm, often referred to as the
    Dutch national flag problem, to sort the array in-place.
    It maintains three pointers: 'left_index' to track the position to place the next '0' (red), 'current_index'
    for the currently evaluating number, and 'right_index' to place the next '2' (blue).
    If the current number is '0', it swaps this number with the number at 'left_index' position,
    then moves both 'left_index' and 'current_index' one step to the right.
    If it's '1', it leaves the number in place and just moves 'current_index'.
    If it's '2', it swaps this number with the number at 'right_index' position and decrements 'right_index'.

    The time complexity of this solution is O(n) because we perform a single pass over the list 'nums'.
    The space complexity is O(1) because we only used a few integer variables and didn't use any additional
    data structure that scales with the size of the input.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")  # Display an input list

    print("\n--- Main Loop (Dutch National Flag Algorithm) ---")
    left_index = 0  # Position to place the next '0'
    current_index = 0  # Current index being evaluated
    right_index = len(nums) - 1  # Position to place the next '2'
    iteration_data = [[left_index, current_index, right_index, nums.copy()]]

    while current_index <= right_index:
        print(f"\n\tWhile current_index ({current_index}) <= right_index ({right_index}):")
        print(f"\t\tleft_index = {left_index}, current_index = {current_index}, right_index = {right_index}")
        print(f"\t\tnums = {nums}")

        if nums[current_index] == 0:
            print("\t\tCondition: nums[current_index] == 0")
            print(f"\t\tSwapping 0 with element at left_index ({nums[left_index]})")
            print(f"\t\tIncrementing left_index {left_index} and current_index {current_index} by 1")
            nums[left_index], nums[current_index] = nums[current_index], nums[left_index]
            left_index += 1
            current_index += 1
        elif nums[current_index] == 1:
            print("\t\tCondition: nums[current_index] == 1")
            print("\t\tSkipping 1 (already in correct position)")
            print(f"\t\tIncrementing current_index {current_index} by 1")
            current_index += 1
        else:  # nums[current_index] == 2
            print("\t\tCondition: nums[current_index] == 2")
            print(f"\t\tSwapping 2 with element at right_index ({nums[right_index]})")
            print(f"\t\tDecrementing right_index {right_index} by 1")
            nums[current_index], nums[right_index] = nums[right_index], nums[current_index]
            right_index -= 1

        iteration_data.append([left_index, current_index, right_index, nums.copy()])

    print("\n--- Iteration Summary (Pointer Positions and Array State) ---")
    headers = ["left_index", "current_index", "right_index", "nums"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tSorted nums: {nums}")


# <------------------------------------------------- June 13th, 2024 ------------------------------------------------->
# 2037. Minimum Number of Moves to Seat Everyone

# Given two arrays seats and students of length n, representing the positions of n seats and n students in a room,
# you can adjust the position of each student by 1 unit any number of times.
# The task is to find the minimum number of moves required to seat all students such that no two students share the
# same seat, even if initially, multiple seats or students may occupy the same position.


def minMovesToSeat1(seats: List[int], students: List[int]) -> int:
    """
    Calculates the minimum number of moves required to seat each student in a corresponding seat.

    This function first sorts both the `seats` and `students` lists in ascending order.
    Then, it iterates through each corresponding seat-student pair,
    calculates the absolute difference between their positions, and accumulates this into the `moves` variable.
    The underlying idea is that the optimal strategy involves matching the lowest-ranked student with the lowest-ranked
    seat, the second-lowest with the second-lowest, and so on.

    The time complexity of this solution is O(n log n) due to the sorting of the lists.
    The iteration and calculations within the loop take O(n), but the sorting dominates the overall complexity.
    The space complexity is O(n) due to the implementation of Python's `sort()` method, which may use up to O(n)
    additional space in some cases (especially for versions prior to Python 3.11).
    """
    print("\n--- Input Parameters ---")
    print(f"\tseats = {seats}")
    print(f"\tstudents = {students}")

    print("\n--- Sorting Seats and Students ---")
    seats.sort()
    students.sort()
    print(f"\tSorted seats: {seats}")
    print(f"\tSorted students: {students}")

    print("\n--- Calculating Moves ---")
    moves = 0
    iteration_data = []  # Collect data for iteration summary
    for i, (seat, student) in enumerate(zip(seats, students)):
        move = abs(student - seat)
        moves += move
        print(f"\tIteration {i + 1}:")
        print(f"\t\tSeat: {seat}, Student: {student}, Move: {move}, Total Moves: {moves}")
        iteration_data.append([i + 1, seat, student, move, moves])

    print("\n--- Iteration Summary (Moves) ---")
    headers = ["Iteration", "Seat", "Student", "Move", "Total Moves"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tMinimum Moves: {moves}")
    return moves


def minMovesToSeat2(seats: List[int], students: List[int]) -> int:
    """
    Calculates the minimum number of moves required to seat each student in a corresponding seat.

    This function uses a counting approach where it first finds the maximum position among both seats and students.
    Then, it creates a list `differences` to track the net difference (seats - students) at each position.
    Positive values indicate excess seats, negative values indicate excess students.
    By iterating through the differences, the function keeps track of the cumulative mismatch (`unmatched`)
    and adds the absolute value of this mismatch to the total `moves`.
    This works because each unmatched student at a position needs to be moved, and the absolute value
    of the mismatch represents the minimum distance they need to travel to find an available seat.

    The time complexity is O(n + max(seats, students)) because we iterate over both lists once and then over a list
    whose size depends on the maximum value in either input list.
    In many cases, this might be faster than the O(n log n) sorting solution.
    The space complexity is O(max(seats, students)) due to the `differences` list, which scales with the maximum value
    in the input lists.
    """
    print("\n--- Input Parameters ---")
    print(f"\tseats = {seats}")
    print(f"\tstudents = {students}")

    max_position = max(max(seats), max(students))
    print(f"\n--- Maximum Position: {max_position} ---")

    # Stores difference between the number of seats and students at each position
    differences = [0] * (max_position + 1)
    print(f"\n--- Initialized Differences Array: {differences} ---")

    print("\n--- Updating Differences ---")
    diff_iteration_data = []
    for i, (seat, student) in enumerate(zip(seats, students)):
        differences[seat] += 1
        differences[student] -= 1
        print(f"\tIteration {i + 1}:  seat={seat}, student={student}, differences={differences}")
        diff_iteration_data.append([i + 1, seat, student, differences.copy()])  # Copy is important here

    print("\n--- Iteration Summary (Differences) ---")
    headers = ["Iteration", "Seat", "Student", "Differences"]
    print(tabulate(diff_iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Calculating Moves ---")
    moves, unmatched = 0, 0
    move_iteration_data = []
    for i, difference in enumerate(differences):
        unmatched += difference
        moves += abs(unmatched)
        print(f"\tPosition {i}: difference = {difference}, unmatched = {unmatched}, moves = "
              f"{moves - abs(unmatched)} + abs({unmatched}) = {moves}")
        move_iteration_data.append([i, difference, unmatched, moves])

    print("\n--- Iteration Summary (Moves Calculation) ---")
    headers = ["Position", "Difference", "Unmatched", "Moves"]
    print(tabulate(move_iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tMinimum Moves: {moves}")
    return moves


# <------------------------------------------------- June 14th, 2024 ------------------------------------------------->
# 945. Minimum Increment to Make Array Unique

# You are given an integer array nums.
# In one move, you can pick an index i where 0 <= i < nums.length and increment nums[i] by 1.
# Return the minimum number of moves to make every value in nums unique.


def minIncrementForUnique1(nums: List[int]) -> int:
    """
    Calculates the minimum number of increments needed to make all elements in a list unique.

    The function sorts the list of numbers first, and uses a variable 'prev' to keep track of the previous number.
    Then it iterates over the sorted numbers, and for each number, it checks if it is smaller or equal to 'prev'.
    If it is, that means this number is a duplicate and the function increments 'moves' by the difference between
    'prev' and this number plus one, effectively making the duplicate a new unique number one greater than 'prev'.
    The 'prev' is then updated to this new unique number.
    If the number is not smaller or equal to 'prev', 'prev' is simply updated to this number.
    In the end, the function returns the total number of increments ('moves').

    The time complexity of this function is O(n log n) due to the sorting operation,
    where n is the length of the input list.
    The space complexity is O(n) due to the space required to sort 'nums'.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")

    moves = 0
    prev = -1
    iteration_data = []  # To store data for iteration summary

    nums.sort()  # Sort the input list
    print("\n--- Sorted List ---")
    print(f"\tSorted nums: {nums}")

    print("\n--- Main Loop (Making Elements Unique) ---")
    for idx, num in enumerate(nums):
        print(f"\nIteration {idx + 1}:")
        print(f"\tNumber: {num}")
        print(f"\tPrevious Value (prev): {prev}")
        increment = 0

        print(f"\t--- Decision Point: Checking if {num} <= {prev} ---")
        if num <= prev:
            increment = prev - num + 1
            print(f"\t\tDuplicate found. Incrementing by {prev} - {num} + 1 = {increment}")
            moves += increment
            prev += 1  # Update prev to the new unique value
        else:
            print(f"\t\tNo duplicate. Moving to next number.")
            prev = num

        print(f"\tUpdated prev: {prev}")
        print(f"\tTotal moves so far: {moves}")

        # Collect data for iteration summary
        iteration_data.append([idx + 1, num, prev, increment if num <= prev else 0, moves])

    print("\n--- Iteration Summary (Increment Details) ---")
    headers = ["Iteration", "Number", "Prev", "Increment", "Total Moves"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Minimum Increments: {moves}")
    return moves


def minIncrementForUnique2(nums: List[int]) -> int:
    """
    Calculates the minimum number of increments needed to make all elements in a list unique.

    This function uses a counting-based approach.
    It creates an array `num_counts` where the index represents a number and the value of that index is the count of
    occurrences of that number in the input list `nums`.
    It then iterates through `num_counts`, propagating excess duplicates to the next index (incrementing the number)
    and tracking the total `moves` needed.
    If there are duplicates at the last index, it calculates the sum of consecutive increments required to make them
    unique using the formula for the sum of first n natural numbers.

    The time complexity of this function is O(n + m), where n is the length of the input list and m is the maximum
    value in the list `max(nums)`.
    This is because we iterate through the list once and potentially through the `num_counts` array once.
    The space complexity is O(m), where m is the maximum value in the list, due to the size of the `num_counts` array.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")

    moves = 0
    max_num = max(nums)
    num_counts = [0] * (max_num + 1)

    iteration_data = []

    print("\n--- Main Loop (Building Count Dictionary) ---")
    for index, num in enumerate(nums):
        print(f"\n\tIteration {index + 1}:")
        print(f"\t\tNumber: {num}")

        num_counts[num] += 1
        print(f"\t\tIncrementing count for {num}: {num_counts}")
        iteration_data.append([index + 1, num, num_counts.copy()])

    print("\n--- Iteration Summary (num_counts Updates) ---")
    headers = ["Iteration", "Number", "num_counts"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Main Loop (Propagating Duplicates) ---")
    for index in range(len(num_counts) - 1):
        print(f"\n\tIndex: {index}")
        print(f"\t\tCount at index: {num_counts[index]}")

        print("\t--- Decision Point (Duplicate Check) ---")
        if num_counts[index] <= 1:
            print(f"\t\tNo duplicates or single occurrence. Moving to next index.")
            continue

        print(f"\t\tNum_counts before increment: {num_counts}")

        duplicates = num_counts[index] - 1
        print(f"\t\t{duplicates} duplicates found.")
        num_counts[index + 1] += duplicates
        print(f"\t\tPropagating {duplicates} duplicates to the next index ({index + 1}): {num_counts}")
        moves += duplicates

        print(f"\t\tTotal moves so far: {moves}")

    print("\n--- Final Check (Last Index Duplicates) ---")
    if num_counts[-1] > 1:
        n = num_counts[-1] - 1
        print(f"\t{n} duplicates found at the last index.")
        moves += n * (n + 1) // 2
        print(f"\t\tIncrementing moves by {n * (n + 1) // 2} (sum of first {n} natural numbers)")

    print("\n--- Function Returning ---")
    print(f"\tFinal Minimum Increments: {moves}")
    return moves


# <------------------------------------------------- June 15th, 2024 ------------------------------------------------->
# 502. IPO

# LeetCode is planning its IPO and wants to maximize its capital by completing at most `k` distinct projects,
# each with a certain profit `profits[i]` and a minimum starting capital `capital[i]`.
# Initially, LeetCode has `w` capital, and the task is to select a list of `n` projects that maximizes the
# final capital, considering that the profit from a completed project increases the total capital.


def findMaximizedCapital1(k: int, w: int, profits: List[int], capital: List[int]) -> int:
    """
    Finds the maximum capital achievable by completing at most 'k' projects given initial capital 'w',
    project profits, and capital requirements.

    This function uses a greedy approach with two heaps to optimize project selection.
    It first checks whether the initial capital 'w' is more than the maximum capital required across all projects.
    If it is, it simply returns the initial capital plus the total profits from the 'k' most profitable projects.
    If not, the function manages the projects using two heap data structures - one min-heap for 'projects'
    sorted by their capital requirements and the other (max-heap) for 'possible_projects'
    storing the profits negatively for getting the maximum profit as the heap head (since heapq is a min-heap).
    The function loops 'k' times (or up to the number of possible projects), each time adding affordable projects to
    the 'possible_projects' heap, and removing ("finishing") the most profitable project from 'possible_projects'
    heap while adding its profit to our current capital 'w'.
    This way, we ensure that at each step, we are choosing the project which is affordable and gives the maximum profit.

    The time complexity of this solution is O(n log n), where `n` is the number of projects.
    This is because the solution involves transforming the list into a heap
    (O(n log n)) and performing up to k heap operations (O(k log n)), giving a total of O(n log n) since k <= n.
    The space complexity is O(n) due to the storage requirements for the heaps.
    """
    print("\n--- Input Parameters ---")
    print(f"\tk = {k}")
    print(f"\tw = {w}")
    print(f"\tprofits = {profits}")
    print(f"\tcapital = {capital}")

    print("\n--- Initial Check ---")
    if w >= max(capital):
        print("\tAll projects are affordable initially.")
        result = w + sum(heapq.nlargest(k, profits))
        print(f"\tResult: {result}")
        return result

    print("\n--- Project Heap Initialization ---")
    projects = list(zip(capital, profits))
    heapq.heapify(projects)
    print(f"\tprojects (min-heap by capital): {projects}")

    possible_projects = []  # Max-heap of affordable projects (by profit)

    print("\n--- Main Loop (Project Selection) ---")
    for i in range(k):
        print(f"\nIteration {i + 1}:")
        print(f"\tCurrent capital (w): {w}")

        print("\t\tAdding Affordable Projects:")
        print(f"\t\t\tProjects available: {projects}")
        while projects and projects[0][0] <= w:
            cap, prof = heapq.heappop(projects)
            heapq.heappush(possible_projects, -prof)  # Negate for max-heap behavior
            print(f"\t\t\tProject (capital={cap}, profit={prof}) added to possible_projects")

        if not possible_projects:
            print("\t\tNo more affordable projects. Exiting loop.")
            break

        print(f"\t\tPossible Projects (max-heap by profit): {possible_projects}")
        max_profit = -heapq.heappop(possible_projects)  # Get and negate max profit
        w += max_profit
        print(f"\t\tChose project with profit {max_profit}. Updated capital: {w}")

    print("\n--- Function Returning ---")
    print(f"\tFinal capital: {w}")
    return w


# <------------------------------------------------- June 16th, 2024 ------------------------------------------------->
# 330. Patching Array

# Given a sorted integer array `nums` and an integer `n`, add/patch elements to the array such that any number in
# the range `[1, n]` inclusive can be formed by the sum of some elements in the array.
# Return the minimum number of patches required.


def minPatches1(nums: List[int], n: int) -> int:
    """
    Determines the minimum number of patches required to make a sorted list of numbers cover all integers from 1 to n.

    The function uses a greedy algorithm approach.
    It starts with `possible_patch` as 1 (the smallest possible number we need to form),
    and `patches` as 0 (the count of patches needed).
    It then traverses `nums`, checking each number to see if it is within the current range [1, possible_patch).
    If a `nums` element can be added to the range without leaving a gap, we update `possible_patch` to include it.
    If not, a patch is necessary, and we double `possible_patch` to cover the gap.
    Doubling `possible_patch` ensures that if we can cover all numbers in [1, possible_patch) with the current list,
    we can cover all numbers in [1, 2 * possible_patch) by adding `possible_patch` to the list.
    The optimal patch is always `possible_patch`, as it is the smallest integer outside the current range.
    The function continues this process until `possible_patch` exceeds `n`.

    The time complexity of this solution is O(m + log(n)) because in the worst-case scenario,
    it iterates over each element in `nums` (which takes O(m), where `m` is the size of `nums`),
    and it doubles `possible_patch` until it is larger than `n` (which takes O(log(n))).
    The space complexity is O(1) because it doesn't create any new data structures that grow with the size of the input.
    """
    # --- Input Parameters ---
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tn = {n}")

    # --- Initialization ---
    patches = 0
    possible_patch = 1
    index = 0
    iteration_data = []  # Collect data for iteration summary

    # --- Main Loop (Greedy Patching) ---
    print("\n--- Main Loop (Greedy Patching) ---")
    iterations = 0
    while possible_patch <= n and index < len(nums):
        iterations += 1
        print(f"\nIteration {iterations}:")
        if iterations > 1:
            current_coverage = f"[1, {possible_patch})"
            print(f"\tCurrent coverage: [1, {possible_patch})")
        else:
            current_coverage = "No coverage yet"
            print(f"\tInitial coverage: No coverage yet")
        print(f"\tCurrent possible_patch: {possible_patch}")

        prev_index = index
        prev_possible_patch = possible_patch

        print(f"\tChecking if nums[{index}] = {nums[index]} is within the range [1, {possible_patch})")
        if nums[index] <= possible_patch:
            print(f"\t\tIn range. Updating possible_patch to {possible_patch} + {nums[index]} = "
                  f"{possible_patch + nums[index]}")
            action = f"Add `nums[{index}]` = {nums[index]} to `possible_patch` and increment index"
            possible_patch += nums[index]
            index += 1
        else:
            print(f"\t\tOut of range. Doubling possible_patch to {possible_patch * 2}")
            print(f"\t\tAdding patch: {patches} + 1 = {patches + 1}")
            action = f"Patch needed. Double `possible_patch` to {possible_patch * 2}"
            possible_patch *= 2
            patches += 1

        iteration_data.append([iterations, prev_possible_patch, current_coverage, prev_index, nums[prev_index], action,
                               patches])

    while possible_patch <= n:
        iterations += 1
        print(f"\nIteration {iterations}:")
        print(f"\tCurrent coverage: [1, {possible_patch})")
        print(f"\tCurrent possible_patch: {possible_patch}")
        print(f"\tPossible_patch {possible_patch} is not covering {n} yet. Doubling possible_patch to "
              f"{possible_patch * 2})")
        print(f"\tAdding patch: {patches} + 1 = {patches + 1}")

        prev_possible_patch = possible_patch
        action = f"Patch needed. Double `possible_patch` to {possible_patch * 2}"
        possible_patch *= 2
        patches += 1

        iteration_data.append([iterations, prev_possible_patch, f"[1, {prev_possible_patch})", index, "N/A", action,
                               patches])

    print(f"\nFinally, n = {n} is covered by the range [1, {possible_patch})")

    # --- Iteration Summary ---
    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Possible Patch", "Current Coverage", "Index", "Num", "Action", "Patches"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    # --- Function Returning ---
    print("\n--- Function Returning ---")
    print(f"\tMinimum patches required: {patches}")
    return patches


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for June 10th, 2024
# Expected output: 3
# heightChecker1(heights=[1, 2, 6, 4, 2, 5])

# Test cases for June 11th, 2024
arr1 = [2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19]
arr2 = [2, 1, 4, 3, 9, 6]
# relativeSortArray1(arr1, arr2)  # Expected output: [2,2,2,1,4,3,3,9,6,7,19]
# relativeSortArray2(arr1, arr2)  # Expected output: [2,2,2,1,4,3,3,9,6,7,19]

# Test cases for June 12th, 2024
nums = [2, 0, 2, 1, 1, 0]
# sortColors1(nums)  # Expected output: [0,0,1,1,2,2]
# sortColors2(nums)  # Expected output: [0,0,1,1,2,2]

# Test cases for June 13th, 2024
seats = [4, 1, 5, 9]
students = [1, 3, 2, 6]
# minMovesToSeat1(seats, students)  # Expected output: 7
# minMovesToSeat2(seats, students)  # Expected output: 7

# Test cases for June 14th, 2024
nums_2 = [3, 2, 1, 2, 1, 3]
# minIncrementForUnique1(nums_2)  # Expected output: 6
# minIncrementForUnique2(nums_2)  # Expected output: 6

# Test cases for June 15th, 2024
k = 3
w = 0
profits = [1, 2, 3, 4, 5]
capital = [0, 1, 1, 3, 3]
# findMaximizedCapital1(k, w, profits, capital)  # Expected output: 9

# Test cases for June 16th, 2024
# minPatches1([1, 5, 10], 50)  # Expected output: 4
