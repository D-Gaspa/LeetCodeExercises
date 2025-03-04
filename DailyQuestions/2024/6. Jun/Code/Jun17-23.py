import bisect
import heapq
import math
from bisect import bisect_right
from collections import deque
from typing import List

from tabulate import tabulate


# June 2024, Week 4: June 17th - June 23rd

# <------------------------------------------------- June 17th, 2024 ------------------------------------------------->
# 633. Sum of Square Numbers

# Given a non-negative integer `c`, decide whether there are two integers `a` and `b` such that `a^2 + b^2 = c`.


def judgeSquareSum1(c: int) -> bool:
    """
    Determines if a given non-negative integer 'c' can be expressed as the sum of squares of two integers 'a' and 'b'.

    The function uses a two-pointer technique starting from 0 and the square root of 'c'.
    It iteratively checks the sum of squares of the two pointers, 'start_index' and 'end_index'.
    If the sum is less than 'c', it increases 'start_index' to get a larger sum.
    If the sum is greater than 'c', it decreases 'end_index' to get a smaller sum.
    If the sum is equal to 'c', the function returns True as it has found the pair of numbers.
    If no such pair is found after the loop, it returns False.
    This approach works because if there exist two numbers 'a' and 'b' such that a^2 + b^2 = c,
    then 'a' and 'b' must each be less than or equal to sqrt(c).

    The time complexity of this function is O(√c) because, in the worst case,
    the while loop iterates up to the square root of 'c' times.
    The space complexity is O(1) as it uses a constant amount of extra space.
    """
    # Input Parameters
    print("\n--- Input Parameters ---")
    print(f"\tc = {c}")

    # Initialization
    print("\n--- Initialization ---")
    start_index = 0
    end_index = int(math.sqrt(c))
    print(f"\tstart_index = {start_index}")
    print(f"\tend_index = sqrt({c}) = {end_index}")

    # Main Loop (Two-Pointer Search)
    print("\n--- Main Loop (Two-Pointer Search) ---")
    iteration_data = []  # To collect data for the iteration summary
    iteration = 0
    while start_index <= end_index:
        iteration += 1
        print(f"\nWhile {start_index} <= {end_index}: (Iteration {iteration})")
        squares_sum = start_index * start_index + end_index * end_index
        print(f"\tsquares_sum = {squares_sum} ({start_index}^2 + {end_index}^2)")

        print("\tDecision Point:")
        if squares_sum < c:
            print(f"\t\tsquares_sum ({squares_sum}) < c ({c}), increasing start_index to {start_index + 1}")
            iteration_data.append([iteration, start_index, end_index, squares_sum, f"{squares_sum} < {c}",
                                   "`start_index += 1`"])
            start_index += 1
        elif squares_sum > c:
            print(f"\t\tsquares_sum ({squares_sum}) > c ({c}), decreasing end_index to {end_index - 1}")
            iteration_data.append([iteration, start_index, end_index, squares_sum, f"{squares_sum} > {c}",
                                   "`end_index -= 1`"])
            end_index -= 1
        else:
            print(f"\t\tsquares_sum ({squares_sum}) == c ({c}), returning True")
            # Iteration Summary Data
            iteration_data.append([iteration, start_index, end_index, squares_sum, f"{squares_sum} == {c}",
                                   "Return True"])
            result = True
            break

    else:  # No solution found in the loop
        print("\tNo solution found within the loop.")
        result = False

    # Iteration Summary Table
    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "`start_index`", "`end_index`", "`squares_sum`", "Comparison", "Action"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    # Function Return
    print("\n--- Function Returning ---")
    print(f"\tResult: {result}")
    return result


def judgeSquareSum2(c: int) -> bool:
    """
    Determines if a given non-negative integer 'c' can be expressed as the sum of squares of two integers 'a' and 'b'.

    The function uses properties from number theory, particularly Fermat's theorem on sums of two squares.
    According to the theorem, a non-negative integer can be expressed as a sum of two squares if and only if every
    prime factor of the form (4k + 3) has an even exponent in the factorization of 'c'.

    The function iterates through possible prime factors up to the square root of 'c'.
    For each factor, it counts the number of times it divides 'c'.
    If a prime factor of the form (4k + 3) divides 'c' an odd number of times, the function returns False.
    Additionally, after factoring out all smaller primes, if the remaining part of 'c' is a prime of the form (4k + 3),
    the function also returns False.
    If no such prime factors are found, the function returns True.

    The time complexity of this solution is O(√c log c) because it iterates up to the square root of 'c' and
    performs division operations for each prime factor (log c).
    The space complexity is O(1) as it uses a constant amount of extra space.
    """
    # Input Parameters
    print("\n--- Input Parameters ---")
    print(f"\tc = {c}")

    # Initialization
    print("\n--- Initialization ---")
    index = 2
    original_c = c  # Store the original value of c for later reference
    print(f"\tindex = {index}")

    # Main Loop (Checking Prime Factors)
    print("\n--- Main Loop (Checking Prime Factors) ---")
    iteration_data = []  # To collect data for the iteration summary
    iteration = 0
    while index * index <= c:
        needs_iteration_data = True
        iteration += 1
        print(f"\nWhile {index}^2 <= {c}: (Iteration {iteration})")
        divisors_count = 0

        # Decision Point 1 (Divisibility)
        print(f"\tDecision Point 1 (Divisibility):")
        if c % index == 0:
            print(f"\t\tc is divisible by {index}")

            # Inner Loop (Counting Divisors)
            print("\t\tInner Loop (Counting Divisors):")
            while c % index == 0:
                divisors_count += 1
                c //= index
                print(f"\t\t\tdivisors_count = {divisors_count}, c = {c}")

            print(f"\t\t\t{index}^({divisors_count}) is a factor of c")

            # Decision Point 2 (Odd Exponent and Form)
            is_odd_exponent = divisors_count % 2 == 1
            is_form_4k_plus_3 = index % 4 == 3

            print(f"\t\tDecision Point 2 (Odd Exponent and Form):")
            if not is_form_4k_plus_3:
                print(f"\t\t\t{index} is not of the form 4k + 3, continuing")
                action = "Continue"
            else:
                print(f"\t\t\t{index} is of the form 4k + 3")
                if is_odd_exponent:
                    print(f"\t\t\t{index} has an odd exponent, returning False")
                    action = f"Return False (Odd exponent and form 4k + 3)"
                else:
                    print(f"\t\t\t{index} has an even exponent, continuing")
                    action = "Continue (Even exponent)"

            # Iteration Summary Data
            iteration_data.append([iteration, index, divisors_count, c, "True" if index % 4 == 3 else "False", action])
            needs_iteration_data = False

            if is_odd_exponent and is_form_4k_plus_3:
                result = False
                break
        else:
            print(f"\t\tc is not divisible by {index}")

        if needs_iteration_data:
            iteration_data.append([iteration, index, divisors_count, c, "False", "Continue"])

        index += 1

    else:  # Loop completed without finding a disqualifying factor
        # Decision Point 3 (Remaining Value)
        print("\n--- Decision Point 3 (Remaining Value) ---")
        print(f"\tRemaining value of c after factorization: {c}")
        if c % 4 != 3:
            print(f"\t\tRemaining value is not of the form 4k + 3, returning True")
            result = True
        else:
            print(f"\t\tRemaining value is of the form 4k + 3, returning False")
            result = False

    # Iteration Summary Table
    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "'index'", "'divisors_count'", "'c'", "`index % 4 == 3`", "`Result`"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    # Function Return
    print("\n--- Function Returning ---")
    print(f"\tInput c = {original_c}, Result: {result}")  # Print original input for clarity
    return result


# <------------------------------------------------- June 18th, 2024 ------------------------------------------------->
# 826. Most Profit Assigning Work

# Given n jobs each with a difficulty and profit, and m workers each with an ability,
# assign each worker at most one job that they can complete
# (i.e., a job with difficulty at most equal to the worker's ability) to maximize the profit.
# A job can be completed multiple times, and if a worker cannot complete any job, their profit is $0.
# Return the maximum profit achievable.


def maxProfitAssignment1(difficulty: List[int], profit: List[int], worker: List[int]) -> int:
    """
    Calculates the maximum total profit that workers can achieve based on their abilities and the given jobs'
    difficulties and profits.

    The function first determines the maximum ability of all workers, then initializes a list to store the maximum
    profit for each level of difficulty up to this maximum ability.
    By iterating through each job's difficulty and profit, it updates this list to ensure it captures the highest
    profit available for each difficulty level up to the maximum ability.
    The function then adjusts this list so that for any given difficulty, it reflects the highest-profit achievable
    up to that difficulty level, because a job with a lower difficulty may have a higher profit.
    Finally, it sums up the highest possible profit for each worker based on their respective abilities.

    The time complexity of this solution is O(n + m + max_ability), where `n` is the number of jobs,
    `m` is the number of workers, and `max_ability` is the maximum ability of any worker.
    This is because it iterates through the difficulties and profits once (O(n)),
    the range of abilities (O(max_ability)), and the workers once (O(m)).
    The space complexity is O(max_ability) due to the list storing maximum profits per difficulty level.
    """
    print("\n--- Input Parameters ---")
    print(f"\tdifficulty = {difficulty}")
    print(f"\tprofit = {profit}")
    print(f"\tworker = {worker}")

    print("\n--- Preprocessing: Finding Maximum Ability ---")
    max_ability = max(worker)
    print(f"\tmax_ability = {max_ability}")

    print("\n--- Main Loop (Building Max Profit Array) ---")
    max_profit_per_diff = [0] * (max_ability + 1)
    iteration_data = []  # To collect data for the iteration summary
    for i, (diff, prof) in enumerate(zip(difficulty, profit)):
        print(f"\n\tIteration {i + 1}:")
        print(f"\t\tCurrent job: (difficulty={diff}, profit={prof})")

        if diff <= max_ability:
            old_profit = max_profit_per_diff[diff]
            max_profit_per_diff[diff] = max(max_profit_per_diff[diff], prof)
            print(f"\t\tUpdated max_profit_per_diff[{diff}]: {old_profit} -> {max_profit_per_diff[diff]}")
        else:
            print(f"\t\tSkipped job (difficulty exceeds max_ability)")

        iteration_data.append(
            [i + 1, diff, prof, max_profit_per_diff[:]])  # Store a copy of the list for each iteration

    print("\n--- Iteration Summary (Max Profit Array Building) ---")
    headers = ["Iteration", "Difficulty", "Profit", "max_profit_per_diff"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Second Loop (Propagating Maximum Profits) ---")
    copy_max_profit_per_diff = max_profit_per_diff.copy()
    for index in range(1, max_ability + 1):
        print(f"\n\tIteration {index}/{max_ability}:")
        old_profit = max_profit_per_diff[index]
        max_profit_per_diff[index] = max(max_profit_per_diff[index], max_profit_per_diff[index - 1])
        print(f"\t\tUpdated max_profit_per_diff[{index}]: {old_profit} -> {max_profit_per_diff[index]}")

    print(f"\nPrevious Max Profits: {copy_max_profit_per_diff}")
    print(f"Updated Max Profits: {max_profit_per_diff}")

    print("\n--- Calculating Total Profit ---")
    total_profit = 0
    for i, ability in enumerate(worker):
        print(f"\n\tWorker {i + 1}:")
        print(f"\t\tAbility: {ability}")
        print(f"\t\tMaximum Profit: {max_profit_per_diff[ability]}")
        print(f"\t\tCurrent Total Profit: ({total_profit} + {max_profit_per_diff[ability]}) = "
              f"{total_profit + max_profit_per_diff[ability]}")
        total_profit += max_profit_per_diff[ability]

    print(f"\tTotal Profit: {total_profit}")

    print("\n--- Function Returning ---")
    return total_profit


def maxProfitAssignment2(difficulty: List[int], profit: List[int], worker: List[int]) -> int:
    """
    Calculates the maximum total profit that workers can achieve based on their abilities and the given jobs'
    difficulties and profits.

    This function first sorts the jobs by difficulty while pairing each job with its corresponding profit.
    It then processes these sorted jobs to create a list where each job entry holds the highest profit available
    up to that difficulty level.
    This transformation ensures that for any worker's ability, we can quickly find the best possible profit they can
    achieve using binary search.
    Since we are dealing with a tuple of (difficulty, profit), the binary search is performed on the difficulty values,
    and the other value (profit) is used with 'float('inf')' as the upper bound for the binary search.
    This way, we ensure that `bisect_right` will find the index where a worker's ability can be inserted in the sorted
    job list with the highest profit available up to that difficulty.
    We subtract 1 to get the index of the highest-paying job the worker can perform.
    By summing up the maximum achievable profits for all workers, the function computes the total maximum profit.

    The time complexity of this solution is O((n + m) log n), where `n` is the number of jobs, and `m` is the number of
    workers.
    This is because sorting the jobs takes O(n log n) and each worker's job search takes O(log n) due to
    binary search, repeated `m` times; hence, the total time complexity is O(n log n + m log n) = O((n + m) log n).
    The space complexity is O(n) for storing the processed job list.
    """
    # --- Input Parameters ---
    print("\n--- Input Parameters ---")
    print(f"\tdifficulty = {difficulty}")
    print(f"\tprofit = {profit}")
    print(f"\tworker = {worker}")

    # --- Sorting Jobs by Difficulty ---
    print("\n--- Sorting Jobs by Difficulty ---")
    jobs = sorted(zip(difficulty, profit))
    print(f"\tjobs (sorted) = {jobs}")

    # --- Main Loop (Transforming Jobs with Max Profit) ---
    print("\n--- Main Loop (Transforming Jobs with Max Profit) ---")
    max_profit_so_far = 0
    iteration_data_1 = []  # Collect data for iteration summary table
    for index, job in enumerate(jobs):
        print(f"\n\tIteration {index + 1}/{len(jobs)}:")
        print(f"\t\tCurrent job: {job}")
        max_profit_so_far = max(max_profit_so_far, job[1])
        jobs[index] = (job[0], max_profit_so_far)
        print(f"\t\tUpdated job: {jobs[index]}")
        iteration_data_1.append([index + 1, job, max_profit_so_far, jobs[index]])

    # --- Iteration Summary (Transformed Jobs) ---
    print("\n--- Iteration Summary (Transformed Jobs) ---")
    headers = ["Iteration", "Original Job", "Max Profit So Far", "Updated Job"]
    print(tabulate(iteration_data_1, headers=headers, tablefmt="fancy_grid"))

    print(f"\nJobs with maximum profit: {jobs}")

    # --- Calculating Total Profit (Worker Loop) ---
    print("\n--- Calculating Total Profit (Worker Loop) ---")
    total_profit = 0
    for i, ability in enumerate(worker):
        print(f"\n\tWorker {i + 1}/{len(worker)}:")
        print(f"\t\tWorker ability: {ability}")
        index = bisect_right(jobs, (ability, float('inf')))
        print(f"\t\tJob index found: {index}")
        if index > 0:
            profit = jobs[index - 1][1]
            print(f"\t\tProfit from job: {profit}")
            print(f"\t\tTotal profit so far: {total_profit} + {profit} = {total_profit + profit}")
            total_profit += profit
        else:
            print(f"\t\tNo suitable job found.")

    # --- Final Result ---
    print("\n--- Final Result ---")
    print(f"\tTotal Profit: {total_profit}")
    return total_profit


def maxProfitAssignment3(difficulty: List[int], profit: List[int], worker: List[int]) -> int:
    """
    Calculates the maximum total profit that workers can achieve based on their abilities and the given jobs'
    difficulties and profits.

    This function sorts the jobs by difficulty and pairs them with their respective profits.
    It also sorts the workers based on their abilities.
    As it iterates through the sorted workers, it keeps track of the maximum profit available for any job that a
    worker can perform up to their ability.
    By accumulating this maximum profit for each worker, it computes the total maximum profit that can be achieved.

    The time complexity of this solution is O(n log n + m log m), where `n` is the number of jobs and `m` is the
    number of workers as the jobs and workers are sorted (taking O(n log n) and O(m log m) time, respectively).
    The space complexity is O(n) for storing the sorted list of jobs.
    """
    # --- Input Parameters ---
    print("\n--- Input Parameters ---")
    print(f"\tdifficulty = {difficulty}")
    print(f"\tprofit = {profit}")
    print(f"\tworker = {worker}")

    # --- Sort Jobs and Workers ---
    print("\n--- Sorting Jobs and Workers ---")
    jobs = sorted(zip(difficulty, profit))
    worker.sort()  # Sort workers in-place for efficiency
    print(f"\tjobs (sorted) = {jobs}")
    print(f"\tworker (sorted) = {worker}")

    # --- Main Loop (Calculate Total Profit) ---
    print("\n--- Main Loop (Calculate Total Profit) ---")
    total_profit = 0
    max_profit_so_far, job_index = 0, 0
    iteration_data = []  # Collect data for iteration summary table
    for i, ability in enumerate(worker):
        print(f"\n\tWorker {i + 1}/{len(worker)}:")
        print(f"\t\tCurrent worker ability: {ability}")

        print(f"\t\tCurrent max_profit_so_far: {max_profit_so_far}")

        # --- Inner Loop (Update Max Profit) ---
        while job_index < len(jobs) and jobs[job_index][0] <= ability:
            print(f"\t\t\tConsidering job: {jobs[job_index]}")
            if max_profit_so_far < jobs[job_index][1]:
                print(f"\t\t\tUpdated max_profit_so_far to {jobs[job_index][1]}")
            else:
                print(f"\t\t\tKeeping max_profit_so_far at {max_profit_so_far}")
            max_profit_so_far = max(max_profit_so_far, jobs[job_index][1])
            job_index += 1

        print(f"\t\tTotal profit updated to: {total_profit} + {max_profit_so_far} = {total_profit + max_profit_so_far}")
        print(f"\t\tNext job: {jobs[job_index]}")
        total_profit += max_profit_so_far
        iteration_data.append([i + 1, ability, max_profit_so_far, total_profit])

    # --- Iteration Summary ---
    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Worker Ability", "Max Profit So Far", "Total Profit"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    # --- Final Result ---
    print("\n--- Final Result ---")
    print(f"\tTotal Profit: {total_profit}")
    return total_profit


# <------------------------------------------------- June 19th, 2024 ------------------------------------------------->
# 1482. Minimum Number of Days to Make m Bouquets

# Given an integer array bloom_day, and integers `m` and `k`, determine the minimum number of days to wait to make `m`
# bouquets from the garden, each requiring `k` adjacent flowers.
# If it's impossible to make m bouquets, return -1.


def minDays1_1(bloom_day: List[int], m: int, k: int) -> int:
    """
    Determines the minimum number of days required to make `m` bouquets using `k` adjacent flowers from the garden,
    given the bloom days of the flowers `bloom_day`.

    The function performs a binary search over the range of bloom days to find the lowest day value at which it's
    possible to make the required number of bouquets.
    It first checks if it's possible to make the bouquets given the constraint, and then performs binary search on the
    range of bloom days (min(bloom_day) to max(bloom_day)) to find the earliest day that satisfies the condition.
    The `can_make_bouquets` helper function checks if it's possible to make the bouquets by a given day by iterating
    over the bloom_day list and counting the number of adjacent flowers that have bloomed by that day.
    The binary search reduces the search space logarithmically, and the helper function ensures that the feasibility
    of making bouquets is checked efficiently.

    The time complexity of this solution is O(n log D), where `n` is the length of the `bloom_day` list and `D` is the
    range of bloom days (max(bloom_day) - min(bloom_day)).
    This is because the binary search runs in O(log D) time, and each check within the search takes O(n) time.
    The space complexity is O(1) since only a few extra variables are used.
    """
    print(f"\tbloom_day = {bloom_day}")
    print(f"\tm = {m}")
    print(f"\tk = {k}")

    print(f"\nChecking if {m} bouquets of {k} flowers can be made with the given flowers:")
    if k * m > len(bloom_day):
        print("\n--- Not Enough Flowers ---")
        print(f"\tImpossible to make {m} bouquets of {k} flowers.")
        return -1
    print(f"There are enough flowers ({len(bloom_day)}), we need {k * m}.")

    def can_make_bouquets(day: int) -> bool:
        print(f"\n--- Checking if {m} bouquets of {k} flowers can be made by day {day} ---")
        bouquet_count = 0
        flowers = 0
        iteration_data = []  # to store iteration data to be printed
        enough_bouquets = False

        for i, bloom in enumerate(bloom_day):
            if enough_bouquets:
                break
            print(f"\tFlower {i + 1} with bloom day {bloom}:")
            if bloom <= day and enough_bouquets is False:
                print(f"\t\tAdding flower to bouquet. Total Flowers: {flowers + 1}")
                flowers += 1
                copy_bouquet_count = bouquet_count
                copy_bouquet_count += 1 if flowers == k else 0
                action = "Increment Flowers"
                if flowers == k:
                    action = "Make Bouquet and Reset Flowers"
                iteration_data.append([i + 1, bloom, "Yes", action, flowers, copy_bouquet_count])
                if flowers == k:
                    bouquet_count += 1
                    print(f"\t\tBouquet made. Total Bouquets: {bouquet_count}")
                    if bouquet_count >= m:
                        print("\t\tEnough bouquets made. Returning True.")
                        enough_bouquets = True
                        break
                    flowers = 0
            else:
                print("\t\tFlower cannot be used. Resetting flower count.")
                flowers = 0
                iteration_data.append([i + 1, bloom, "No", "Reset Flowers", flowers, bouquet_count])

        print("\n--- Iteration Summary (can_make_bouquets) ---")
        headers = ["Flower", "Bloom Day", f"Bloom Day <= {day}", "Action", "Flowers", f"Bouquets ({k} size)"]
        print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

        if enough_bouquets:
            return True
        return bouquet_count >= m

    left_index, right_index = min(bloom_day), max(bloom_day)
    print(f"\n--- Binary Search: Initial Range: ({left_index}, {right_index}) ---")

    binary_search_iterations = []  # store data for binary search table
    iterations = 0

    while left_index < right_index:
        iterations += 1
        mid_index = (left_index + right_index) // 2
        print(f"\nWhile {left_index} < {right_index}: with mid_index = {mid_index}")

        if can_make_bouquets(mid_index):
            print(f"\tEnough bouquets can be made. Adjusting right index to mid_index: {mid_index}")
            binary_search_iterations.append([iterations, left_index, right_index, mid_index, "Yes",
                                             f"right = mid ({mid_index})"])
            right_index = mid_index
        else:
            print(f"\tNot enough bouquets can be made. Adjusting left index to mid_index + 1: {mid_index + 1}")
            binary_search_iterations.append([iterations, left_index, right_index, mid_index, "No",
                                             f"left = mid + 1 ({mid_index + 1})"])
            left_index = mid_index + 1

    print(f"\nBinary Search Ended with left_index = {left_index} and right_index = {right_index}")

    print("\n--- Binary Search Summary ---")
    headers = ["Iteration", "Left Index", "Right Index", "Mid Index", "Can Make Bouquets?", "Action"]
    print(tabulate(binary_search_iterations, headers=headers, tablefmt="fancy_grid"))
    print(f"\n--- Binary Search Complete: Final left_index (Result): {left_index} ---")
    return left_index


def minDays1_2(bloom_day: List[int], m: int, k: int) -> int:
    """
    Determines the minimum number of days required to make `m` bouquets using `k` adjacent flowers from the garden,
    given the bloom days of the flowers `bloom_day`.

    The function leverages a binary search on a sorted list of unique bloom days to find the minimum day at which the
    required number of bouquets could be made.
    While it has a similar structure to the `minDays1_1` function, it optimizes the search space by considering only
    the unique bloom days, sorted in advance to potentially reduce the number of comparisons needed.
    The `can_make_bouquets` function is also slightly optimized in this version, and instead of resetting the flower
    count immediately after finding k flowers, it continues to count potential bouquets from the remaining flowers.
    As the previous version, the binary search reduces the search space logarithmically, and the helper function ensures
    that the feasibility of making bouquets is checked efficiently.

    The time complexity of this solution is O(n log n), where `n is the length of the bloom_day list.
    This is because transforming the list to a set takes O(n) time, and sorting the unique bloom days takes O(U log U)
    time, where `U` is the number of unique bloom days.
    The binary search runs in O(log U) time, and each check within the search takes O(n) time.
    Since U <= n, the overall time complexity simplifies from O(n + U log U + n log U) to O(n log n).
    The space complexity is O(U), which is at most O(n) in the worst case when all bloom days are unique.
    """
    print(f"\tbloom_day = {bloom_day}")
    print(f"\tm = {m}")
    print(f"\tk = {k}")

    if k * m > len(bloom_day):
        print(f"\nChecking if {m} bouquets of {k} flowers can be made with the given flowers:")
        print("\n--- Not Enough Flowers ---")
        print(f"\tImpossible to make {m} bouquets of {k} flowers.")
        return -1

    print(f"\nChecking if {m} bouquets of {k} flowers can be made with the given flowers:")
    print(f"There are enough flowers ({len(bloom_day)}), we need {k * m}.")

    def can_make_bouquets(day: int) -> bool:
        """Helper function to check if `m` bouquets can be made by a given day."""
        print(f"\n--- Checking if {m} bouquets of {k} flowers can be made by day {day} ---")
        bouquet_count = 0
        flowers = 0
        iteration_data = []  # to store iteration data to be printed
        enough_bouquets = False
        for i, bloom in enumerate(bloom_day):
            if enough_bouquets:
                break
            print(f"\tFlower {i + 1}/{len(bloom_day)} with bloom day {bloom}:")
            if bloom <= day:
                print(f"\t\tAdding flower to bouquet. Total Flowers: {flowers + 1}")
                flowers += 1
                copy_bouquet_count = bouquet_count
                action = "Increment Flowers"
                if i == len(bloom_day) - 1:
                    copy_bouquet_count += flowers // k
                    action = "Final Bouquet Calculation"

                iteration_data.append([i + 1, bloom, "Yes", action, flowers, copy_bouquet_count])
            else:
                bouquet_count += flowers // k  # Calculate bouquets from accumulated flowers before resetting
                print(f"\t\tFlower cannot be used. Calculate bouquets from accumulated flowers: ({flowers} // {k}) = "
                      f"{flowers // k}")
                print(f"\t\tTotal Bouquets: {bouquet_count}")
                flowers = 0
                iteration_data.append([i + 1, bloom, "No", "Reset Flowers/Calculate Bouquets", flowers,
                                       bouquet_count])
                if bouquet_count >= m:
                    return True
        bouquet_count += flowers // k
        print(f"\tCalculate bouquets from remaining flowers: ({flowers} // {k}) = {flowers // k}")
        print(f"\tTotal Bouquets: {bouquet_count}")

        print("\n--- Iteration Summary (can_make_bouquets) ---")
        headers = ["Flower", "Bloom Day", f"Bloom Day <= {day}", "Action", "Flowers", f"Bouquets ({k} size)"]
        print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

        return bouquet_count >= m

    unique_bloom_days = sorted(set(bloom_day))
    print("\n--- Unique Bloom Days ---")
    print(f"\t{unique_bloom_days}")

    left_index, right_index = 0, len(unique_bloom_days) - 1
    print(f"\n--- Binary Search: Initial Range: ({left_index}, {right_index}) ---")

    binary_search_iterations = []
    iterations = 0
    while left_index < right_index:
        iterations += 1
        mid_index = (left_index + right_index) // 2
        print(f"\nWhile {left_index} < {right_index}: with mid_index = {mid_index}")

        if can_make_bouquets(unique_bloom_days[mid_index]):
            print(f"\tEnough bouquets can be made in {unique_bloom_days[mid_index]} days. "
                  f"Adjusting right index to mid_index: {mid_index}")
            binary_search_iterations.append([iterations, left_index, right_index, mid_index, "Yes",
                                             f"right = mid ({mid_index})"])
            right_index = mid_index
        else:
            print(f"\tNot enough bouquets can be made in {unique_bloom_days[mid_index]} days. "
                  f"Adjusting left index to mid_index + 1: {mid_index + 1}")
            binary_search_iterations.append([iterations, left_index, right_index, mid_index, "No",
                                             f"left = mid + 1 ({mid_index + 1})"])
            left_index = mid_index + 1

    print(f"\nBinary Search Ended with left_index = {left_index} and right_index = {right_index}")

    print("\n--- Binary Search Summary ---")
    headers = ["Iteration", "Left Index", "Right Index", "Mid Index", "Can Make Bouquets?", "Action"]
    print(tabulate(binary_search_iterations, headers=headers, tablefmt="fancy_grid"))
    print(f"\n--- Binary Search Complete: Final left_index (Result): {left_index} ---")
    print(f"\n--- Function Returning ---")
    print(f"\tResult: unique_bloom_days[{left_index}] = {unique_bloom_days[left_index]}")
    return unique_bloom_days[left_index]


# <------------------------------------------------- June 20th, 2024 ------------------------------------------------->
# 1552. Magnetic Force Between Two Balls

# In Earth C-137, Rick has `n` baskets at positions given by `position[i]` and Morty has `m` balls.
# The task is to distribute the balls into the baskets such that the minimum magnetic force
# (defined as `|x - y|` between two balls at positions `x` and `y`) is maximized.
# The function should return this maximum minimum magnetic force.


def maxDistance1(position: List[int], m: int) -> int:
    """
    Determines the maximum possible minimum magnetic force between any two balls placed in baskets.

    This function uses a binary search approach to find the optimal minimum distance between balls.
    It first sorts the position list, then employs a helper function can_place_balls` to check if it's possible
    to place all `m` balls with a given minimum distance.
    The binary search efficiently narrows down the range of possible distances.

    The search space is defined by the following bounds:
    - Lower Bound (`1`): The minimum possible force is 1, when balls are placed in adjacent positions.
    - Upper Bound (`(max(position) - min(position)) // (m - 1)`): To maximize the minimum distance, we aim to spread
      the balls as far apart as possible.
      This upper bound represents the theoretical maximum average distance achievable with `m` balls and `m-1` gaps.

    The time complexity of this solution is O(n log(n * range)), where `n` is the length of the position list and range
    is `(max(position) - min(position)) / (m - 1)`.
    This is because we sort the position list in O(n log n) time and perform binary search with the `can_place_balls`
    check in O(n log(range)) time, and when combined, it results in O(n log(n) + n log(range)) = O(n log(n * range)).
    The space complexity is O(n) due to the sorting operation.
    """

    def can_place_balls(min_distance: int) -> bool:
        print(f"\n--- can_place_balls Check (min_distance = {min_distance}) ---")
        remaining_balls = m - 1
        next_valid_position = position[0] + min_distance
        print(f"\tInitial next_valid_position: {position[0]} + {min_distance} = {next_valid_position}")

        for i, pos in enumerate(position[1:], start=1):
            print(f"\tChecking position[{i}] = {pos}")
            if pos >= next_valid_position:
                print(f"\t\t{pos} >= {next_valid_position}. Ball placed. Remaining balls: {remaining_balls - 1}")
                remaining_balls -= 1
                next_valid_position = pos + min_distance
                print(f"\t\tUpdated next_valid_position: {next_valid_position}")

            if remaining_balls == 0:
                print(f"\tAll balls placed successfully with min_distance {min_distance}")
                return True

        success = remaining_balls <= 0
        print(f"\tPlacement success: {success}")
        return success

    # Input Parameters
    print("\n--- Input Parameters ---")
    print(f"\tposition = {position}")
    print(f"\tm = {m}")

    # Sorting the positions
    position.sort()
    print("\n--- Sorted Positions ---")
    print(f"\tposition (sorted) = {position}")

    # Binary Search Initialization
    start_index, end_index = 1, (position[-1] - position[0]) // (m - 1)
    print("\n--- Binary Search Initialization ---")
    print(f"\tstart_index = {start_index}")
    print(f"\tend_index = {end_index}")

    # Main Binary Search Loop
    iteration_data = []
    iteration_count = 0

    while start_index < end_index:
        iteration_count += 1
        mid_index = 1 + (start_index + end_index) // 2
        print(f"\nIteration {iteration_count}:")
        print(f"\tWhile start_index ({start_index}) < end_index ({end_index}):")
        print(f"\tmid_index = {mid_index}")

        can_place = can_place_balls(mid_index)
        iteration_data.append([iteration_count, start_index, end_index, mid_index, can_place])

        if can_place:
            start_index = mid_index
            print(f"\tCan place balls with min_distance {mid_index}. New start_index = {start_index}")
        else:
            end_index = mid_index - 1
            print(f"\tCannot place balls with min_distance {mid_index}. New end_index = {end_index}")

    print(f"\nWhile loop ended because start_index ({start_index}) >= end_index ({end_index})")

    # Iteration Summary
    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Start Index", "End Index", "Mid Index", "Can Place"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    # Function Return
    print("\n--- Function Returning ---")
    result = start_index
    print(f"Maximum possible minimum magnetic force: {result}")
    return result


# <------------------------------------------------- June 21st, 2024 ------------------------------------------------->
# 1052. Grumpy Bookstore Owner

# A bookstore owner has a store open for `n` minutes, with `customers[i]` representing the number of customers entering
# at minute `i`.
# The owner can be grumpy (`grumpy[i]` is 1) or not (`grumpy[i]` is 0), affecting customer satisfaction, but can use a
# technique to avoid being grumpy for a single `minutes` period.
# Return the maximum number of customers that can be satisfied throughout the day.


def maxSatisfied1(customers: List[int], grumpy: List[int], minutes: int) -> int:
    """
    Calculates the maximum number of satisfied customers in a bookstore given a limited period where the owner can avoid
    being grumpy.

    The function first calculates the number of customers that are normally satisfied when the owner is not grumpy.
    It then uses a sliding window technique to find the period during which the owner can suppress their grumpiness to
    maximize the number of additionally satisfied customers.
    This is achieved by iterating through the customers and grumpy lists, keeping track of the maximum number of
    additionally satisfied customers over any given period of 'minutes' length.
    There is a slight optimization by combining the normal and additional satisfaction calculations into the same loop.

    The time complexity of this solution is O(n), where `n` is the length of the customers list,
    because it processes each element of the list a constant number of times.
    The space complexity is O(1), since it uses a fixed amount of additional space regardless of the input size.
    """
    # Input Parameters
    print("\n--- Input Parameters ---")
    print(f"\tcustomers = {customers}")
    print(f"\tgrumpy = {grumpy}")
    print(f"\tminutes = {minutes}")

    # Initialization
    print("\n--- Initialization ---")
    satisfied_customers = 0
    potential_customers = 0
    print(f"\tsatisfied_customers = {satisfied_customers}")
    print(f"\tpotential_customers = {potential_customers}")

    # Main Loop (Initial Window Calculation)
    print("\n--- Main Loop (Initial Window Calculation) ---")
    initial_window_data = []
    for i, (customer, grump) in enumerate(zip(customers[:minutes], grumpy[:minutes])):
        print(f"\n--- Minute {i + 1}/{minutes} ---")
        print(f"\tcustomer = {customer}, grump = {grump}")

        print(f"\tChecking if the owner is grumpy at minute {i}...")
        if grump:
            print(f"\t\tOwner is grumpy. Adding {customer} potential customers")
            print(f"\t\tPotential Customers: {potential_customers} + {customer} = {potential_customers + customer}")
            potential_customers += customer
        else:
            print(f"\t\tOwner is not grumpy. Adding satisfied customers: {customer}")
            print(f"\t\tSatisfied Customers: {satisfied_customers} + {customer} = {satisfied_customers + customer}")
            satisfied_customers += customer

        initial_window_data.append([i + 1, customer, grump, satisfied_customers, potential_customers])
        print(f"\tCurrent state: satisfied_customers = {satisfied_customers}, "
              f"potential_customers = {potential_customers}")

    max_potential_customers = potential_customers
    print(f"\nAfter initial window: max_potential_customers = {max_potential_customers}")

    # Iteration Summary (Initial Window)
    print("\n--- Iteration Summary (Initial Window) ---")
    headers = ["Minute", "Customer", "Grumpy", "Satisfied Customers", "Potential Customers"]
    print(tabulate(initial_window_data, headers=headers, tablefmt="fancy_grid"))

    # Sliding Window Loop
    print("\n--- Sliding Window Loop ---")
    sliding_window_data = []
    for minute in range(minutes, len(customers)):
        print(f"\n--- Minute {minute + 1}/{len(customers)} ---")

        print(f"\tcustomer = {customers[minute]}, grump = {grumpy[minute]}")

        print(f"\tChecking if the owner is grumpy at minute {minute + 1}...")
        if grumpy[minute]:
            print(f"\t\tOwner is grumpy. Adding {customers[minute]} potential customers")
            print(f"\t\tPotential Customers: {potential_customers} + {customers[minute]} = "
                  f"{potential_customers + customers[minute]}")
            potential_customers += customers[minute]

        else:
            print(f"\t\tOwner is not grumpy. Adding satisfied customers: {customers[minute]}")
            print(f"\t\tSatisfied Customers: {satisfied_customers} + {customers[minute]} = "
                  f"{satisfied_customers + customers[minute]}")
            satisfied_customers += customers[minute]

        print(f"\tWindow: [{minute - minutes + 2}, {minute + 1}]")

        # Remove unsatisfied customers that are outside the current window
        print(f"\tTrying to remove potential customers at minute {minute - minutes + 1} because it is no longer "
              f"inside the window...")
        if grumpy[minute - minutes]:
            print(f"\t\tRemoving {customers[minute - minutes]} customers, as the owner was grumpy at minute "
                  f"{minute - minutes + 1}.")
            print(f"\t\tPotential Customers: {potential_customers} - {customers[minute - minutes]} = "
                  f"{potential_customers - customers[minute - minutes]}")
            potential_customers -= customers[minute - minutes]
        else:
            print(f"\t\tNo customers to remove, owner was not grumpy at minute {minute - minutes + 1}.")

        print(
            f"\tBefore max calculation: max_potential_customers = {max_potential_customers}, "
            f"potential_customers = {potential_customers}")
        max_potential_customers = max(max_potential_customers, potential_customers)
        print(f"\tAfter max calculation: max_potential_customers = {max_potential_customers}")

        sliding_window_data.append([minute + 1, f"[{minute - minutes + 2}, {minute + 1}]", customers[minute],
                                    grumpy[minute], satisfied_customers, potential_customers, max_potential_customers])

    # Iteration Summary (Sliding Window)
    print("\n--- Iteration Summary (Sliding Window) ---")
    headers = ["Minute", "Window", "Customers", "Grumpy", "Satisfied Customers", "Potential Customers", "Max Potential"]
    print(tabulate(sliding_window_data, headers=headers, tablefmt="fancy_grid"))

    # Function Return
    print("\n--- Function Returning ---")
    result = satisfied_customers + max_potential_customers
    print(f"\tFinal satisfied_customers: {satisfied_customers}")
    print(f"\tFinal max_potential_customers: {max_potential_customers}")
    print(f"\tResult: {result}")
    return result


# <------------------------------------------------- June 22nd, 2024 ------------------------------------------------->
# 1248. Count Number of Nice Subarrays

# Given an array of integers `nums` and an integer `k`, return the number of nice subarrays.
# A continuous subarray is called nice if there are `k` odd numbers on it.


def numberOfSubarrays1(nums: List[int], k: int) -> int:
    """
    Counts the number of continuous subarrays within `nums` that contain exactly `k` odd numbers.

    This function uses a sliding window approach combined with a counting technique
    to efficiently find nice subarrays. It maintains a window that expands when
    odd numbers are encountered and contracts when the odd count exceeds k. The
    key insight is that once a nice subarray is found, any extension of it that
    doesn't include new odd numbers is also nice. This allows for efficient counting
    without explicitly listing all subarrays.

    The time complexity is O(n), where n is the length of nums. Each element is
    processed at most twice: once in the outer loop and potentially once in the
    inner while loop. The space complexity is O(1) as it uses only a constant
    amount of extra space regardless of the input size.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tk = {k}")

    print("\n--- Initialization ---")
    total_nice_subarrays = 0  # Total count of subarrays with exactly k odd numbers
    current_nice_subarrays = 0  # Count of nice subarrays ending at current position
    odd_count = 0  # Number of odd numbers in the current window
    start_index = 0  # Starting index of the current window
    print(f"\ttotal_nice_subarrays = {total_nice_subarrays}")
    print(f"\tcurrent_nice_subarrays = {current_nice_subarrays}")
    print(f"\todd_count = {odd_count}")
    print(f"\tstart_index = {start_index}")

    iteration_data = []

    print("\n--- Main Loop (Iterating over each number in nums) ---")
    for idx, num in enumerate(nums):
        print(f"\n--- Processing Number {idx + 1}/{len(nums)}: num = {num} ---")

        print(f"\tIs {num} odd? {num % 2 == 1}")
        if num % 2 == 1:
            odd_count += 1
            current_nice_subarrays = 0
            print(f"\t\todd_count incremented to {odd_count}")
            print(f"\t\tcurrent_nice_subarrays reset to {current_nice_subarrays} (New odd starts potential subarrays)")

        print(f"\n\t--- While loop (Contracting window if odd_count ({odd_count}) == k ({k})) ---")
        while odd_count == k:
            print(f"\t\todd_count ({odd_count}) == k ({k})")
            print(f"\t\t--- Contracting Window ---")
            current_nice_subarrays += 1
            print(f"\t\tcurrent_nice_subarrays incremented to {current_nice_subarrays} (Found a nice subarray)")

            print(f"\t\tChecking if nums[{start_index}] = {nums[start_index]} is odd...")
            if nums[start_index] % 2 == 1:
                print(f"\t\t\todd_count decremented to {odd_count} - 1 = {odd_count - 1}")
                odd_count -= 1
            else:
                print(f"\t\t\tNo change to odd_count, nums[{start_index}] is not odd.")
            start_index += 1
            print(f"\t\tstart_index incremented to {start_index} (Contract window start)")

        if current_nice_subarrays > 0:
            print(f"\ttotal_nice_subarrays updated to {total_nice_subarrays} + {current_nice_subarrays} = "
                  f"{total_nice_subarrays + current_nice_subarrays}")
            total_nice_subarrays += current_nice_subarrays

        # Collecting Iteration Data
        iteration_data.append([idx + 1, num, odd_count, current_nice_subarrays, total_nice_subarrays, start_index])

    print("\n--- Iteration Summary ---")
    headers = ["Number Index", "Number", "Odd Count", "Current Nice Subarrays", "Total Nice Subarrays", "Start Index"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tReturning total_nice_subarrays = {total_nice_subarrays}")
    return total_nice_subarrays


def numberOfSubarrays2(nums: List[int], k: int) -> int:
    """
    Counts the number of continuous subarrays within `nums` that contain exactly `k`
    odd numbers.

    This function uses a two-pass approach to efficiently count nice subarrays.
    In the first pass, it counts the number of even integers between each pair
    of odd integers, storing these counts in the 'even_counts' list. The second
    pass uses these counts to calculate the total number of nice subarrays.
    This method leverages the fact that for each sequence of k odd numbers,
    the number of nice subarrays is the product of the number of even numbers
    before the first odd number and after the last odd number in the sequence.

    The time complexity is O(n), where n is the length of nums, as it makes two
    passes through the input list. The space complexity is O(1) as it uses only
    a constant amount of extra space regardless of the input size.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tk = {k}")

    print("\n--- Initialization ---")
    even_counts = []
    current_even_count = 1
    print(f"\teven_counts = {even_counts}")
    print(f"\tcurrent_even_count = {current_even_count}")

    print("\n--- First Pass: Counting Even Numbers ---")
    for i, num in enumerate(nums):
        print(f"\n--- Processing number {i + 1}/{len(nums)}: {num} ---")
        print(f"\tCurrent state: current_even_count = {current_even_count}")

        print(f"\tIs {num} even? {num % 2 == 0}")
        if num % 2 == 0:
            current_even_count += 1
            print(f"\t\tNumber is even. Incrementing current_even_count to {current_even_count}")
        else:
            print(f"\t\tNumber is odd. Appending current_even_count {current_even_count} to even_counts")
            even_counts.append(current_even_count)
            print(f"\t\tResetting current_even_count to 1")
            current_even_count = 1

        print(f"\tUpdated even_counts: {even_counts}")

    print("\n--- Finalizing even_counts ---")
    even_counts.append(current_even_count)
    print(f"\tAppending final current_even_count {current_even_count} to even_counts")
    print(f"\tFinal even_counts: {even_counts}")

    print("\n--- Calculating Nice Subarrays ---")
    total_nice_subarrays = 0
    iteration_data = []
    for i, (left_even_count, right_even_count) in enumerate(zip(even_counts, even_counts[k:])):
        print(f"\n--- Subarray {i + 1}/{len(even_counts) - k} ---")
        print(f"\tleft_even_count = {left_even_count}, right_even_count = {right_even_count}")

        nice_subarrays = left_even_count * right_even_count
        total_nice_subarrays += nice_subarrays

        print(f"\tCalculation: {left_even_count} * {right_even_count} = {nice_subarrays}")
        print(f"\tUpdated total_nice_subarrays: {total_nice_subarrays}")

        iteration_data.append([i + 1, left_even_count, right_even_count, nice_subarrays, total_nice_subarrays])

    print("\n--- Iteration Summary ---")
    headers = ["Subarray", "Left Even Count", "Right Even Count", "Nice Subarrays", "Total Nice Subarrays"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Result: {total_nice_subarrays}")
    return total_nice_subarrays


# <------------------------------------------------- June 23rd, 2024 ------------------------------------------------->
# 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

# Given an array of integers `nums` and an integer `limit`, return the size of the longest non-empty contiguous subarray
# such that the absolute difference between any two elements of this subarray is less than or equal to `limit`.


def longestSubarray1(nums: List[int], limit: int) -> int:
    """
    Finds the length of the longest subarray where the absolute difference between any two elements is at most 'limit'.

    This function uses a sliding window approach combined with a sorted list to track the elements in the current
    window.
    It iterates through the array once, maintaining a sorted representation of the current window.
    When a new element is added, it's inserted into its correct position in the sorted list.
    If the difference between the maximum and minimum elements (which are always at the ends of the sorted list)
    exceeds the limit, the window is shrunk from the left by removing the leftmost element from the sorted list.

    The time complexity of this solution is O(n^2) in the worst case, where n is the length of the input array.
    This is because for each element (O(n)), we might need to perform an insertion into a sorted list (O(n) using
    bisect.insort).
    However, in practice, it can be much faster if the subarrays tend to be short.
    The space complexity is O(n), as in the worst case, the sorted list might contain all elements of the input array.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tlimit = {limit}")

    print("\n--- Initialization ---")
    left_index = 0
    sorted_nums = []
    print(f"\tleft_index = {left_index}")
    print(f"\tsorted_nums = {sorted_nums}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for i, num in enumerate(nums):
        print(f"\n--- Iteration {i + 1}/{len(nums)} ---")
        print(f"\tCurrent number: {num}")
        print(f"\tCurrent state: left_index = {left_index}, sorted_nums = {sorted_nums}")

        print("\tInserting new element into sorted_nums:")
        bisect.insort(sorted_nums, num)
        print(f"\t\tsorted_nums after insertion: {sorted_nums}")

        print("\tChecking if the difference between the maximum and minimum elements exceeds the limit...")
        if sorted_nums[-1] - sorted_nums[0] > limit:
            print(
                f"\t\tCondition true: {sorted_nums[-1]} - {sorted_nums[0]} ="
                f" {sorted_nums[-1] - sorted_nums[0]} > {limit}")
            print("\t\tAction: Removing leftmost element and updating left_index")
            removed_index = bisect.bisect(sorted_nums, nums[left_index]) - 1
            removed_value = sorted_nums.pop(removed_index)
            print(f"\t\t\tRemoved {removed_value} from index {removed_index}")
            left_index += 1
            print(f"\t\t\tUpdated left_index to {left_index}")
        else:
            print(
                f"\t\tCondition false: {sorted_nums[-1]} - {sorted_nums[0]} ="
                f" {sorted_nums[-1] - sorted_nums[0]} <= {limit}")
            print("\t\tAction: No action needed")

        current_length = i - left_index + 1
        iteration_data.append([i + 1, num, left_index, sorted_nums.copy(), current_length])
        print(f"\tCurrent subarray length: {current_length}")

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Current Number", "Left Index", "Sorted Nums", "Current Length"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    result = len(nums) - left_index
    print(f"\tCalculating result: len(nums) - left_index = {len(nums)} - {left_index} = {result}")
    print(f"\tFinal Result: {result}")
    return result


def longestSubarray2(nums: List[int], limit: int) -> int:
    """
    Finds the length of the longest subarray where the absolute difference between any two elements is at most 'limit'.

    This function uses a sliding window approach combined with two heaps (min and max) to efficiently track the minimum
    and maximum elements in the current window.
    As we iterate through the array, we expand the window to the right.
    If the difference between the max and min elements exceeds the limit, we shrink the window from the left until the
    condition is satisfied again.
    The heaps allow us to quickly retrieve and update the min and max elements as the
    window changes.

    The time complexity of this solution is O(n log n), where n is the length of the input array. This is because we
    perform n iterations, and in each iteration, we might need to perform heap operations (push and pop) which take
    O(log n) time. The space complexity is O(n) as in the worst case, all elements might be stored in the heaps.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tlimit = {limit}")

    print("\n--- Initialization ---")
    max_heap = []
    min_heap = []
    left_index = 0
    max_length = 0
    print(f"\tmax_heap = {max_heap}")
    print(f"\tmin_heap = {min_heap}")
    print(f"\tleft_index = {left_index}")
    print(f"\tmax_length = {max_length}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for right_index, num in enumerate(nums):
        print(f"\n--- Iteration {right_index + 1}/{len(nums)} ---")
        print(f"\tCurrent number: {num}")
        print(f"\tCurrent state: left_index = {left_index}, max_length = {max_length}")
        print(f"\tmax_heap = {max_heap}")
        print(f"\tmin_heap = {min_heap}")

        print("\tPushing current element into heaps:")
        heapq.heappush(max_heap, (-num, right_index))
        heapq.heappush(min_heap, (num, right_index))
        print(f"\t\tAfter push: max_heap = {max_heap}")
        print(f"\t\tAfter push: min_heap = {min_heap}")

        print("\tChecking if the difference between the max and min elements exceeds the "
              "limit and adjusting the window:")
        while -max_heap[0][0] - min_heap[0][0] > limit:
            print(
                f"\t\tCondition true: {-max_heap[0][0]} - {min_heap[0][0]} = "
                f"{-max_heap[0][0] - min_heap[0][0]} > {limit}")
            left_index = min(max_heap[0][1], min_heap[0][1]) + 1
            print(f"\t\tUpdating left_index to {left_index}")

            print("\t\tRemoving elements outside the current window:")
            while max_heap[0][1] < left_index:
                removed = heapq.heappop(max_heap)
                print(f"\t\t\tRemoved {removed} from max_heap")
            while min_heap[0][1] < left_index:
                removed = heapq.heappop(min_heap)
                print(f"\t\t\tRemoved {removed} from min_heap")
        else:
            print(
                f"\t\tCondition false: {-max_heap[0][0]} - {min_heap[0][0]} = "
                f"{-max_heap[0][0] - min_heap[0][0]} <= {limit}")

        current_length = right_index - left_index + 1
        max_length = max(max_length, current_length)
        print(f"\tUpdated max_length: {max_length}")

        iteration_data.append([right_index + 1, num, left_index, current_length, max_length,
                               [(-v[0], v[1]) for v in max_heap[:3]], [v for v in min_heap[:3]]])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Current Number", "Left Index", "Current Length", "Max Length",
               "Max Heap (top 3)", "Min Heap (top 3)"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Result: {max_length}")
    return max_length


def longestSubarray3(nums: List[int], limit: int) -> int:
    """
    Finds the length of the longest subarray where the absolute difference between any two elements is at most 'limit'.

    This function uses a sliding window approach with two monotonic queues (deques) to efficiently track the maximum
    and minimum elements in the current window.
    As we iterate through the array, we maintain a monotonically decreasing queue for maximum values and a monotonically
    increasing queue for minimum values.
    This allows us to quickly access the maximum and minimum elements of the current window at any time.
    When the difference between these extremes exceeds the limit, we shrink the window from the left,
    updating the queues accordingly.

    The time complexity of this solution is O(n), where n is the length of the input array.
    This is because each element is pushed and popped at most once from each deque,
    and all operations on deques are O(1).
    The space complexity is O(n) in the worst case, as the deques might need to store all elements of the input array.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tlimit = {limit}")

    print("\n--- Initialization ---")
    max_queue = deque()
    min_queue = deque()
    left_index = 0
    print(f"\tmax_queue = {list(max_queue)}")
    print(f"\tmin_queue = {list(min_queue)}")
    print(f"\tleft_index = {left_index}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for right_index, num in enumerate(nums):
        print(f"\n--- Iteration {right_index + 1}/{len(nums)} ---")
        print(f"\tCurrent number: {num}")
        print(f"\tCurrent state: left_index = {left_index}")
        print(f"\tmax_queue = {list(max_queue)}")
        print(f"\tmin_queue = {list(min_queue)}")

        print("\tUpdating max_queue:")
        while max_queue and num > max_queue[-1]:
            removed = max_queue.pop()
            print(f"\t\tRemoved {removed} from max_queue")
        max_queue.append(num)
        print(f"\t\tAppended {num} to max_queue")
        print(f"\t\tUpdated max_queue = {list(max_queue)}")

        print("\tUpdating min_queue:")
        while min_queue and num < min_queue[-1]:
            removed = min_queue.pop()
            print(f"\t\tRemoved {removed} from min_queue")
        min_queue.append(num)
        print(f"\t\tAppended {num} to min_queue")
        print(f"\t\tUpdated min_queue = {list(min_queue)}")

        print("\tChecking if the difference between the max and min elements exceeds the limit...")
        if max_queue[0] - min_queue[0] > limit:
            print(f"\t\tCondition true: {max_queue[0]} - {min_queue[0]} = {max_queue[0] - min_queue[0]} > {limit}")
            print("\t\tShrinking window:")
            if max_queue[0] == nums[left_index]:
                removed = max_queue.popleft()
                print(f"\t\t\tRemoved {removed} from max_queue")
            if min_queue[0] == nums[left_index]:
                removed = min_queue.popleft()
                print(f"\t\t\tRemoved {removed} from min_queue")
            left_index += 1
            print(f"\t\t\tUpdated left_index to {left_index}")
        else:
            print(f"\t\tCondition false: {max_queue[0]} - {min_queue[0]} = {max_queue[0] - min_queue[0]} <= {limit}")
            print("\t\tNo action needed")

        current_length = right_index - left_index + 1
        iteration_data.append([right_index + 1, num, left_index, current_length, list(max_queue), list(min_queue)])
        print(f"\tCurrent window length: {current_length}")

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Current Number", "Left Index", "Current Length", "Max Queue", "Min Queue"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    result = len(nums) - left_index
    print(f"\tCalculating result: len(nums) - left_index = {len(nums)} - {left_index} = {result}")
    print(f"\tFinal Result: {result}")
    return result

# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for June 17th, 2024
# Expected output: True
# judgeSquareSum1(c=98)
# judgeSquareSum2(c=98)

# Test cases for June 18th, 2024
# Expected output: 160
# maxProfitAssignment1(difficulty=[5, 12, 2, 6, 15, 7], profit=[10, 30, 20, 25, 50, 35], worker=[10, 5, 7, 12, 8])
# maxProfitAssignment2(difficulty=[5, 12, 2, 6, 15, 7], profit=[10, 30, 20, 25, 50, 35], worker=[10, 5, 7, 12, 8])
# maxProfitAssignment3(difficulty=[5, 12, 2, 6, 15, 7], profit=[10, 30, 20, 25, 50, 35], worker=[10, 5, 7, 12, 8])

# Test cases for June 19th, 2024
# Expected output: 9
# minDays1_1(bloom_day=[3, 2, 4, 9, 10, 4, 3, 4], m=3, k=2)
# minDays1_2(bloom_day=[3, 2, 4, 9, 10, 4, 3, 4], m=3, k=2)

# Test cases for June 20th, 2024
# Expected output: 3
# maxDistance1(position=[64, 16, 128, 8, 2, 32, 1, 4], m=4)

# Test cases for June 21st, 2024
# Expected output: 16
# maxSatisfied1(customers=[1, 0, 1, 2, 1, 1, 7, 5], grumpy=[0, 1, 0, 1, 0, 1, 0, 1], minutes=3)

# Test cases for June 22nd, 2024
# Expected output: 10
# numberOfSubarrays1(nums=[2, 1, 2, 1, 2, 1, 2, 2], k=2)
# numberOfSubarrays2(nums=[2, 1, 2, 1, 2, 1, 2, 2], k=2)

# Test cases for June 23rd, 2024
# Expected output: 4
# longestSubarray1(nums=[10, 1, 2, 4, 7, 2], limit=5)
# longestSubarray2(nums=[10, 1, 2, 4, 7, 2], limit=5)
# longestSubarray3(nums=[10, 1, 2, 4, 7, 2], limit=5)
