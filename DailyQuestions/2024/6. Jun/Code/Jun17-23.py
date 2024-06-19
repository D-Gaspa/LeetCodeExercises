import math
from bisect import bisect_right
from typing import List

from tabulate import tabulate


# Week 3: June 17th - June 23rd, 2024

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
    print(f"\tend_index = {end_index}")

    # Main Loop (Two-Pointer Search)
    print("\n--- Main Loop (Two-Pointer Search) ---")
    iteration_data = []  # To collect data for the iteration summary
    iteration = 0
    while start_index <= end_index:
        iteration += 1
        print(f"\nWhile {start_index} <= {end_index}: (Iteration {iteration})")
        squares_sum = start_index * start_index + end_index * end_index
        print(f"\tsquares_sum = {squares_sum} ({start_index}^2 + {end_index}^2)")

        # Decision Point
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
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <------------------------------------------------- June 20th, 2024 ------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <------------------------------------------------- June 21st, 2024 ------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <------------------------------------------------- June 22nd, 2024 ------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <------------------------------------------------- June 23rd, 2024 ------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for June 17th, 2024
# judgeSquareSum1(98)  # Expected output: True
# judgeSquareSum2(98)  # Expected output: True

# Test cases for June 18th, 2024
difficulty = [5, 12, 2, 6, 15, 7, 9]
profit = [10, 30, 20, 25, 50, 35, 40]
worker = [10, 5, 7, 12, 8]
# maxProfitAssignment1(difficulty, profit, worker)  # Expected output: 170
# maxProfitAssignment2(difficulty, profit, worker)  # Expected output: 170
# maxProfitAssignment3(difficulty, profit, worker)  # Expected output: 170

# Test cases for June 19th, 2024

# Test cases for June 20th, 2024

# Test cases for June 21st, 2024

# Test cases for June 22nd, 2024

# Test cases for June 23rd, 2024
