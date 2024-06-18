import math

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
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


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

# Test cases for June 19th, 2024

# Test cases for June 20th, 2024

# Test cases for June 21st, 2024

# Test cases for June 22nd, 2024

# Test cases for June 23rd, 2024
