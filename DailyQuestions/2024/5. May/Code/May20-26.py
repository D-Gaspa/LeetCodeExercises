from collections import defaultdict
from typing import List


# Week 4: May 20th - 26th

# <-------------------------------------------------- May 20th, 2024 -------------------------------------------------->
# 1863. Sum of All Subset XOR Totals

# Given an array nums, return the sum of all XOR totals for every subset of nums.

def subsetXORSum1(nums: List[int]) -> int:
    """
    Calculates the sum of XOR totals over all possible subsets of an integer list.

    This solution uses bit manipulation to generate all subsets and calculate their XOR totals.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """
    n = len(nums)
    ans = 0

    # Iterate over all possible subsets using bit representations (2^n subsets) [1 << n = 2^n]
    for i in range(1 << n):
        xor = 0

        for j in range(n):  # Calculate XOR total for the current subset.
            if i & (1 << j):  # Bitwise trick to check if j-th element is in this subset
                xor ^= nums[j]
        ans += xor

    return ans


def subsetXORSum2(nums: List[int]) -> int:
    """Calculates the sum of XOR totals over all possible subsets of an integer list.

    This solution uses a bitwise OR operation to combine all numbers and then calculates the sum of XOR totals.
    The time complexity of this solution is O(n) where n is the length of the input list.
    """
    combined_bits = 0  # Accumulate bitwise OR of all numbers

    for num in nums:
        combined_bits |= num

    # Calculate the sum of XOR totals by multiplying the combined bits with 2^(n-1)
    return combined_bits * (1 << (len(nums) - 1))


# <-------------------------------------------------- May 21st, 2024 -------------------------------------------------->
# 78. Subsets

# Given an integer array nums of unique elements, return all possible subsets (the power set).


def subsets1(nums: List[int]) -> List[List[int]]:
    """
    Generates all possible subsets of an integer list.

    This solution uses bit manipulation to generate all subsets.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """
    n = len(nums)
    ans = []

    # Iterate over all possible subsets using bit representations (2^n subsets) [1 << n = 2^n]
    for i in range(1 << n):
        subset = []

        for j in range(n):
            if i & (1 << j):  # Bitwise trick to check if j-th element is in this subset
                subset.append(nums[j])

        ans.append(subset)

    return ans


def subsets2(nums: List[int]) -> List[List[int]]:
    """
    Generates all possible subsets of an integer list.

    This solution uses backtracking to generate all subsets.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """

    def backtrack(start: int, current_subset: List[int]):
        """Backtracking function to generate all subsets."""
        ans.append(current_subset[:])

        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()  # Backtrack by removing the last element

    ans = []
    backtrack(0, [])
    return ans


def subsets3(nums: List[int]) -> List[List[int]]:
    """
    Generates all possible subsets of an integer list.

    This solution uses an iterative approach to generate all subsets.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """
    ans = [[]]

    for num in nums:
        ans += [curr + [num] for curr in ans]  # Add the current number to all existing subsets

    return ans


# <-------------------------------------------------- May 22nd, 2024 -------------------------------------------------->
# 131. Palindrome Partitioning

# Given a string 's', partition 's' such that every substring of the partition is a palindrome
# Return all possible palindrome partitions of 's'.


def partition1(s: str) -> List[List[str]]:
    """
    Generates all possible palindrome partitions of a string.

    This solution uses backtracking to generate all palindrome partitions.
    The time complexity of this solution is O(n * 2^n) where n is the length of the input string.
    """

    def is_palindrome(s, start, end):
        """Checks if a substring is a palindrome."""
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    def backtrack(start: int, current_partition: List[str]):
        """Backtracking function to generate all palindrome partitions."""
        if start == len(s):
            result.append(current_partition[:])  # Found a valid partition
            return

        for end in range(start, len(s)):
            if is_palindrome(s, start, end):
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()  # Backtrack by removing the last element

    result = []
    backtrack(0, [])
    return result


def partition2(s: str) -> List[List[str]]:
    """
    Generates all possible palindrome partitions of a string.

    This solution uses dynamic programming and backtracking to generate all palindrome partitions.
    The time complexity of this solution is O(n * 2^n) where n is the length of the input string.
    """
    n = len(s)
    dp_table = [[False] * n for _ in range(n)]  # DP table to store palindrome substrings

    for start in range(n - 1, -1, -1):
        for end in range(start, n):
            # A substring is a palindrome if the start and end characters are the same,
            # and the substring between them is also a palindrome
            if s[start] == s[end] and (end - start < 2 or dp_table[start + 1][end - 1]):
                dp_table[start][end] = True

    def backtrack(start: int, current_partition: List[str]):
        """Backtracking function to generate all palindrome partitions."""
        if start == n:
            result.append(current_partition[:])  # Found a valid partition
            return

        for end in range(start, n):
            if dp_table[start][end]:
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()  # Backtrack by removing the last element

    result = []
    backtrack(0, [])
    return result


# <-------------------------------------------------- May 23rd, 2024 -------------------------------------------------->
# 2597. The Number of Beautiful Subsets

# Count the number of non-empty subsets in an array of positive integers where no two elements have an absolute
# difference equal to a given positive integer k.


def beautifulSubsets1(nums: List[int], k: int) -> int:
    """
    Counts the number of beautiful subsets in an integer list.

    This solution uses backtracking to generate all subsets and check if they are beautiful.
    The time complexity of this solution is O(2^n) where n is the length of the input list.
    """
    n = len(nums)
    beautiful_count = 0
    current_subset = []

    def backtrack(start_index: int):
        nonlocal beautiful_count  # Access the outer variable

        # Base case: a subset is found, check if it's beautiful
        if len(current_subset) > 0:
            for i in range(len(current_subset) - 1):
                if abs(current_subset[i] - current_subset[-1]) == k:
                    return  # Not beautiful, prune the search
            beautiful_count += 1

        # Recursive case: try adding each remaining element
        for i in range(start_index, n):
            current_subset.append(nums[i])
            backtrack(i + 1)  # Explore subsets starting from the next index
            current_subset.pop()  # Remove the last added element (backtracking)

    backtrack(0)  # Start backtracking from the beginning
    return beautiful_count


def beautifulSubsets2(nums: List[int], k: int) -> int:
    """
    Counts the number of beautiful subsets in the given list of numbers.

    This solution uses dynamic programming and memoization to count beautiful subsets.
    The time complexity of this solution is O(2^n) where n is the length of the input list.
    """
    total_count = 1
    remainder_groups = defaultdict(lambda: defaultdict(int))
    memo = {}

    def count_beautiful_subsets(subsets: List[tuple], current_index: int) -> int:
        """Recursively counts beautiful subsets starting from the current index."""
        if current_index == len(subsets):
            return 1  # Base case: empty subset is beautiful

        # Create a key for the memoization dictionary based on the current index and remaining subsets
        key = (current_index, tuple(subsets[current_index:]))
        if key in memo:
            return memo[key]  # Return memoized result if available

        # Divide the subsets into two groups, excluding and including the current number, then count them
        exclude_count = count_beautiful_subsets(subsets, current_index + 1)
        include_count = (1 << subsets[current_index][1]) - 1

        # If the next number is 'k' apart from the current number, it must be in a different subset, so skip it
        if current_index + 1 < len(subsets) and subsets[current_index + 1][0] - subsets[current_index][0] == k:
            include_count *= count_beautiful_subsets(subsets, current_index + 2)
        else:
            # Otherwise, the next number can be in the same subset, so include it
            include_count *= count_beautiful_subsets(subsets, current_index + 1)

        # Store the total count in memoization dictionary for future use, then return it
        total_count = exclude_count + include_count
        memo[key] = total_count
        return total_count

    # Group numbers by their remainders when divided by k
    for num in nums:
        remainder_groups[num % k][num] += 1

    # Calculate beautiful subsets for each remainder group
    for group in remainder_groups.values():
        sorted_subsets = sorted(group.items())
        total_count *= count_beautiful_subsets(sorted_subsets, 0)

    return total_count - 1  # Exclude the empty subset


# <-------------------------------------------------- May 24th, 2024 -------------------------------------------------->
