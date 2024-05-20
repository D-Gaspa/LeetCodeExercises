from typing import List


# Week 4: May 20th - 26th

# <-------------------------------------------------- May 20th, 2024 -------------------------------------------------->
# 1863. Sum of All Subset XOR Totals

# Given an array nums, return the sum of all XOR totals for every subset of nums.

def subsetXORSum1(nums: List[int]) -> int:
    """
    Calculates the sum of XOR totals over all possible subsets of an integer list.

    This solution uses bit manipulation to generate all subsets and calculate their XOR totals.
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
    """

    combined_bits = 0  # Accumulate bitwise OR of all numbers

    for num in nums:
        combined_bits |= num

    # Calculate the sum of XOR totals by multiplying the combined bits with 2^(n-1)
    return combined_bits * (1 << (len(nums) - 1))


def subsetXORSum3(nums: List[int]) -> int:
    def dfs(i, xor):
        if i == len(nums):
            return xor
        return dfs(i + 1, xor ^ nums[i]) + dfs(i + 1, xor)

    return dfs(0, 0)

# <-------------------------------------------------- May 21st, 2024 -------------------------------------------------->
