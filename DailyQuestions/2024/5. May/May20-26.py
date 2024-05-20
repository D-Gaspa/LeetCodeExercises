from typing import List


# Week 4: May 20th - 26th

# <-------------------------------------------------- May 20th, 2024 -------------------------------------------------->
# 1863. Sum of All Subset XOR Totals

# Given an array nums, return the sum of all XOR totals for every subset of nums.

def subsetXORSum1(nums: List[int]) -> int:
    n = len(nums)
    ans = 0
    for i in range(1 << n):
        xor = 0
        for j in range(n):
            if i & (1 << j):
                xor ^= nums[j]
        ans += xor
    return ans


def subsetXORSum2(nums: List[int]) -> int:
    def dfs(i, xor):
        if i == len(nums):
            return xor
        return dfs(i + 1, xor ^ nums[i]) + dfs(i + 1, xor)

    return dfs(0, 0)

# <-------------------------------------------------- May 21st, 2024 -------------------------------------------------->
