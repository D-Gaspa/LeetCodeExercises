# Daily Question (01/05/2024)
from bisect import bisect_left


# 300. Longest Increasing Subsequence
# Medium

# Given an integer array nums, return the length of the longest strictly increasing subsequence.

# O(n^2) solution with DP
def lengthOfLIS1(nums):
    dp = [1] * len(nums)  # Initialize the dp array with 1s

    # Iterate over each element in the array
    for i in range(len(nums)):
        # Check all elements before the current element
        for j in range(i):
            # If a smaller element is found
            if nums[i] > nums[j]:
                # Update the dp value for the current element.
                dp[i] = max(dp[i], dp[j] + 1)

    # The maximum value in dp array is the length of the longest increasing subsequence
    return max(dp)


# n log n solution
def lengthOfLIS2(nums):
    dp = [0] * len(nums)  # Initialize an array to hold the smallest tail of all increasing subsequences
    size = 0  # Size of the longest subsequence found so far

    for num in nums:
        # Binary search to find the correct position of num in dp array
        i, j = 0, size
        while i != j:
            mid = (i + j) // 2  # Find the middle index
            # If num is larger, it can extend the subsequence
            if dp[mid] < num:
                i = mid + 1
            else:
                j = mid

        # Update the dp array and size
        dp[i] = num  # Place num in its correct position
        size = max(i + 1, size)  # Update the size of the longest subsequence

    return size  # The size is the length of the longest increasing subsequence


# n log n solution with binary search using bisect_left
def lengthOfLIS3(nums):
    sub = []  # Initialize the subsequence array

    for num in nums:
        i = bisect_left(sub, num)  # Find the index where the current element should be inserted

        # If num is greater than all elements in sub, append it to sub
        if i == len(sub):
            sub.append(num)

        # Otherwise, replace the element at index i with num
        else:
            sub[i] = num

    return len(sub)
