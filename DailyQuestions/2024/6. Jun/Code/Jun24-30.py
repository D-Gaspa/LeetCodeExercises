# June 2024, Week 5: June 24th - June 30th
from collections import deque
from typing import List


# <------------------------------------------------- June 24th, 2024 ------------------------------------------------->
# 995. Minimum Number of K Consecutive Bit Flips

# Given a binary array `nums` and an integer `k`, return the minimum number of k-bit flips required so that there is no
# `0` in the array; if it is not possible, return `-1.
# A k-bit flip is choosing a subarray of length `k` from `nums` and simultaneously changing every `0` in the
# subarray to `1`, and every `1` in the subarray to `0`.


def minKBitFlips1(nums: List[int], k: int) -> int:
    if k == 1:
        return nums.count(0)

    flip_window_deque = deque()
    current_flipped_state = 0
    total_flips = 0

    for index, num in enumerate(nums):
        if index >= k:
            current_flipped_state ^= flip_window_deque.popleft()

        if current_flipped_state == num:
            if index + k > len(nums):
                return -1
            flip_window_deque.append(1)
            current_flipped_state ^= 1
            total_flips += 1
        else:
            flip_window_deque.append(0)

    return total_flips


def minKBitFlips2(nums: List[int], k: int) -> int:
    if k == 1:
        return nums.count(0)

    flip_count = 0
    expected_state = 1

    for index in range(k, len(nums)):
        if nums[index - k] != expected_state:
            expected_state = 1 - expected_state
            flip_count += 1

        if expected_state == 0:
            nums[index] = 1 - nums[index]

    last_value = nums[-1]
    if not all(num == last_value for num in nums[-k:]):
        return -1

    return flip_count + (1 if last_value != expected_state else 0)


def minKBitFlips3(nums: List[int], k: int) -> int:
    if k == 1:
        return nums.count(0)

    n = len(nums)
    active_flips = 0
    total_flips = 0

    for index in range(n):
        if index >= k and nums[index - k] == 2:
            active_flips -= 1

        if (active_flips % 2) == nums[index]:
            if index + k > n:
                return -1
            nums[index] = 2
            active_flips += 1
            total_flips += 1

    return total_flips


# <------------------------------------------------- June 25th, 2024 ------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass

# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024
