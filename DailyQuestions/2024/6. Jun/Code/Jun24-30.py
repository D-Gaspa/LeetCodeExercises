# June 2024, Week 5: June 24th - June 30th
from collections import deque
from typing import List


# <------------------------------------------------- June 24th, 2024 ------------------------------------------------->
# 995. Minimum Number of K Consecutive Bit Flips

# Given a binary array `nums` and an integer `k`, return the minimum number of k-bit flips required so that there is no
# `0` in the array; if it is not possible, return `-1.
# A k-bit flip is choosing a contiguous subarray of length `k` from `nums` and simultaneously changing every `0` in the
# subarray to `1`, and every `1` in the subarray to `0`.


def minKBitFlips1(nums: List[int], k: int) -> int:
    """
    Determines the minimum number of k-bit flips required to convert all elements in `nums` to 1.

    This function uses a sliding window approach with a deque to efficiently track the flips.
    It maintains a 'current_flipped_state' to represent the cumulative effect of flips on the
    current element, avoiding the need to actually modify the input array.
    The algorithm iterates through the array once, deciding whether to flip at each position based on the
    current state and the original value.
    This approach allows for efficient handling of overlapping flips without the need
    to recalculate previous operations.

    The time complexity of this solution is O(n), where `n` is the length of `nums`, because it
    processes each element once with constant-time operations.
    The space complexity is O(k)
    due to the deque storing at most `k` elements to track the sliding window of flips.
    """
    if k == 1:
        return nums.count(0)  # Optimization for k=1 case

    flip_window_deque = deque()
    current_flipped_state = 0
    total_flips = 0

    for index, num in enumerate(nums):
        if index >= k:
            # Remove the effect of the flip that's now out of the window
            current_flipped_state ^= flip_window_deque.popleft()

        if current_flipped_state == num:
            # The current state matches the original value, so a flip is needed
            if index + k > len(nums):
                return -1  # Not enough elements left for a flip
            flip_window_deque.append(1)
            current_flipped_state ^= 1
            total_flips += 1
        else:
            flip_window_deque.append(0)  # No flip needed at this position

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

# Test cases for June 24th, 2024
# Expected output: 3
minKBitFlips1(nums=[0, 0, 0, 1, 0, 1, 1, 0], k=3)
minKBitFlips2(nums=[0, 0, 0, 1, 0, 1, 1, 0], k=3)
minKBitFlips3(nums=[0, 0, 0, 1, 0, 1, 1, 0], k=3)

# Test cases for June 25th, 2024

# Test cases for June 26th, 2024

# Test cases for June 27th, 2024

# Test cases for June 28th, 2024

# Test cases for June 29th, 2024

# Test cases for June 30th, 2024
