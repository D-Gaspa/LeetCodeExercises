# July, 2024
# <-------------------------------------------------- Week 1, July -------------------------------------------------->
# 1101. The Earliest Moment When Everyone Becomes Friends
from typing import List

from Utils.graph_utils import UnionFind


# Given `n` people in a social group labeled from `0` to `n - 1` and an array `logs` where
# `logs[i] = [timestamp_i, x_i, y_i]` indicates that `x_i` and `y_i` become friends at time `timestamp_i`,
# return the earliest time when every person is acquainted with every other person, or `-1` if it is not possible.
# Friendship is symmetric and transitive.


def earliestAcq1(logs: List[List[int]], n: int) -> int:
    if len(logs) < n - 1:
        return -1

    logs.sort(key=lambda x: x[0])
    friends_uf = UnionFind(n)

    for timestamp, person_1, person_2 in logs:
        friends_uf.union(person_1, person_2)
        if friends_uf.is_single_component():
            return timestamp

    return -1


# <-------------------------------------------------- Week 2, July -------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <-------------------------------------------------- Week 3, July -------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <-------------------------------------------------- Week 4, July -------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <-------------------------------------------------- Week 5, July -------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->
# Test cases for Week 1, July
# Expected output: 3
print(earliestAcq1(logs=[[0, 2, 0], [1, 0, 1], [3, 0, 3], [4, 1, 2], [7, 3, 1]], n=4))

# Expected output: 20,190,301
print(earliestAcq1(
    logs=[[20190101, 0, 1], [20190104, 3, 4], [20190107, 2, 3], [20190211, 1, 5], [20190224, 2, 4], [20190301, 0, 3],
          [20190312, 1, 2], [20190322, 4, 5]],
    n=6))

# Test cases for Week 2, July

# Test cases for Week 3, July

# Test cases for Week 4, July

# Test cases for Week 5, July
