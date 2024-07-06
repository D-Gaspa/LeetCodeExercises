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
    """
    Determines the earliest time when all people in a social group become acquainted.

    This function uses the Union-Find data structure to efficiently track friend connections
    and group formations. It sorts the logs by timestamp and processes each friendship event
    chronologically. The algorithm leverages the properties of the Union-Find structure to
    quickly check if all individuals have become part of a single connected component.

    The time complexity of this solution is O(m log m + n + m * α(n)), where 'm' is the number of
    logs, 'n' is the number of people and α(n) is the inverse Ackermann function.
    This is due to the initial sort operation (O(m log m)), the UnionFind initialization (O(n)) and for
    each log (m), the find and union operations (O(α(n))).
    The space complexity is O(m + n), accounting for the sorted logs and the UnionFind data structure.
    """
    if len(logs) < n - 1:
        return -1  # Not enough friendships to connect all people

    logs.sort(key=lambda x: x[0])  # Sort logs by timestamp
    friend_network_uf = UnionFind(n)

    for timestamp, person_a, person_b in logs:
        friend_network_uf.union(person_a, person_b)
        if friend_network_uf.is_single_component():
            return timestamp  # All people are connected

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
