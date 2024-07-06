# July, 2024
# <-------------------------------------------------- Week 1, July -------------------------------------------------->
# 1101. The Earliest Moment When Everyone Becomes Friends

from tabulate import tabulate

from Utils.graph_utils import UnionFindWithLogs


# Given `n` people in a social group labeled from `0` to `n - 1` and an array `logs` where
# `logs[i] = [timestamp_i, x_i, y_i]` indicates that `x_i` and `y_i` become friends at time `timestamp_i`,
# return the earliest time when every person is acquainted with every other person, or `-1` if it is not possible.
# Friendship is symmetric and transitive.


def earliestAcq1(logs, n):
    print("\n--- Input Parameters ---")
    print(f"\tlogs = {logs}")
    print(f"\tn = {n}")

    print("\n--- Initialization ---")
    if len(logs) < n - 1:
        print(f"\tNot enough friendships: {len(logs)} < {n - 1}")
        return -1

    print("\n--- Sorting Logs ---")
    logs.sort(key=lambda x: x[0])
    print("\tSorted logs:")
    print(tabulate(logs, headers=["Timestamp", "Person A", "Person B"], tablefmt="fancy_grid"))

    print("\n--- Initializing Union-Find ---")
    friend_network_uf = UnionFindWithLogs(n)
    print(f"\tInitialized UnionFind with {n} elements")

    print("\n--- Processing Friendships ---")
    iteration_data = []
    for i, (timestamp, person_a, person_b) in enumerate(logs):
        print(f"\n--- Friendship {i + 1}/{len(logs)} ---")
        print(f"\tTimestamp: {timestamp}")
        print(f"\tPerson A: {person_a}")
        print(f"\tPerson B: {person_b}")

        print("\tPerforming Union Operation:")
        union_result = friend_network_uf.union(person_a, person_b)
        print(f"\t\tUnion result: {union_result}")

        print("\tChecking for Single Component:")
        is_single = friend_network_uf.is_single_component()
        print(f"\t\tIs single component: {is_single}")

        iteration_data.append([i + 1, timestamp, person_a, person_b, union_result, is_single])

        if is_single:
            print(f"\n--- All People Connected at Timestamp {timestamp} ---")
            print("\n--- Iteration Summary ---")
            headers = ["Iteration", "Timestamp", "Person A", "Person B", "Union Result", "Is Single Component"]
            print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))
            return timestamp

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Timestamp", "Person A", "Person B", "Union Result", "Is Single Component"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("\tNot all people could be connected")
    print("\tFinal Result: -1")
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
earliestAcq1(logs=[[0, 2, 0], [1, 0, 1], [3, 0, 3], [4, 1, 2], [7, 3, 1]], n=4)

# Test cases for Week 2, July

# Test cases for Week 3, July

# Test cases for Week 4, July

# Test cases for Week 5, July
