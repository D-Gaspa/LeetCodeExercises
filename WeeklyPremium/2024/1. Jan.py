# Weekly Premium for Jan 2024

# 1. 1066. Campus Bikes II

# Medium

# On a campus represented as a 2D grid, there are n workers and m bikes, with n <= m. Each worker and bike is a 2D
# coordinate on this grid. We assign one unique bike to each worker so that the sum of the Manhattan distances
# between each worker and their assigned bike is minimized. Return the minimum possible sum of Manhattan distances
# between each worker and their assigned bike. The Manhattan distance between two points p1 and p2 is:
# Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|.


def assignBikes(workers, bikes):
    def manhattan(worker, bike):
        # Calculate the Manhattan distance between a worker and a bike
        return abs(worker[0] - bike[0]) + abs(worker[1] - bike[1])

    def backtrack(worker_index, bike_mask):
        # Base case: all workers have been assigned a bike
        if worker_index == len(workers):
            return 0
        # If the current worker and bike combination has been seen before, return the distance
        if (worker_index, bike_mask) in min_distance_memo:
            return min_distance_memo[(worker_index, bike_mask)]

        # Initialize the minimum distance to infinity
        min_distance = float('inf')
        # Iterate over each bike
        for bike_index, bike in enumerate(bikes):
            # If the bike has not been assigned to a worker
            if not (bike_mask & (1 << bike_index)):
                # Calculate the distance between the current worker and bike and backtrack to the next worker
                distance = manhattan(workers[worker_index], bike) + \
                           backtrack(worker_index + 1, bike_mask | (1 << bike_index))
                # Update the minimum distance
                min_distance = min(min_distance, distance)

        # Memoize the minimum distance
        min_distance_memo[(worker_index, bike_mask)] = min_distance
        return min_distance

    # Initialize the memoization dictionary
    min_distance_memo = {}
    return backtrack(0, 0)


# Tests

print(assignBikes([[0, 0], [2, 1]], [[1, 2], [3, 3]]))

print(assignBikes([[0, 0], [1, 1], [2, 0]], [[1, 0], [2, 2], [2, 1]]))
