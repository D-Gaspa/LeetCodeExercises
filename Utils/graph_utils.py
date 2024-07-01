class UnionFind:
    """
    Implements the Union-Find data structure with path compression and union by rank optimizations.
    """

    def __init__(self, size: int, offset: int = 0):
        """Initializes the Union-Find data structure with the given number of elements and optional offset."""
        self.parent = [i for i in range(size + offset)]
        self.rank = [1] * (size + offset)
        self.num_components = size

    def find(self, x: int) -> int:
        """Finds the root of the set containing x, applying path compression for efficiency."""
        if x == self.parent[x]:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Unites the sets containing x and y if they are not already in the same set.
        Returns True if a new union was performed, False otherwise.
        """
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
        self.num_components -= 1
        return True

    def is_connected(self, x: int, y: int) -> bool:
        """Checks if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def get_components(self) -> int:
        """Returns the current number of disjoint sets."""
        return self.num_components

    def is_single_component(self) -> bool:
        """Checks if all elements are in a single set."""
        return self.num_components == 1
