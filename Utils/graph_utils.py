class UnionFind:
    """
    Implements the Union-Find data structure with path compression and union by rank optimizations.
    """

    def __init__(self, size: int, offset: int = 0):
        """Initializes the Union-Find data structure with the given number of elements and optional offset."""
        self.root = [i for i in range(size + offset)]
        self.rank = [1] * (size + offset)
        self.num_components = size

    def find(self, x: int) -> int:
        """Finds the root of the set containing x, applying path compression for efficiency."""
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

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
            self.root[root_x] = root_y
        else:
            self.root[root_y] = root_x
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


class UnionFindWithLogs:

    def __init__(self, size: int, offset: int = 0):
        self.root = [i for i in range(size + offset)]
        self.rank = [1] * (size + offset)
        self.num_components = size
        print(f"\n--- Initializing UnionFind ---")
        print(f"\tSize: {size}")
        print(f"\tOffset: {offset}")
        print(f"\tInitial root array: {self.root}")
        print(f"\tInitial rank array: {self.rank}")
        print(f"\tInitial number of components: {self.num_components}")

    def find(self, x: int) -> int:
        print(f"\n\t--- Finding root for element {x} ---")
        path = [x]
        while x != self.root[x]:
            x = self.root[x]
            path.append(x)
        root = x
        print(f"\t\tPath to root: {' -> '.join(map(str, path))}")

        # Path compression
        for node in path[:-1]:
            self.root[node] = root
        print(f"\t\tAfter path compression, root array: {self.root}")

        return root

    def union(self, x: int, y: int) -> bool:
        print(f"\n\t--- Uniting sets containing {x} and {y} ---")
        root_x = self.find(x)
        root_y = self.find(y)
        print(f"\t\tRoot of {x}: {root_x}")
        print(f"\t\tRoot of {y}: {root_y}")

        if root_x == root_y:
            print(f"\t\t{x} and {y} are already in the same set. No union performed.")
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.root[root_x] = root_y
            print(f"\t\tAttaching tree rooted at {root_x} to {root_y}")
        else:
            self.root[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
                print(f"\t\tAttaching tree rooted at {root_y} to {root_x} and incrementing rank of {root_x}")
            else:
                print(f"\t\tAttaching tree rooted at {root_y} to {root_x}")

        self.num_components -= 1
        print(f"\t\tUpdated root array: {self.root}")
        print(f"\t\tUpdated rank array: {self.rank}")
        print(f"\t\tNumber of components reduced to: {self.num_components}")
        return True

    def is_connected(self, x: int, y: int) -> bool:
        print(f"\n\t--- Checking if {x} and {y} are connected ---")
        result = self.find(x) == self.find(y)
        print(f"\t\tResult: {result}")
        return result

    def get_components(self) -> int:
        print(f"\n\t--- Getting number of components ---")
        print(f"\t\tCurrent number of components: {self.num_components}")
        return self.num_components

    def is_single_component(self) -> bool:
        print(f"\n\t--- Checking if all elements are in a single component ---")
        result = self.num_components == 1
        print(f"\t\tResult: {result}")
        return result
