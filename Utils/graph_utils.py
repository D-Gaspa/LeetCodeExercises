from graphviz import Graph


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

    def __init__(self, size: int, offset: int = 0, images_only: bool = False):
        self.root = [i for i in range(size + offset)]
        self.rank = [1] * (size + offset)
        self.num_components = size
        self.images_only = images_only
        if not images_only:
            print(f"\n--- Initializing UnionFind ---")
            print(f"\tSize: {size}")
            print(f"\tOffset: {offset}")
            print(f"\tInitial root array: {self.root}")
            print(f"\tInitial rank array: {self.rank}")
            print(f"\tInitial number of components: {self.num_components}")
        print(f"\nInitialized UnionFind: {UnionFindVisualizer.visualize(self, 'union-find-initial')}")

    def find(self, x: int) -> int:
        if not self.images_only:
            print(f"\n\t--- Finding root for element {x} ---")
        path = [x]
        while x != self.root[x]:
            x = self.root[x]
            path.append(x)
        root = x
        if not self.images_only:
            print(f"\t\tPath to root: {' -> '.join(map(str, path))}")

        # Path compression
        for node in path[:-1]:
            self.root[node] = root
        if not self.images_only:
            print(f"\t\tAfter path compression, root array: {self.root}")

        return root

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if not self.images_only:
            print(f"\n\t--- Uniting sets containing {x} and {y} ---")
            print(f"\t\tRoot of {x}: {root_x}")
            print(f"\t\tRoot of {y}: {root_y}")

        if root_x == root_y:
            if not self.images_only:
                print(f"\t\t{x} and {y} are already in the same set. No union performed.")
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.root[root_x] = root_y
            if not self.images_only:
                print(f"\t\tAttaching tree rooted at {root_x} to {root_y}")
        else:
            self.root[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
                if not self.images_only:
                    print(f"\t\tAttaching tree rooted at {root_y} to {root_x} and incrementing rank of {root_x}")
            else:
                if not self.images_only:
                    print(f"\t\tAttaching tree rooted at {root_y} to {root_x}")

        self.num_components -= 1
        if not self.images_only:
            print(f"\t\tUpdated root array: {self.root}")
            print(f"\t\tUpdated rank array: {self.rank}")
            print(f"\t\tNumber of components reduced to: {self.num_components}")
        print(f"Union performed: {UnionFindVisualizer.visualize(self, f'union-find-after-union-{x}-{y}')}")
        return True

    def is_connected(self, x: int, y: int) -> bool:
        result = self.find(x) == self.find(y)
        if not self.images_only:
            print(f"\n\t--- Checking if {x} and {y} are connected ---")
            print(f"\t\tResult: {result}")
        return result

    def get_components(self) -> int:
        if not self.images_only:
            print(f"\n\t--- Getting number of components ---")
            print(f"\t\tCurrent number of components: {self.num_components}")
        return self.num_components

    def is_single_component(self) -> bool:
        result = self.num_components == 1
        if not self.images_only:
            print(f"\n\t--- Checking if all elements are in a single component ---")
            print(f"\t\tResult: {result}")
        return result


class UnionFindVisualizer:
    @staticmethod
    def visualize(uf: UnionFind | UnionFindWithLogs, file_name: str) -> str:
        dot = Graph(comment='Union-Find')
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr('node', shape='circle', style='filled', color='lightblue', fontcolor='black')
        dot.attr('edge', color='black')

        # Create a subgraph for each set
        sets = {}
        for i in range(len(uf.root)):
            root = uf.find(i)
            if root not in sets:
                sets[root] = []
            sets[root].append(i)

        for root, elements in sets.items():
            with dot.subgraph(name=f'cluster_{root}') as c:
                c.attr(label=f'Set {root}', style='rounded', color='lightgrey')
                for element in elements:
                    c.node(str(element), str(element))
                    if element != root:
                        dot.edge(str(root), str(element), style='dashed')

        # Use the provided file name directly
        dot.render(file_name, format='png', cleanup=True)

        return file_name + '.png'
