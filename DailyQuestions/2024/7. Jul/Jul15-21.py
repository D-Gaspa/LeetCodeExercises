# Week 3: July 15th - July 21st, 2024
from typing import List, Optional

from tabulate import tabulate

from Utils.trees_utils import BinaryTreeNode, BinaryTreeVisualizer


# <------------------------------------------------- July 15th, 2024 ------------------------------------------------->
# 2196. Create Binary Tree From Descriptions

# You are given a 2D integer array `descriptions`, where each element `[parent_i, child_i, isLeft_i]` specifies that
# `parent_i` is the parent of `child_i` in a binary tree with unique values.
# If `isLeft_i == 1`, `child_i` is the left child; if `isLeft_i == 0`, `child_i` is the right child.
# Construct and return the root of the binary tree described by `descriptions`.
# The binary tree is guaranteed to be valid.


def createBinaryTree1(descriptions: List[List[int]]) -> Optional[BinaryTreeNode]:
    print("\n--- Input Parameters ---")
    print(f"\tdescriptions = {descriptions}")

    print("\n--- Initialization ---")
    node_map = {child: BinaryTreeNode(child) for _, child, _ in descriptions}
    print("\tInitialized node_map:")
    print(tabulate([[child, f"BinaryTreeNode({child})"] for child in node_map],
                   headers=["Child Value", "Node"], tablefmt="fancy_grid"))

    print("\n--- Main Loop ---")
    root = None
    iteration_data = []
    for i, (parent_value, child_value, is_left_child) in enumerate(descriptions, 1):
        print(f"\n--- Iteration {i}/{len(descriptions)} ---")
        print(f"\tCurrent description: parent={parent_value}, child={child_value}, is_left_child={is_left_child}")

        iteration_data.append([i, parent_value, child_value,
                               "Left" if is_left_child else "Right",
                               "Created" if parent_value not in node_map else "Existed"])

        print("\tChecking if parent node exists:")
        if parent_value not in node_map:
            print(f"\t\tParent node {parent_value} doesn't exist. Creating it.")
            root = node_map[parent_value] = BinaryTreeNode(parent_value)
            print(f"\t\tSet as root: {root.val}")
        else:
            print(f"\t\tParent node {parent_value} already exists.")

        print("\tAssigning child node to parent:")
        if is_left_child:
            print(f"\t\tAssigning {child_value} as left child of {parent_value}")
            node_map[parent_value].left = node_map[child_value]
        else:
            print(f"\t\tAssigning {child_value} as right child of {parent_value}")
            node_map[parent_value].right = node_map[child_value]

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Parent", "Child", "Child Position", "Parent Node Status"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tRoot node: {root.val if root else None}")

    # Visualize the final tree
    final_tree_image = BinaryTreeVisualizer.visualize(root, 'final_tree')
    print(f"\tFinal tree structure: {final_tree_image}")

    return root


# <------------------------------------------------- July 16th, 2024 ------------------------------------------------->
# 2096. Step-By-Step Directions From a Binary Tree Node to Another

# Given the root of a binary tree with `n` nodes uniquely valued from 1 to `n`, find the shortest path between two
# nodes: startValue (s) and destValue (t).
# Return the path as a string of uppercase letters 'L' (left child),
# 'R' (right child), and 'U' (parent), representing the step-by-step directions from `s` to `t`.

# Introductory Problem
# 236. Lowest Common Ancestor of a Binary Tree
# Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
# The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and
# `q` as descendants (where we allow a node to be a descendant of itself).


def getDirections1(root: Optional[BinaryTreeNode], start_value: int, dest_value: int) -> str:
    def findCommonAncestor(node: BinaryTreeNode) -> Optional[BinaryTreeNode]:
        # Base case: if node is None or is one of the target nodes
        if node is None or node.val == start_value or node.val == dest_value:
            return node

        # Recurse on left and right subtrees
        left_result = findCommonAncestor(node.left)
        right_result = findCommonAncestor(node.right)

        # If both subtrees return a node, this is the LCA
        if left_result and right_result:
            return node
        # Otherwise, return the non-None result (if any)
        return left_result or right_result

    lowest_common_ancestor = findCommonAncestor(root)
    path_to_start = []
    path_to_dest = []
    findPath(lowest_common_ancestor, start_value, path_to_start)
    findPath(lowest_common_ancestor, dest_value, path_to_dest)

    # Construct the final path: 'U's to go up to LCA, then path down to destination
    initial_path = 'U' * len(path_to_start)
    final_path = ''.join(path_to_dest)

    return initial_path + final_path


def findPath(current_node: BinaryTreeNode, target_value: int, path: List[str]) -> bool:
    # Recursively find the path to target node, appending directions to 'path'
    if current_node.val == target_value:
        return True
    if current_node.left:
        path.append('L')
        if findPath(current_node.left, target_value, path):
            return True
        path.pop()
    if current_node.right:
        path.append('R')
        if findPath(current_node.right, target_value, path):
            return True
        path.pop()
    return False


def getDirections2(root: Optional[BinaryTreeNode], start_value: int, dest_value: int) -> str:
    # Find paths from root to start and destination nodes
    path_to_start = []
    path_to_dest = []
    findPath(root, start_value, path_to_start)
    findPath(root, dest_value, path_to_dest)

    # Find the length of the common prefix in both paths
    common_prefix_length = 0
    while (common_prefix_length < len(path_to_start) and
           common_prefix_length < len(path_to_dest) and
           path_to_start[common_prefix_length] == path_to_dest[common_prefix_length]):
        common_prefix_length += 1

    # Construct the final path: 'U's to go up to LCA, then path down to destination
    initial_path = 'U' * (len(path_to_start) - common_prefix_length)
    final_path = ''.join(path_to_dest[common_prefix_length:])

    return initial_path + final_path

# <------------------------------------------------- July 17th, 2024 ------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <------------------------------------------------- July 18th, 2024 ------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <------------------------------------------------- July 19th, 2024 ------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <------------------------------------------------- July 20th, 2024 ------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <------------------------------------------------- July 21st, 2024 ------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for July 15th, 2024
# Expected output:
# root = BinaryTreeNode(60)
# root.left = BinaryTreeNode(52)
# root.left.left = BinaryTreeNode(45)
# root.left.right = BinaryTreeNode(58)
# root.left.left.left = BinaryTreeNode(41)
# root.left.left.left.left = BinaryTreeNode(39)
# root.left.left.left.right = BinaryTreeNode(43)
# createBinaryTree1(descriptions=[[52, 58, 0], [41, 39, 1], [52, 45, 1], [41, 43, 0], [45, 41, 1], [60, 52, 1]])

# Test cases for July 16th, 2024
# Expected output: "UURL"
# root = BinaryTreeNode(val=5)
# root.left = BinaryTreeNode(val=1)
# root.right = BinaryTreeNode(val=2)
# root.left.left = BinaryTreeNode(val=3)
# root.right.left = BinaryTreeNode(val=6)
# root.right.right = BinaryTreeNode(val=4)
#
# getDirections1(root=root, start_value=3, dest_value=6)

# Test cases for July 17th, 2024

# Test cases for July 18th, 2024

# Test cases for July 19th, 2024

# Test cases for July 20th, 2024

# Test cases for July 21st, 2024
