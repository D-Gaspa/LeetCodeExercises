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
    print("\n--- Input Parameters ---")
    print(f"\troot = {BinaryTreeVisualizer.visualize(root, 'root_tree')}")
    print(f"\tstart_value = {start_value}, dest_value = {dest_value}")

    def findCommonAncestor(node: BinaryTreeNode) -> Optional[BinaryTreeNode]:
        print(f"\n--- Finding Common Ancestor for node {node.val if node else 'None'} ---")
        if node is None or node.val == start_value or node.val == dest_value:
            if node is None:
                print("\tNode is None")
            else:
                print(f"\tNode {node.val} matches start/dest value")
            print(f"\tReturning node {node.val if node else 'None'}")
            return node

        print(f"\tRecursing on left subtree of node {node.val} -> {node.left.val if node.left else 'None'}")
        left_result = findCommonAncestor(node.left)
        print(f"\tRecursing on right subtree of node {node.val} -> {node.right.val if node.right else 'None'}")
        right_result = findCommonAncestor(node.right)

        if left_result and right_result:
            print(f"\tFound LCA at node {node.val}")
            return node
        return left_result or right_result

    print("\n--- Finding Lowest Common Ancestor ---")
    lowest_common_ancestor = findCommonAncestor(root)
    print(f"\tLowest Common Ancestor: {lowest_common_ancestor.val}")

    print("\n--- Initialization ---")
    path_to_start = []
    path_to_dest = []
    print(f"\tpath_to_start = {path_to_start}")
    print(f"\tpath_to_dest = {path_to_dest}")

    print("\n--- Finding Path to Start ---")
    findPath(lowest_common_ancestor, start_value, path_to_start)
    print(f"\tPath to start: {path_to_start}")

    print("\n--- Finding Path to Destination ---")
    findPath(lowest_common_ancestor, dest_value, path_to_dest)
    print(f"\tPath to destination: {path_to_dest}")

    print("\n--- Constructing Final Path ---")
    initial_path = 'U' * len(path_to_start)
    final_path = ''.join(path_to_dest)
    result = initial_path + final_path

    print("\n--- Path Construction Summary ---")
    headers = ["Component", "Value", "Length"]
    summary_data = [
        ["Initial Path (U's)", initial_path, len(initial_path)],
        ["Final Path", final_path, len(final_path)],
        ["Result", result, len(result)]
    ]
    print(tabulate(summary_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Result: {result}")

    return result


def findPath(current_node: BinaryTreeNode, target_value: int, path: List[str]) -> bool:
    print(f"\n--- Finding Path from node {current_node.val} to {target_value} ---")
    if current_node.val == target_value:
        print(f"\tFound target node {target_value}")
        return True

    if current_node.left:
        print(f"\tGoing left from node {current_node.val} -> {current_node.left.val}")
        path.append('L')
        if findPath(current_node.left, target_value, path):
            print(f"\tFound target {target_value} in left subtree of node {current_node.val}")
            return True
        print(f"\tBacktracking from left of node {current_node.val}")
        path.pop()

    if current_node.right:
        print(f"\tGoing right from node {current_node.val} -> {current_node.right.val}")
        path.append('R')
        if findPath(current_node.right, target_value, path):
            print(f"\tFound target {target_value} in right subtree of node {current_node.val}")
            return True
        print(f"\tBacktracking from right of node {current_node.val}")
        path.pop()

    print(f"\tTarget {target_value} not found in subtree of node {current_node.val}")
    return False


def getDirections2(root: Optional[BinaryTreeNode], start_value: int, dest_value: int) -> str:
    print("\n--- Input Parameters ---")
    print(f"\troot = {BinaryTreeVisualizer.visualize(root, 'root_tree')}")
    print(f"\tstart_value = {start_value}, dest_value = {dest_value}")

    print("\n--- Initialization ---")
    path_to_start = []
    path_to_dest = []
    print(f"\tpath_to_start "
          f"= {path_to_start}")
    print(f"\tpath_to_dest = {path_to_dest}")

    print("\n--- Finding Path to Start ---")
    findPath(root, start_value, path_to_start)
    print(f"\tPath to start: {path_to_start}")

    print("\n--- Finding Path to Destination ---")
    findPath(root, dest_value, path_to_dest)
    print(f"\tPath to destination: {path_to_dest}")

    print("\n--- Finding Common Prefix ---")
    common_prefix_length = 0
    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Start Path", "Dest Path", "Common?"]
    iteration_data = []

    while (common_prefix_length < len(path_to_start) and
           common_prefix_length < len(path_to_dest) and
           path_to_start[common_prefix_length] == path_to_dest[common_prefix_length]):
        is_common = path_to_start[common_prefix_length] == path_to_dest[common_prefix_length]
        iteration_data.append([
            common_prefix_length + 1,
            path_to_start[common_prefix_length] if common_prefix_length < len(path_to_start) else "N/A",
            path_to_dest[common_prefix_length] if common_prefix_length < len(path_to_dest) else "N/A",
            "Yes" if is_common else "No"
        ])
        common_prefix_length += 1

    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))
    print(f"\tCommon Prefix Length: {common_prefix_length}")

    print("\n--- Constructing Final Path ---")
    initial_path = 'U' * (len(path_to_start) - common_prefix_length)
    final_path = ''.join(path_to_dest[common_prefix_length:])
    result = initial_path + final_path

    print("\n--- Path Construction Summary ---")
    headers = ["Component", "Value", "Length"]
    summary_data = [
        ["Path to Start", ''.join(path_to_start), len(path_to_start)],
        ["Path to Destination", ''.join(path_to_dest), len(path_to_dest)],
        ["Common Prefix", ''.join(path_to_start[:common_prefix_length]), common_prefix_length],
        ["Initial Path (U's)", initial_path, len(initial_path)],
        ["Final Path", final_path, len(final_path)],
        ["Result", result, len(result)]
    ]
    print(tabulate(summary_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Result: {result}")

    return result


# <------------------------------------------------- July 17th, 2024 ------------------------------------------------->
# 1110. Delete Nodes And Return Forest

# Given the `root` of a binary tree, each node in the tree has a distinct value.
# After deleting all nodes with a value in `to_delete`, we are left with a forest (a disjoint union of trees).
# Return the roots of the trees in the remaining forest.
# You may return the result in any order.


def delNodes1(root: Optional[BinaryTreeNode], to_delete: List[int]) -> List[BinaryTreeNode]:
    print("\n--- Input Parameters ---")
    print(f"\troot = {root}")
    print(f"\tto_delete = {to_delete}")

    print("\n--- Initialization ---")
    forest_roots = []
    nodes_to_delete = set(to_delete)
    print(f"\tforest_roots = {forest_roots}")
    print(f"\tnodes_to_delete = {nodes_to_delete}")

    iteration_data = []
    iteration_count = 0

    def processNodesAndBuildForest(current_node: BinaryTreeNode) -> Optional[BinaryTreeNode]:
        nonlocal iteration_count
        iteration_count += 1
        print(f"\n--- Iteration {iteration_count} ---")
        print(f"\tCurrent node: {current_node.val if current_node else None}")

        if not current_node:
            print("\tNode is None, returning None")
            return None

        print(f"\tProcessing left child of node {current_node.val} -> "
              f"{current_node.left.val if current_node.left else "None"}")
        current_node.left = processNodesAndBuildForest(current_node.left)
        print(f"\tFunction returned, updated left child of node {current_node.val} -> "
              f"{current_node.left.val if current_node.left else "None"}")

        print(f"\tProcessing right child of node {current_node.val} -> "
              f"{current_node.right.val if current_node.right else "None"}")
        current_node.right = processNodesAndBuildForest(current_node.right)
        print(f"\tFunction returned, updated right child of node {current_node.val} -> "
              f"{current_node.right.val if current_node.right else "None"}")

        print(f"\tChecking if node {current_node.val} should be deleted")
        if current_node.val in nodes_to_delete:
            print(f"\tNode {current_node.val} will be deleted")
            if current_node.left:
                print(f"\tAdding left child {current_node.left.val} to forest_roots")
                forest_roots.append(current_node.left)
            if current_node.right:
                print(f"\tAdding right child {current_node.right.val} to forest_roots")
                forest_roots.append(current_node.right)

            iteration_data.append([iteration_count, current_node.val, "Deleted",
                                   f"Left: {current_node.left.val if current_node.left else None}, "
                                   f"Right: {current_node.right.val if current_node.right else None}"])
            print("\tReturning None (node deleted)")
            return None
        else:
            iteration_data.append([iteration_count, current_node.val, "Kept",
                                   f"Left: {current_node.left.val if current_node.left else None}, "
                                   f"Right: {current_node.right.val if current_node.right else None}"])
            print(f"\tReturning node {current_node.val} (not deleted)")
            return current_node

    print("\n--- Starting Tree Traversal ---")
    root = processNodesAndBuildForest(root)

    print("\n--- Post-Traversal Processing ---")
    if root:
        print(f"\tAdding root {root.val} to forest_roots")
        forest_roots.append(root)
    else:
        print("\tRoot was deleted, not adding to forest_roots")

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Node Value", "Action", "Children"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal forest_roots: {[node.val for node in forest_roots]}")
    return forest_roots


def delNodes2(root: Optional[BinaryTreeNode], to_delete: List[int]) -> List[BinaryTreeNode]:
    print("\n--- Input Parameters ---")
    print(f"\troot = {root}")
    print(f"\tto_delete = {to_delete}")

    print("\n--- Initialization ---")
    nodes_to_delete = set(to_delete)
    forest_roots = []
    print(f"\tnodes_to_delete = {nodes_to_delete}")
    print(f"\tforest_roots = {forest_roots}")

    iteration_data = []
    iteration_count = 0

    def processNodesAndBuildForest(current_node: Optional[BinaryTreeNode], is_root: bool) -> Optional[BinaryTreeNode]:
        nonlocal iteration_count
        iteration_count += 1
        print(f"\n--- Iteration {iteration_count} ---")
        print(f"\tCurrent node: {current_node.val if current_node else None}")
        print(f"\tIs root: {is_root}")

        if not current_node:
            print("\tNode is None, returning None")
            return None

        is_deleted = current_node.val in nodes_to_delete
        print(f"\tCurrent node {current_node.val} is in to_delete: {is_deleted}")

        if is_root and not is_deleted:
            print(f"\tAdding node {current_node.val} to forest_roots (root node)")
            forest_roots.append(current_node)

        print(f"\tProcessing left child of node {current_node.val} -> "
              f"{current_node.left.val if current_node.left else None}")
        current_node.left = processNodesAndBuildForest(current_node.left, is_deleted)
        print(f"\tProcessing right child of node {current_node.val} -> "
              f"{current_node.right.val if current_node.right else None}")
        current_node.right = processNodesAndBuildForest(current_node.right, is_deleted)

        if is_deleted:
            print(f"\tNode {current_node.val} will be deleted")
            iteration_data.append([iteration_count, current_node.val, "Deleted", is_root,
                                   f"Left: {current_node.left.val if current_node.left else None}, "
                                   f"Right: {current_node.right.val if current_node.right else None}"])
            print("\tReturning None (node deleted)")
            return None
        else:
            iteration_data.append([iteration_count, current_node.val, "Kept", is_root,
                                   f"Left: {current_node.left.val if current_node.left else None}, "
                                   f"Right: {current_node.right.val if current_node.right else None}"])
            print(f"\tReturning node {current_node.val} (not deleted)")
            return current_node

    print("\n--- Starting Tree Traversal ---")
    processNodesAndBuildForest(root, True)

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Node Value", "Action", "Was Root", "Children"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal forest_roots: {[node.val for node in forest_roots]}")
    return forest_roots


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
# getDirections1(root=root, start_value=3, dest_value=6)
# getDirections2(root=root, start_value=3, dest_value=6)

# Test cases for July 17th, 2024
# Expected output:
# [BinaryTreeNode(val=3, left=BinaryTreeNode(val=4)), BinaryTreeNode(val=6), BinaryTreeNode(val=7)]
root = BinaryTreeNode(val=1)
root.left = BinaryTreeNode(val=2)
root.right = BinaryTreeNode(val=3)
root.right.left = BinaryTreeNode(val=4)
root.right.right = BinaryTreeNode(val=5)
root.right.right.left = BinaryTreeNode(val=6)
root.right.right.right = BinaryTreeNode(val=7)
# delNodes1(root=root, to_delete=[2, 1, 5])
delNodes2(root=root, to_delete=[2, 1, 5])

# Test cases for July 18th, 2024

# Test cases for July 19th, 2024

# Test cases for July 20th, 2024

# Test cases for July 21st, 2024
