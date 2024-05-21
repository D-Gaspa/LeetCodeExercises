from typing import Optional


# Daily Question (01/08/2024)

# 938. Range Sum of BST
# Easy

# Given the root node of a binary search tree and two integers low and high, return the sum of values of all nodes
# with a value in the inclusive range [low, high].


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def rangeSumBST(root: Optional[TreeNode], low: int, high: int) -> int:
    if root is None:
        return 0

    # If the root value is less than the lower bound, then we only need to search the right subtree
    if root.val < low:
        return rangeSumBST(root.right, low, high)
    # If the root value is greater than the upper bound, then we only need to search the left subtree
    elif root.val > high:
        return rangeSumBST(root.left, low, high)
    # If the root value is within the range, then we need to search both subtrees
    else:
        return root.val + rangeSumBST(root.left, low, high) + rangeSumBST(root.right, low, high)

# Daily Question (01/09/2024)

# 872. Leaf-Similar Trees
# Easy

# Consider all the leaves of a binary tree, from left to right order; the values of those leaves form a leaf value
# sequence. Two binary trees are considered leaf-similar if their leaf value sequence is the same.
# Return true if and only if the two given trees with head nodes root1 and root2 are leaf-similar.


def leafSimilar(root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
    # DFS to get the leaf values
    def dfs(node: TreeNode, leaves: list) -> list:
        if node is None:
            return leaves

        # If the node is a leaf, append its value to the 'leaves' list
        if node.left is None and node.right is None:
            leaves.append(node.val)

        # Recursively call dfs on the left and right children
        dfs(node.left, leaves)
        dfs(node.right, leaves)

        return leaves

    # Compare the leaf values of the two trees
    return dfs(root1, []) == dfs(root2, [])

# Daily Question (01/10/2024)

# 2385. Amount of Time for Binary Tree to be Infected

# You are given the root of a binary tree with unique values, and an integer start. At minute 0, an infection starts
# from the node with value start.
# Each minute, a node becomes infected if:
# 1. The node is currently uninfected.
# 2. The node is adjacent to an infected node.
# Return the number of minutes needed for the entire tree to be infected.


class Solution:
    def __init__(self):
        self.max_distance = 0

    def amountOfTime(self, root, start):
        self.traverse(root, start)
        return self.max_distance

    def traverse(self, root, start):
        depth = 0
        if root is None:
            return depth

        left_depth = self.traverse(root.left, start)
        right_depth = self.traverse(root.right, start)

        if root.val == start:
            self.max_distance = max(left_depth, right_depth)
            depth = -1
        elif left_depth >= 0 and right_depth >= 0:
            depth = max(left_depth, right_depth) + 1
        else:
            distance = abs(left_depth) + abs(right_depth)
            self.max_distance = max(self.max_distance, distance)
            depth = min(left_depth, right_depth) - 1

        return depth

# Daily Question (01/11/2024)

# Daily Question (01/12/2024)

# Daily Question (01/13/2024)

# Daily Question (01/14/2024)
