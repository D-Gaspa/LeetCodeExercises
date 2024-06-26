# June 2024, Week 5: June 24th - June 30th
from collections import deque
from typing import List

from tabulate import tabulate


# <------------------------------------------------- June 24th, 2024 ------------------------------------------------->
# 995. Minimum Number of K Consecutive Bit Flips

# Given a binary array `nums` and an integer `k`, return the minimum number of k-bit flips required so that there is no
# `0` in the array; if it is not possible, return `-1.
# A k-bit flip is choosing a contiguous subarray of length `k` from `nums` and simultaneously changing every `0` in the
# subarray to `1`, and every `1` in the subarray to `0`.


def minKBitFlips1(nums: List[int], k: int) -> int:
    """
    Determines the minimum number of k-bit flips required to convert all elements in `nums` to 1.

    This function uses a sliding window approach with a deque to efficiently track the flips.
    It maintains a 'current_flipped_state' to represent the cumulative effect of flips on the
    current element, avoiding the need to actually modify the input array.
    The algorithm iterates through the array once, deciding whether to flip at each position based on the
    current state and the original value.
    This approach allows for efficient handling of overlapping flips without the need
    to recalculate previous operations.

    The time complexity of this solution is O(n), where `n` is the length of `nums`, because it
    processes each element once with constant-time operations.
    The space complexity is O(k)
    due to the deque storing at most `k` elements to track the sliding window of flips.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tk = {k}")

    print("\n--- Initialization ---")
    if k == 1:
        result = nums.count(0)
        print(f"\tOptimization for k=1: returning nums.count(0) = {result}")
        return result

    flip_window_deque = deque()
    current_flipped_state = 0
    total_flips = 0
    print(f"\tflip_window_deque = {flip_window_deque}")
    print(f"\tcurrent_flipped_state = {current_flipped_state}")
    print(f"\ttotal_flips = {total_flips}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for index, num in enumerate(nums):
        print(f"\n--- Element {index + 1}/{len(nums)} ---")
        print(f"\tCurrent element: nums[{index}] = {num}")
        print(f"\tCurrent state: current_flipped_state = {current_flipped_state}, total_flips = {total_flips}")
        print(f"\tflip_window_deque = {flip_window_deque}")

        if index >= k:
            print(f"\tWindow size reached (index {index} >= k {k})")
            removed_flip = flip_window_deque.popleft()
            print(f"\t\tRemoving oldest flip from window: {removed_flip}")
            current_flipped_state ^= removed_flip
            print(f"\t\tUpdated current_flipped_state = {current_flipped_state}")

        print(f"\tDecision Point: Should we flip the current element?")
        if current_flipped_state == num:
            print(f"\t\tCondition true: current_flipped_state ({current_flipped_state}) == num ({num})")
            if index + k > len(nums):
                print(f"\t\t\tNot enough elements left for a flip (index {index} + k {k} > len(nums) {len(nums)})")
                print("\t\t\tReturning -1")
                return -1
            flip_window_deque.append(1)
            print("\t\t\tAdded flip to window: flip_window_deque.append(1)")
            current_flipped_state ^= 1
            print(f"\t\t\tUpdated current_flipped_state = {current_flipped_state}")
            total_flips += 1
            print(f"\t\t\tIncremented total_flips to {total_flips}")
        else:
            print(f"\t\tCondition false: current_flipped_state ({current_flipped_state}) != num ({num})")
            flip_window_deque.append(0)
            print("\t\t\tAdded no-flip to window: flip_window_deque.append(0)")

        iteration_data.append([index, num, current_flipped_state, total_flips, list(flip_window_deque)])

    print("\n--- Iteration Summary ---")
    headers = ["Index", "Element", "Flipped State", "Total Flips", "Flip Window"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Result: total_flips = {total_flips}")
    return total_flips


def minKBitFlips2(nums: List[int], k: int) -> int:
    """
    Computes the minimum number of k-bit flips needed to convert all elements in nums to 1.

    This function uses a clever in-place marking technique to track flips efficiently.
    It uses the value 2 to mark the start of a flip in the original array, allowing it to
    implicitly store flip information without additional data structures.
    The 'active_flips' variable keeps track of the number of active flips affecting the current element,
    enabling quick decisions on whether to flip.
    This approach combines the benefits of in-place modification with efficient flip tracking.

    The time complexity of this solution is O(n), where n is the length of nums, as it
    processes each element once with constant-time operations.
    The space complexity is O(1)
    since it modifies the input array in-place and uses only a constant amount of extra space.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tk = {k}")

    print("\n--- Initialization ---")
    if k == 1:
        result = nums.count(0)
        print(f"\tOptimization for k=1: returning nums.count(0) = {result}")
        return result

    n = len(nums)
    active_flips = 0
    total_flips = 0
    print(f"\tn = {n}")
    print(f"\tactive_flips = {active_flips}")
    print(f"\ttotal_flips = {total_flips}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for index in range(n):
        print(f"\n--- Element {index + 1}/{n} ---")
        print(f"\tCurrent element: nums[{index}] = {nums[index]}")
        print(f"\tCurrent state: active_flips = {active_flips}, total_flips = {total_flips}")

        print("\tChecking for ending flip:")
        if index >= k and nums[index - k] == 2:
            print(f"\t\tFlip ending at index {index}: nums[{index - k}] == 2")
            active_flips -= 1
            print(f"\t\tDecremented active_flips to {active_flips}")
        else:
            print(f"\t\tNo flip ending at index {index}")

        print("\tDecision Point: Should we flip the current element?")
        if (active_flips % 2) == nums[index]:
            print(f"\t\tCondition true: (active_flips % 2) ({active_flips % 2}) == nums[{index}] ({nums[index]})")
            if index + k > n:
                print(f"\t\t\tNot enough elements left for a flip (index {index} + k {k} > n {n})")
                print("\t\t\tReturning -1")
                return -1
            print(f"\t\t\tMarking flip start: Setting nums[{index}] = 2")
            nums[index] = 2
            active_flips += 1
            total_flips += 1
            print(f"\t\t\tIncremented active_flips to {active_flips}")
            print(f"\t\t\tIncremented total_flips to {total_flips}")
        else:
            print(f"\t\tCondition false: (active_flips % 2) ({active_flips % 2}) != nums[{index}] ({nums[index]})")
            print("\t\t\tNo flip needed at this index")

        iteration_data.append([index, nums[index], active_flips, total_flips])

    print("\n--- Iteration Summary ---")
    headers = ["Index", "Element Value", "Active Flips", "Total Flips"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Result: total_flips = {total_flips}")
    return total_flips


# <------------------------------------------------- June 25th, 2024 ------------------------------------------------->
# 1038. Binary Search Tree to Greater Sum Tree

# Given the `root` of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST
# is changed to the original key plus the sum of all keys greater than the original key in BST.


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def bstToGst1(root: TreeNode) -> TreeNode:
    """
    Converts a Binary Search Tree (BST) to a Greater Sum Tree (GST).

    This function performs an in-order traversal of the BST in reverse order (right-root-left),
    updating each node's value to be the sum of its original value plus all greater values in the tree.
    The algorithm leverages the BST property where all right subtree values are greater than the current node,
    and all left subtree values are smaller.
    By traversing right first, we accumulate the sum of greater values,
    which is then used to update the current node and propagated to the left subtree.

    The time complexity is O(n), where `n` is the number of nodes in the tree, as each node is visited exactly once.
    The space complexity is O(h), where `h` is the height of the tree, due to the recursive call stack.
    In the worst case of an unbalanced tree, this could be O(n) (skewed tree), but for a balanced BST,
    it would be O(log n).
    """
    cumulative_sum = 0

    def update_node_values(node: TreeNode) -> None:
        nonlocal cumulative_sum
        if node.right:
            update_node_values(node.right)
        cumulative_sum += node.val
        node.val = cumulative_sum
        if node.left:
            update_node_values(node.left)

    update_node_values(root)
    return root


def bstToGst2(root: TreeNode) -> TreeNode:
    """
    Converts a Binary Search Tree (BST) to a Greater Sum Tree (GST).

    This function performs an in-order traversal of the BST in reverse order (right-root-left) using a stack,
    updating each node's value to be the sum of its original value plus all greater values in the tree.
    The algorithm uses an explicit stack to simulate the recursive process, pushing all right nodes onto
    the stack first, then processing the current node, and finally pushing left nodes. This approach
    maintains the property of visiting nodes in descending order of value.

    The time complexity is O(n), where n is the number of nodes in the tree, as each node is visited exactly once.
    The space complexity is O(h), where h is the height of the tree, due to the stack used for traversal.
    In the worst case of an unbalanced tree, this could be O(n) (skewed tree), but for a balanced BST,
    it would be O(log n).
    """

    def push_right_nodes(node):
        """Helper function to push all right nodes onto the stack."""
        while node:
            stack.append(node)
            node = node.right

    stack = []
    current_node = root
    cumulative_sum = 0

    push_right_nodes(current_node)

    while stack:
        current_node = stack.pop()
        cumulative_sum += current_node.val
        current_node.val = cumulative_sum
        push_right_nodes(current_node.left)

    return root


def bstToGst3(root: TreeNode) -> TreeNode:
    """
    Converts a Binary Search Tree (BST) to a Greater Sum Tree (GST).

    This function performs a reverse in-order traversal (right-root-left) of the BST using Morris Traversal,
    updating each node's value to be the sum of its original value plus all greater values in the tree.
    The algorithm uses a threaded binary tree approach, temporarily modifying the tree structure to navigate
    without recursion or an explicit stack.

    The Morris Traversal creates temporary links from the successor node (rightmost node of the left subtree)
    to the current node, allowing backtracking without using a stack.
    These temporary links are removed after use, restoring the original tree structure.

    The time complexity is O(n), where `n` is the number of nodes in the tree.
    Although each node may be visited up to
    three times (to create the link, to process the node, and to remove the link),
    this still results in linear time complexity.
    The space complexity is O(1) as it uses only a constant amount of extra space, regardless of the tree size.
    """

    def find_successor(current):
        """Helper function to find the successor node (leftmost node of the right subtree)."""
        successor = current.right
        while successor.left and successor.left is not current:
            successor = successor.left
        return successor

    cumulative_sum = 0
    current = root
    while current:
        if not current.right:
            # No right child: process the current node and move to the left
            cumulative_sum += current.val
            current.val = cumulative_sum
            current = current.left
        else:
            successor = find_successor(current)
            if not successor.left:
                # First time visiting: create a temporary link and move right
                successor.left = current
                current = current.right
            else:
                # Second time visiting: remove temp link, process node, and move left
                successor.left = None
                cumulative_sum += current.val
                current.val = cumulative_sum
                current = current.left

    return root


# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for June 24th, 2024
# Expected output: 3
# minKBitFlips1(nums=[0, 0, 0, 1, 0, 1, 1, 0], k=3)
# minKBitFlips2(nums=[0, 0, 0, 1, 0, 1, 1, 0], k=3)

# Test cases for June 25th, 2024
# Expected output:
# TreeNode(30, TreeNode(36, TreeNode(36), TreeNode(35, right=TreeNode(33))),
#              TreeNode(21, TreeNode(26), TreeNode(15, right=TreeNode(8)))))
bstToGst1(TreeNode(4, TreeNode(1, TreeNode(), TreeNode(2, right=TreeNode(3))),
                   TreeNode(6, TreeNode(5), TreeNode(7, right=TreeNode(8)))))
bstToGst2(TreeNode(4, TreeNode(1, TreeNode(), TreeNode(2, right=TreeNode(3))),
                   TreeNode(6, TreeNode(5), TreeNode(7, right=TreeNode(8)))))
bstToGst3(TreeNode(4, TreeNode(1, TreeNode(), TreeNode(2, right=TreeNode(3))),
                   TreeNode(6, TreeNode(5), TreeNode(7, right=TreeNode(8)))))

# Test cases for June 26th, 2024

# Test cases for June 27th, 2024

# Test cases for June 28th, 2024

# Test cases for June 29th, 2024

# Test cases for June 30th, 2024
