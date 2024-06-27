# June 2024, Week 5: June 24th - June 30th
import math
from collections import deque
from typing import List

from tabulate import tabulate

from Utils.trees_utils import TreeNode, TreeVisualizer


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


def bstToGst1(root: TreeNode) -> TreeNode:
    print("\n--- Input Parameters ---")
    print(f"\troot = {TreeVisualizer.visualize(root)}")

    print("\n--- Initialization ---")
    cumulative_sum = 0
    print(f"\tcumulative_sum = {cumulative_sum}")

    iteration_data = []

    def update_node_values(node: TreeNode) -> None:
        nonlocal cumulative_sum
        nonlocal iteration_data

        if node is None:
            return

        print(f"\n--- Processing Node: {node.val} ---")
        print(f"\tCurrent cumulative_sum: {cumulative_sum}")

        print("\n\t--- Traversing Right Subtree ---")
        if node.right:
            print(f"\t\tMoving to right child: {node.right.val}")
            update_node_values(node.right)
        else:
            print("\t\tNo right child")

        print("\n\t--- Updating Current Node ---")
        old_val = node.val
        cumulative_sum += node.val
        node.val = cumulative_sum
        print(f"\t\tUpdating node value:")
        print(f"\t\t\tOld value: {old_val}")
        print(f"\t\t\tAdding to cumulative_sum: {cumulative_sum - node.val} + {old_val} = {cumulative_sum}")
        print(f"\t\t\tNew value: {node.val}")

        iteration_data.append([old_val, cumulative_sum - old_val, cumulative_sum])

        print("\n\t--- Traversing Left Subtree ---")
        if node.left:
            print(f"\t\tMoving to left child: {node.left.val}")
            update_node_values(node.left)
        else:
            print("\t\tNo left child")

    print("\n--- Starting Tree Traversal ---")
    update_node_values(root)

    print("\n--- Iteration Summary ---")
    headers = ["Original Value", "Added to Sum", "New Value (Cumulative Sum)"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tModified Tree: {TreeVisualizer.visualize(root)}")
    return root


def bstToGst2(root: TreeNode) -> TreeNode:
    print("\n--- Input Parameters ---")
    print(f"\troot = {TreeVisualizer.visualize(root)}")

    print("\n--- Initialization ---")
    stack = []
    current_node = root
    cumulative_sum = 0
    print(f"\tstack = {stack}")
    print(f"\tcurrent_node = {current_node.val if current_node else None}")
    print(f"\tcumulative_sum = {cumulative_sum}")

    iteration_data = []

    def push_right_nodes(node: TreeNode) -> None:
        print("\n\t--- Pushing Right Nodes ---")
        while node:
            print(f"\t\tPushing node {node.val} onto stack")
            stack.append(node)
            node = node.right
            print(f"\t\tMoving to right child: {node.val if node else None}")
        print(f"\t\tStack after pushing: {[n.val for n in stack]}")

    print("\n--- Initial Stack Population ---")
    push_right_nodes(current_node)

    print("\n--- Main Traversal Loop ---")
    while stack:
        print(f"\n--- Processing Node ---")
        current_node = stack.pop()
        print(f"\tPopped node: {current_node.val}")
        print(f"\tCurrent stack: {[n.val for n in stack]}")

        old_val = current_node.val
        cumulative_sum += current_node.val
        current_node.val = cumulative_sum

        print("\t--- Node Update ---")
        print(f"\t\tOld value: {old_val}")
        print(f"\t\tNew cumulative_sum: {cumulative_sum}")
        print(f"\t\tUpdated node value: {current_node.val}")

        iteration_data.append([old_val, cumulative_sum - old_val, cumulative_sum])

        print("\t--- Handling Left Subtree ---")
        if current_node.left:
            print(f"\t\tPushing right nodes of left child ({current_node.left.val})")
            push_right_nodes(current_node.left)
        else:
            print("\t\tNo left child")

    print("\n--- Iteration Summary ---")
    headers = ["Original Value", "Added to Sum", "New Value (Cumulative Sum)"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tModified Tree: {TreeVisualizer.visualize(root)}")
    return root


def bstToGst3(root: TreeNode) -> TreeNode:
    print("\n--- Input Parameters ---")
    print(f"\troot = {TreeVisualizer.visualize(root)}")

    def find_successor(current_node: TreeNode) -> TreeNode:
        print("\n\t--- Finding Successor ---")
        successor = current_node.right
        print(f"\t\tStarting from right child: {successor.val if successor else None}")
        while successor.left and successor.left is not current_node:
            successor = successor.left
            print(f"\t\tMoving to left child: {successor.val}")
        print(f"\t\tSuccessor found: {successor.val}")
        return successor

    print("\n--- Initialization ---")
    cumulative_sum = 0
    current_node = root
    print(f"\tcumulative_sum = {cumulative_sum}")
    print(f"\tcurrent_node = {current_node.val if current_node else None}")

    iteration_data = []

    print("\n--- Main Traversal Loop ---")
    while current_node:
        print(f"\n--- Processing Node: {current_node.val} ---")

        if not current_node.right:
            print("\t--- No Right Child ---")
            cumulative_sum, current_node = update_node_and_move_left(cumulative_sum, current_node, iteration_data)
        else:
            successor = find_successor(current_node)
            if not successor.left:
                print("\t--- Creating Temporary Link ---")
                print(f"\t\tLinking successor {successor.val} to current node {current_node.val}")
                successor.left = current_node
                print(f"\t\tMoving to right child: {current_node.right.val}")
                current_node = current_node.right
            else:
                print("\t--- Removing Temporary Link ---")
                print(f"\t\tRemoving link from successor {successor.val} to current node {current_node.val}")
                successor.left = None
                cumulative_sum, current_node = update_node_and_move_left(cumulative_sum, current_node, iteration_data)

    print("\n--- Iteration Summary ---")
    headers = ["Original Value", "Added to Sum", "New Value (Cumulative Sum)"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tModified Tree: {TreeVisualizer.visualize(root)}")
    return root


def update_node_and_move_left(cumulative_sum, current_node, iteration_data):
    old_val = current_node.val
    cumulative_sum += current_node.val
    current_node.val = cumulative_sum
    print(f"\t\tUpdating node value:")
    print(f"\t\t\tOld value: {old_val}")
    print(f"\t\t\tNew cumulative_sum: {cumulative_sum}")
    print(f"\t\t\tUpdated node value: {current_node.val}")
    iteration_data.append([old_val, cumulative_sum - old_val, cumulative_sum])
    print(f"\t\tMoving to left child: {current_node.left.val if current_node.left else None}")
    current_node = current_node.left
    return cumulative_sum, current_node


# 1382. Balance a Binary Search Tree

# Given the `root` of a binary search tree, return a balanced binary search tree with the same node values.
# If there is more than one answer, return any of them.
# A binary search tree is balanced if the depth of the two subtrees of every node never differs by more than `1`.


def balanceBST1(root: TreeNode) -> TreeNode:
    """
    Converts an unbalanced binary search tree (BST) into a balanced one.

    This function uses a two-step approach to balance the BST:
    1. It performs an in-order traversal of the original BST, storing nodes in a list.
       This step ensures that we have a sorted list of nodes, as in-order traversal
       of a BST yields nodes in ascending order.
    2. It then recursively constructs a new balanced BST from this sorted list,
       using the middle element as the root at each step.
       This approach guarantees that the tree remains balanced, as we always choose the median element as the root.

    The time complexity of this solution is O(n), where `n` is the number of nodes in the tree.
    This is because we traverse each node once during the in-order traversal and once
    during the balanced BST construction.
    The space complexity is O(n) as well, due to the storage of all nodes in the
    `inorder_nodes` list and the recursion stack used in both traversal and construction.
    """
    inorder_nodes = []

    def inorder_traverse(root: TreeNode) -> None:
        """Helper function to perform in-order traversal of the tree."""
        if not root:
            return
        inorder_traverse(root.left)
        inorder_nodes.append(root)
        inorder_traverse(root.right)

    def build_balanced_bst(start_index: int, end_index: int) -> TreeNode | None:
        """Helper function to construct a balanced BST from the sorted list of nodes."""
        if start_index > end_index:
            return None

        mid_index = start_index + (end_index - start_index) // 2

        node = inorder_nodes[mid_index]
        node.left = build_balanced_bst(start_index, mid_index - 1)
        node.right = build_balanced_bst(mid_index + 1, end_index)
        return node

    inorder_traverse(root)
    return build_balanced_bst(0, len(inorder_nodes) - 1)


def balanceBST2(root: TreeNode) -> TreeNode:
    """
    Converts an unbalanced binary search tree (BST) into a balanced one.

    This function implements the Day-Stout-Warren (DSW) algorithm to balance a BST in three main steps:
    1. Convert the BST into a "vine" (a right-skewed tree)
    2. Determine the number of nodes needed for a perfect binary tree
    3. Perform a series of left rotations to balance the tree

    The DSW algorithm achieves balance through a series of tree rotations,
    without requiring extra space for node storage. It guarantees a balanced
    tree with a height of O(log n).

    The time complexity of this solution is O(n), where `n` is the number of nodes.
    Despite multiple passes through the tree, each node is processed a constant
    number of times.
    The space complexity is O(1), as it uses only a constant amount of extra space,
    regardless of the input size. This is a key advantage over methods that require
    O(n) auxiliary space.
    """

    def right_rotate(parent: TreeNode, node: TreeNode) -> None:
        """Helper function to perform a right rotation on the given node."""
        left_child = node.left
        node.left = left_child.right
        left_child.right = node
        parent.right = left_child

    def left_rotate(parent: TreeNode, node: TreeNode) -> None:
        """Helper function to perform a left rotation on the given node."""
        right_child = node.right
        node.right = right_child.left
        right_child.left = node
        parent.right = right_child

    def compress_vine(vine_root: TreeNode, rotations: int) -> None:
        """Helper function to perform a series of left rotations to balance the vine."""
        current_node = vine_root
        for _ in range(rotations):
            child = current_node.right
            left_rotate(current_node, child)
            current_node = current_node.right

    dummy_root = TreeNode()
    dummy_root.right = root
    current_node = dummy_root

    # Step 1: Convert BST to vine (right-leaning linked list)
    while current_node.right:
        if current_node.right.left:
            right_rotate(current_node, current_node.right)
        else:
            current_node = current_node.right

    # Step 2: Count nodes and calculate perfect tree size
    node_count = 0
    current_node = dummy_root.right
    while current_node:
        node_count += 1
        current_node = current_node.right

    # Calculate the number of nodes in the perfect tree portion
    perfect_tree_nodes = 2 ** math.floor(math.log2(node_count + 1)) - 1

    # Step 3: Balance the tree through a series of left rotations
    # Perform initial compression
    compress_vine(dummy_root, node_count - perfect_tree_nodes)

    # Perform remaining compressions
    remaining_nodes = perfect_tree_nodes
    while remaining_nodes > 1:
        remaining_nodes //= 2
        compress_vine(dummy_root, remaining_nodes)

    return dummy_root.right


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
# bstToGst1(root=TreeNode(val=4, left=TreeNode(val=1, left=TreeNode(), right=TreeNode(val=2, right=TreeNode(val=3))),
#                         right=TreeNode(val=6, left=TreeNode(5), right=TreeNode(val=7, right=TreeNode(val=8)))))
# bstToGst2(root=TreeNode(val=4, left=TreeNode(val=1, left=TreeNode(), right=TreeNode(val=2, right=TreeNode(val=3))),
#                         right=TreeNode(val=6, left=TreeNode(5), right=TreeNode(val=7, right=TreeNode(val=8)))))
# bstToGst3(root=TreeNode(val=4, left=TreeNode(val=1, left=TreeNode(), right=TreeNode(val=2, right=TreeNode(val=3))),
#                         right=TreeNode(val=6, left=TreeNode(5), right=TreeNode(val=7, right=TreeNode(val=8)))))

# Test cases for June 26th, 2024
# Expected output:
# TreeNode(3, left=TreeNode(2, left=TreeNode(1), right=TreeNode()), right=TreeNode(4))
# balanceBST1(root=TreeNode(val=1, right=TreeNode(val=2, right=TreeNode(val=3, right=TreeNode(val=4)))))
# balanceBST2(root=TreeNode(val=1, right=TreeNode(val=2, right=TreeNode(val=3, right=TreeNode(val=4)))))

# Test cases for June 27th, 2024

# Test cases for June 28th, 2024

# Test cases for June 29th, 2024

# Test cases for June 30th, 2024
