# June 2024, Week 5: June 24th - June 30th
import math
from collections import deque, defaultdict
from typing import List

from tabulate import tabulate

from Utils.graph_utils import UnionFindWithLogs
from Utils.trees_utils import BinaryTreeNode, BinaryTreeVisualizer


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


def bstToGst1(root: BinaryTreeNode) -> BinaryTreeNode:
    print("\n--- Input Parameters ---")
    print(f"\troot = {BinaryTreeVisualizer.visualize(root, file_name='bstToGst1-Input')}")

    print("\n--- Initialization ---")
    cumulative_sum = 0
    print(f"\tcumulative_sum = {cumulative_sum}")

    iteration_data = []

    def update_node_values(node: BinaryTreeNode) -> None:
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
    print(f"\tModified Tree: {BinaryTreeVisualizer.visualize(root, file_name='bstToGst1-Output')}")
    return root


def bstToGst2(root: BinaryTreeNode) -> BinaryTreeNode:
    print("\n--- Input Parameters ---")
    print(f"\troot = {BinaryTreeVisualizer.visualize(root, file_name='bstToGst2-Input')}")

    print("\n--- Initialization ---")
    stack = []
    current_node = root
    cumulative_sum = 0
    print(f"\tstack = {stack}")
    print(f"\tcurrent_node = {current_node.val if current_node else None}")
    print(f"\tcumulative_sum = {cumulative_sum}")

    iteration_data = []

    def push_right_nodes(node: BinaryTreeNode) -> None:
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
    print(f"\tModified Tree: {BinaryTreeVisualizer.visualize(root, file_name='bstToGst2-Output')}")
    return root


def bstToGst3(root: BinaryTreeNode) -> BinaryTreeNode:
    print("\n--- Input Parameters ---")
    print(f"\troot = {BinaryTreeVisualizer.visualize(root, file_name='bstToGst3-Input')}")

    def find_successor(current_node: BinaryTreeNode) -> BinaryTreeNode:
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
    print(f"\tModified Tree: {BinaryTreeVisualizer.visualize(root, file_name='bstToGst3-Output')}")
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


# <------------------------------------------------- June 26th, 2024 ------------------------------------------------->
# 1382. Balance a Binary Search Tree

# Given the `root` of a binary search tree, return a balanced binary search tree with the same node values.
# If there is more than one answer, return any of them.
# A binary search tree is balanced if the depth of the two subtrees of every node never differs by more than `1`.


def balanceBST1(root: BinaryTreeNode) -> BinaryTreeNode:
    print("\n--- Input Parameters ---")
    print(f"\troot = {BinaryTreeVisualizer.visualize(root, file_name='balanceBST1-Input')}")

    print("\n--- Initialization ---")
    inorder_nodes = []
    print(f"\tinorder_nodes = {inorder_nodes}")

    def inorder_traverse(node: BinaryTreeNode) -> None:
        if not node:
            print(f"\t\tNode is None")
            return
        print(f"\n\t--- Traversing Node: {node.val} ---")
        print("\t\tMoving to left child")
        inorder_traverse(node.left)
        print(f"\t\tAppending node {node.val} to inorder_nodes")
        inorder_nodes.append(node)
        print("\t\tMoving to right child")
        inorder_traverse(node.right)

    def build_balanced_bst(start_index: int, end_index: int) -> BinaryTreeNode | None:
        print(f"\n\t--- Building BST: start_index={start_index}, end_index={end_index} ---")
        if start_index > end_index:
            print("\t\tBase case reached: returning None")
            return None

        mid_index = start_index + (end_index - start_index) // 2
        print(f"\t\tCalculated mid_index: {mid_index}")

        node = inorder_nodes[mid_index]
        print(f"\t\tSelected root node: {node.val}")

        print(f"\t\tBuilding left subtree: start_index={start_index}, end_index={mid_index - 1}")
        node.left = build_balanced_bst(start_index, mid_index - 1)
        print(f"\t\tBuilding right subtree: start_index={mid_index + 1}, end_index={end_index}")
        node.right = build_balanced_bst(mid_index + 1, end_index)

        return node

    print("\n--- Performing In-order Traversal ---")
    inorder_traverse(root)

    print("\n--- In-order Traversal Result ---")
    print(f"\tinorder_nodes = {[node.val for node in inorder_nodes]}")

    print("\n--- Building Balanced BST ---")
    balanced_root = build_balanced_bst(0, len(inorder_nodes) - 1)

    print("\n--- Function Returning ---")
    print(f"\tBalanced Tree: {BinaryTreeVisualizer.visualize(balanced_root, file_name='balanceBST1-Output')}")
    return balanced_root


def balanceBST2(root: BinaryTreeNode) -> BinaryTreeNode:
    print("\n--- Input Parameters ---")
    print(f"\tInitial Tree:\n{BinaryTreeVisualizer.visualize(root, file_name='balanceBST2-Input')}")

    right_rotation_count = 0
    left_rotation_count = 0
    compression_count = 0

    def right_rotate(parent: BinaryTreeNode, node: BinaryTreeNode) -> None:
        nonlocal right_rotation_count
        right_rotation_count += 1
        print(f"\n\t--- Right Rotation ---")
        print(f"\t\tBefore rotation saved at: {BinaryTreeVisualizer.visualize(dummy_root.right,
                                                                              file_name=f"balanceBST2-Right_rot_"
                                                                                        f"{right_rotation_count}_before")}")
        left_child = node.left
        print(f"\t\tParent: {parent.val}, Node: {node.val}")
        print(f"\t\tLeft child: {left_child.val}")
        node.left = left_child.right
        left_child.right = node
        parent.right = left_child
        print(f"\t\tAfter rotation saved at: {BinaryTreeVisualizer.visualize(dummy_root.right,
                                                                             file_name=f"balanceBST2-Right_rot_"
                                                                                       f"{right_rotation_count}_after")}")

    def left_rotate(parent: BinaryTreeNode, node: BinaryTreeNode) -> None:
        print(f"\n\t--- Left Rotation ---")
        nonlocal left_rotation_count
        left_rotation_count += 1
        print(f"\t\tBefore rotation saved at: {BinaryTreeVisualizer.visualize(dummy_root.right,
                                                                              file_name=f"balanceBST2-Left_rot_"
                                                                                        f"{left_rotation_count}_before")}")
        print(f"\t\tParent: {parent.val}, Node: {node.val}")
        right_child = node.right
        print(f"\t\tRight child: {right_child.val}")
        node.right = right_child.left
        right_child.left = node
        parent.right = right_child
        print(f"\t\tAfter rotation saved at: {BinaryTreeVisualizer.visualize(dummy_root.right,
                                                                             file_name=f"balanceBST2-Left_rot_"
                                                                                       f"{left_rotation_count}_after")}")

    def compress_vine(vine_root: BinaryTreeNode, rotations: int) -> None:
        print(f"\n\t--- Compressing Vine: {rotations} rotations ---")
        if rotations > 0:
            nonlocal compression_count
            compression_count += 1
            print(f"\t\tBefore compression saved at: {BinaryTreeVisualizer.visualize(dummy_root.right,
                                                                                     file_name=f"balanceBST2-Compression_"
                                                                                               f"{compression_count}_before"
                                                                                     )}")
        else:
            print("\t\tNo rotations needed")
            return
        current_node = vine_root
        for i in range(rotations):
            child = current_node.right
            print(f"\t\tRotation {i + 1}: Left rotate {current_node.val} and {child.val}")
            left_rotate(current_node, child)
            current_node = current_node.right
        print(f"\t\tAfter compression saved at: {BinaryTreeVisualizer.visualize(dummy_root.right,
                                                                                file_name=f"balanceBST2-Compression_"
                                                                                          f"{compression_count}_after")}")

    print("\n--- Initialization ---")
    dummy_root = BinaryTreeNode(val="dummy")
    dummy_root.right = root
    current_node = dummy_root
    print(f"\tDummy root created and connected to tree root")

    print("\n--- Step 1: Convert BST to vine (right-leaning linked list) ---")
    vine_nodes = []
    while current_node.right:
        if current_node.right.left:
            print(f"\tRight rotate needed at node {current_node.right.val}")
            right_rotate(current_node, current_node.right)
        else:
            current_node = current_node.right
            vine_nodes.append(current_node.val)
            print(f"\tMoved to next node: {current_node.val}")

    print("\n\tVine creation complete")
    print(f"\tVine nodes: {vine_nodes}")
    print(f"\tFinal vine structure saved at: {BinaryTreeVisualizer.visualize(dummy_root.right, file_name='balanceBST2-Vine')}")

    print("\n--- Step 2: Count nodes and calculate perfect tree size ---")
    node_count = len(vine_nodes)
    print(f"\tTotal node count: {node_count}")

    perfect_tree_nodes = 2 ** math.floor(math.log2(node_count + 1)) - 1
    print(f"\tPerfect tree nodes: {perfect_tree_nodes}")

    print("\n--- Step 3: Balance the tree through a series of left rotations ---")
    print("\tPerform initial compression")
    initial_compressions = node_count - perfect_tree_nodes
    compress_vine(dummy_root, initial_compressions)

    print("\n\tPerform remaining compressions")
    remaining_nodes = perfect_tree_nodes
    compression_rounds = []
    while remaining_nodes > 1:
        remaining_nodes //= 2
        compress_vine(dummy_root, remaining_nodes)
        compression_rounds.append(remaining_nodes)

    print("\n--- Compression Summary ---")
    print(f"\tInitial compressions: {initial_compressions}")
    print(f"\tCompression rounds: {compression_rounds}")

    print("\n--- Function Returning ---")
    balanced_root = dummy_root.right
    print(
        f"\tFinal Balanced Tree saved at: {BinaryTreeVisualizer.visualize(balanced_root, file_name='balanceBST2-Output')}")
    return balanced_root


# <------------------------------------------------- June 27th, 2024 ------------------------------------------------->
# 1791. Find Center of Star Graph

# Given an undirected star graph of n nodes labeled from 1 to n, represented by a 2D integer array edges where each
# edges[i] = [u_i, v_i] indicates an edge between nodes u_i and v_i, the task is to return the center of this star
# graph.

def findCenter1(edges: List[List[int]]) -> int:
    print("\n--- Input Parameters ---")
    print(f"\tedges = {edges}")

    print("\n--- Initialization ---")
    reference_edge = edges[0]
    comparison_edge = edges[1]
    print(f"\treference_edge = {reference_edge}")
    print(f"\tcomparison_edge = {comparison_edge}")

    print("\n--- Center Node Identification ---")
    print(f"\tChecking if {reference_edge[0]} is in {comparison_edge}")
    if reference_edge[0] in comparison_edge:
        center = reference_edge[0]
        print(f"\t\tCondition true: {reference_edge[0]} is in {comparison_edge}")
        print(f"\t\tCenter node identified: {center}")
    else:
        center = reference_edge[1]
        print(f"\t\tCondition false: {reference_edge[0]} is not in {comparison_edge}")
        print(f"\t\tCenter node must be the other node in reference_edge: {center}")

    print("\n--- Decision Summary ---")
    headers = ["Reference Edge", "Comparison Edge", "Condition", "Center Node"]
    data = [[reference_edge, comparison_edge, f"{reference_edge[0]} in {comparison_edge}", center]]
    print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tReturning center node: {center}")
    return center


# <------------------------------------------------- June 28th, 2024 ------------------------------------------------->
# 2285. Maximum Total Importance of Roads

# Given an integer `n` representing the number of cities and a 2D integer array `roads` denoting bidirectional roads
# between cities, the task is to assign each city a unique value from `1` to `n` such that the sum of the values of two
# cities connected by a road (the road's importance) is maximized.
# The goal is to return the maximum total importance of all roads after optimal assignment.


def maximumImportance1(n: int, roads: List[List[int]]) -> int:
    print("\n--- Input Parameters ---")
    print(f"\tn = {n}")
    print(f"\troads = {roads}")

    print("\n--- Initialization ---")
    city_connections = [0] * n
    print(f"\tcity_connections = {city_connections}")

    print("\n--- Counting Connections ---")
    for i, (city1, city2) in enumerate(roads):
        print(f"\n--- Road {i + 1}/{len(roads)} ---")
        print(f"\tProcessing road: {city1} <-> {city2}")

        print(f"\tBefore update: city_connections = {city_connections}")
        city_connections[city1] += 1
        city_connections[city2] += 1
        print(f"\tAfter update:  city_connections = {city_connections}")

    print("\n--- Connection Count Summary ---")
    connection_summary = [[city, count] for city, count in enumerate(city_connections)]
    print(tabulate(connection_summary, headers=["City", "Connections"], tablefmt="fancy_grid"))

    print("\n--- Initializing Result Variables ---")
    total_importance = 0
    city_value = 1
    print(f"\ttotal_importance = {total_importance}")
    print(f"\tcity_value = {city_value}")

    print("\n--- Sorting and Importance Calculation ---")
    sorted_connections = sorted(city_connections)
    print(f"\tSorted connections: {sorted_connections}")

    iteration_data = []
    for i, connections in enumerate(sorted_connections):
        print(f"\n--- City {i + 1}/{n} ---")
        print(f"\tConnections: {connections}")
        print(f"\tCurrent city_value: {city_value}")

        importance_contribution = city_value * connections
        print(f"\tImportance contribution calculation:")
        print(f"\t\t{city_value} * {connections} = {importance_contribution}")

        total_importance += importance_contribution
        print(f"\tUpdated total_importance: {total_importance}")

        iteration_data.append([i + 1, connections, city_value, importance_contribution, total_importance])

        city_value += 1
        print(f"\tIncremented city_value for next iteration: {city_value}")

    print("\n--- Iteration Summary ---")
    headers = ["City #", "Connections", "Assigned Value", "Importance Contribution", "Total Importance"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal total_importance: {total_importance}")
    return total_importance


# <------------------------------------------------- June 29th, 2024 ------------------------------------------------->
# 2192. All Ancestors of a Node in a Directed Acyclic Graph

# Given a positive integer `n` representing the number of nodes in a Directed Acyclic Graph (DAG) and a 2D integer
# array `edges` denoting unidirectional edges, the task is to return a list where each element is a sorted list of
# ancestors for the corresponding node.
# A node `u` is considered an ancestor of node `v` if `u` can reach `v` via a set of edges.


def getAncestors1(n: int, edges: List[List[int]]) -> List[List[int]]:
    """
    Determines the ancestors of each node in a Directed Acyclic Graph (DAG).

    This function uses a depth-first search (DFS) approach to propagate ancestor information
    through the graph.
    """
    print("\n--- Input Parameters ---")
    print(f"\tn = {n}")
    print(f"\tedges = {edges}")

    print("\n--- Initialization ---")
    direct_children = defaultdict(list)
    ancestors: List[List[int]] = [[] for _ in range(n)]
    print(f"\tdirect_children (initially empty) = {dict(direct_children)}")
    print(f"\tancestors (initially empty) = {ancestors}")

    print("\n--- Building Adjacency List ---")
    for parent, child in edges:
        direct_children[parent].append(child)
        print(f"\tAdding edge: parent {parent} -> child {child}")
        print(f"\tUpdated direct_children = {dict(direct_children)}")

    def propagate_ancestor(current_ancestor: int, current_node: int):
        """Helper function to recursively propagate ancestry information from a node to its descendants."""
        print(f"\n\t--- Propagating ancestor {current_ancestor} from node {current_node} ---")
        print(f"\t\tCurrent node children: {direct_children[current_node]}")
        if not direct_children[current_node]:
            print(f"\t\tNo children for node {current_node}, going back")
            return
        for child in direct_children[current_node]:
            print(f"\t\tChecking child {child} of node {current_node}")
            if not ancestors[child] or ancestors[child][-1] != current_ancestor:
                print(f"\t\t\tAdding ancestor {current_ancestor} to node {child}")
                ancestors[child].append(current_ancestor)
                print(f"\t\t\tUpdated ancestors[{child}] = {ancestors[child]}")
                print(f"\t\t\tRecursively propagating to descendants of {child}")
                propagate_ancestor(current_ancestor, child)
            else:
                print(f"\t\t\tSkipping: {current_ancestor} is already an ancestor of {child}")

    print("\n--- Main Loop: Propagating Ancestors ---")
    iteration_data = []
    for node in range(n):
        print(f"\n--- Processing Node {node}/{n - 1} ---")
        print(f"\tCurrent ancestors: {ancestors}")
        print(f"\tPropagating {node} as an ancestor")
        propagate_ancestor(node, node)
        iteration_data.append([node, ancestors[node], [child for child in direct_children[node]]])

    print("\n--- Iteration Summary ---")
    headers = ["Node", "Ancestors", "Direct Children"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal ancestors list: {ancestors}")
    return ancestors


def getAncestors2(n: int, edges: List[List[int]]) -> List[List[int]]:
    """
    Determines the ancestors of each node in a Directed Acyclic Graph (DAG) using DFS with memoization.
    """
    print("\n--- Input Parameters ---")
    print(f"\tn = {n}")
    print(f"\tedges = {edges}")

    print("\n--- Initialization ---")
    direct_parents = defaultdict(list)
    ancestors = [[] for _ in range(n)]
    print(f"\tdirect_parents (initially empty) = {dict(direct_parents)}")
    print(f"\tancestors (initially empty) = {ancestors}")

    print("\n--- Building Reverse Adjacency List ---")
    for parent, child in edges:
        direct_parents[child].append(parent)
        print(f"\tAdding edge: child {child} <- parent {parent}")
        print(f"\tUpdated direct_parents = {dict(direct_parents)}")

    def find_all_ancestors(node: int) -> set:
        """Helper function to recursively find and memoize all ancestors of a given node."""
        print(f"\n\t--- Finding ancestors for node {node} ---")
        if ancestors[node]:
            print(f"\t\tAncestors for node {node} already computed: {ancestors[node]}")
            return set(ancestors[node])

        print(f"\t\tComputing ancestors for node {node}")
        print(f"\t\tDirect parents of node {node}: {direct_parents[node]}")
        if not direct_parents[node]:
            print(f"\t\tNo direct parents for node {node}, returning empty set")
            ancestors[node] = []
            return set()
        result = set()
        for parent in direct_parents[node]:
            print(f"\t\t\tProcessing parent {parent} of node {node}")
            parent_ancestors = find_all_ancestors(parent)
            print(f"\t\t\tAncestors of parent {parent}: {parent_ancestors}")
            print(f"\t\t\tAdding ancestors of parent {parent} to result")
            result.update(parent_ancestors)
            print(f"\t\t\tUpdated result: {result}")

        print(f"\t\tAdding direct parents of node {node} to result")
        result.update(direct_parents[node])
        print(f"\t\tFinal result for node {node}: {result}")

        ancestors[node] = list(result)
        print(f"\t\tUpdated ancestors[{node}] = {ancestors[node]}")
        return result

    print("\n--- Main Loop: Computing Ancestors ---")
    iteration_data = []
    for node in range(n):
        print(f"\n--- Processing Node {node}/{n - 1} ---")
        if not ancestors[node]:
            print(f"\tAncestors for node {node} not yet computed")
            find_all_ancestors(node)
        else:
            print(f"\tAncestors for node {node} already computed: {ancestors[node]}")
        iteration_data.append([node, ancestors[node], direct_parents[node]])

    print("\n--- Sorting Ancestors ---")
    for node in range(n):
        print(f"\tSorting ancestors for node {node}")
        print(f"\t\tBefore sorting: {ancestors[node]}")
        ancestors[node].sort()
        print(f"\t\tAfter sorting: {ancestors[node]}")

    print("\n--- Iteration Summary ---")
    headers = ["Node", "Ancestors", "Direct Parents"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal ancestors list: {ancestors}")
    return ancestors


def getAncestors3(n: int, edges: List[List[int]]) -> List[List[int]]:
    """
    Determines the ancestors of each node in a Directed Acyclic Graph (DAG) using BFS and topological sorting.
    """
    print("\n--- Input Parameters ---")
    print(f"\tn = {n}")
    print(f"\tedges = {edges}")

    print("\n--- Initialization ---")
    direct_children = defaultdict(list)
    ancestors = [set() for _ in range(n)]
    in_degree = [0] * n
    print(f"\tdirect_children (initially empty) = {dict(direct_children)}")
    print(f"\tancestors (initially empty) = {ancestors}")
    print(f"\tin_degree (initially all zeros) = {in_degree}")

    print("\n--- Building Graph Data Structures ---")
    for parent, child in edges:
        direct_children[parent].append(child)
        ancestors[child].add(parent)
        in_degree[child] += 1
        print(f"\tProcessing edge: parent {parent} -> child {child}")
        print(f"\t\tUpdated direct_children[{parent}] = {direct_children[parent]}")
        print(f"\t\tUpdated ancestors[{child}] = {ancestors[child]}")
        print(f"\t\tUpdated in_degree[{child}] = {in_degree[child]}")

    print(f"\ndirect_children: {dict(direct_children)}")
    print(f"ancestors: {ancestors}")
    print(f"in_degree: {in_degree}")

    print("\n--- Initializing Queue ---")
    queue = deque(node for node in range(n) if in_degree[node] == 0)
    print(f"\tInitial queue (nodes with in_degree 0): {list(queue)}")

    print("\n--- Processing Nodes in Topological Order ---")
    iteration_data = []
    while queue:
        current_node = queue.popleft()
        print(f"\n\t--- Processing Node {current_node} ---")
        print(f"\t\tCurrent queue: {list(queue)}")
        print(f"\t\tCurrent node ancestors: {ancestors[current_node]}")
        print(f"\t\tCurrent node children: {direct_children[current_node]}")

        for child_node in direct_children[current_node]:
            print(f"\n\t\tProcessing child node {child_node}")
            print(f"\t\t\tBefore update: ancestors[{child_node}] = {ancestors[child_node]}, "
                  f"adding ancestors of {current_node} = {ancestors[current_node]}")
            ancestors[child_node].update(ancestors[current_node])
            print(f"\t\t\tAfter update: ancestors[{child_node}] = {ancestors[child_node]}")

            print(f"\t\t\tUpdated in_degree[{child_node}] = {in_degree[child_node]} - 1 = {in_degree[child_node] - 1}")
            in_degree[child_node] -= 1

            if in_degree[child_node] == 0:
                queue.append(child_node)
                print(f"\t\t\tAdded node {child_node} to queue")
                print(f"\t\t\tUpdated queue: {list(queue)}")

        iteration_data.append(
            [current_node, ancestors[current_node], direct_children[current_node], in_degree[current_node]])

    print("\n--- Sorting Ancestors ---")
    sorted_ancestors = []
    for node, node_ancestors in enumerate(ancestors):
        print(f"\tSorting ancestors for node {node}")
        print(f"\t\tBefore sorting: {node_ancestors}")
        sorted_node_ancestors = sorted(node_ancestors)
        print(f"\t\tAfter sorting: {sorted_node_ancestors}")
        sorted_ancestors.append(sorted_node_ancestors)

    print("\n--- Iteration Summary ---")
    headers = ["Processed Node", "Ancestors", "Direct Children", "Final In-Degree"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal sorted ancestors list: {sorted_ancestors}")
    return sorted_ancestors


# <------------------------------------------------- June 30th, 2024 ------------------------------------------------->
# 1579. Remove Max Number of Edges to Keep Graph Fully Traversable

# Alice and Bob have an undirected graph with three types of edges, each type representing who can traverse it.
# The task is to find the maximum number of edges that can be removed such that the graph can still be fully traversed
# by both Alice and Bob, or return -1 if it's impossible.
# Alice can only traverse the first type of edges, Bob can only traverse the second type of edges, and the
# third type of edges can be traversed by both Alice and Bob.


def maxNumEdgesToRemove1(n: int, edges: List[List[int]]) -> int:
    print("\n--- Input Parameters ---")
    print(f"\tn = {n}")
    print(f"\tedges = {edges}")

    print("\n--- Initialization ---")
    alice_uf = UnionFindWithLogs(size=n, offset=1)
    bob_uf = UnionFindWithLogs(size=n, offset=1)
    essential_edges = 0
    print(f"\talice_uf initialized")
    print(f"\tbob_uf initialized")
    print(f"\tessential_edges = {essential_edges}")

    print("\n--- First Pass: Processing Type 3 Edges ---")
    first_pass_data = []
    for i, (edge_type, u, v) in enumerate(edges, 1):
        if edge_type == 3:
            print(f"\n--- Edge {i}/{len(edges)} ---")
            print(f"\tEdge: Type {edge_type}, connects nodes {u} and {v}")

            print("\tProcessing for Alice:")
            alice_union = alice_uf.union(u, v)
            print(f"\t\tUnion result: {alice_union}")

            print("\tProcessing for Bob:")
            bob_union = bob_uf.union(u, v)
            print(f"\t\tUnion result: {bob_union}")

            union_result = alice_union | bob_union
            essential_edges += union_result
            print(f"\tCombined union result: {union_result}")
            print(f"\tUpdated essential_edges: {essential_edges}")

            first_pass_data.append([i, edge_type, u, v, alice_union, bob_union, union_result, essential_edges])

            if alice_uf.is_single_component():
                print("\n\tAlice's graph is fully connected")
                print(f"\tReturning: {len(edges) - essential_edges}")
                return len(edges) - essential_edges

    print("\n--- First Pass Summary ---")
    headers = ["Edge #", "Type", "U", "V", "Alice Union", "Bob Union", "Combined", "Essential Edges"]
    print(tabulate(first_pass_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Second Pass: Processing Type 1 and 2 Edges ---")
    second_pass_data = []
    loop_is_over = False
    for i, (edge_type, u, v) in enumerate(edges, 1):
        if edge_type in [1, 2]:
            print(f"\n--- Edge {i}/{len(edges)} ---")
            print(f"\tEdge: Type {edge_type}, connects nodes {u} and {v}")

            if edge_type == 1:
                print("\tProcessing for Alice:")
                union_result = alice_uf.union(u, v)
                print(f"\t\tUnion result: {union_result}")
            else:
                print("\tProcessing for Bob:")
                union_result = bob_uf.union(u, v)
                print(f"\t\tUnion result: {union_result}")

            if union_result:
                essential_edges += 1
                print(f"\tUpdated essential_edges: {essential_edges}")

            second_pass_data.append([i, edge_type, u, v, union_result, essential_edges])

            print("\tChecking full connectivity:")
            alice_connected = alice_uf.is_single_component()
            bob_connected = bob_uf.is_single_component()
            print(f"\t\tAlice's graph fully connected: {alice_connected}")
            print(f"\t\tBob's graph fully connected: {bob_connected}")

            if alice_connected and bob_connected:
                print("\n\tBoth graphs are fully connected")
                loop_is_over = True
                break

    print("\n--- Second Pass Summary ---")
    headers = ["Edge #", "Type", "U", "V", "Union Result", "Essential Edges"]
    print(tabulate(second_pass_data, headers=headers, tablefmt="fancy_grid"))

    if loop_is_over:
        print(f"\tReturning: len(edges) - essential_edges = {len(edges)} - {essential_edges} ="
              f" {len(edges) - essential_edges}")
        return len(edges) - essential_edges

    print("\n--- Function Returning ---")
    print("\tFull traversal is not possible for both Alice and Bob")
    print("\tReturning: -1")
    return -1

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
# balanceBST1(root=TreeNode(val=4, left=TreeNode(val=3, left=TreeNode(val=2, left=TreeNode(val=1)))))
# balanceBST2(root=TreeNode(val=4, left=TreeNode(val=3, left=TreeNode(val=2, left=TreeNode(val=1)))))

# Test cases for June 27th, 2024
# Expected output: 1
# findCenter1(edges=[[1, 2], [5, 1], [1, 3], [1, 4]])

# Test cases for June 28th, 2024
# Expected output: 43
# maximumImportance1(n=5, roads=[[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [2, 4]])

# Test cases for June 29th, 2024
# Expected output: [[], [], [], [0, 1], [0, 2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3]]
# getAncestors1(n=8, edges=[[0, 3], [0, 4], [1, 3], [2, 4], [2, 7], [3, 5], [3, 6], [3, 7], [4, 6]])
# getAncestors2(n=8, edges=[[0, 3], [0, 4], [1, 3], [2, 4], [2, 7], [3, 5], [3, 6], [3, 7], [4, 6]])
# getAncestors3(n=8, edges=[[0, 3], [0, 4], [1, 3], [2, 4], [2, 7], [3, 5], [3, 6], [3, 7], [4, 6]])

# Test cases for June 30th, 2024
# Expected output: 2
# maxNumEdgesToRemove1(n=4, edges=[[3, 1, 2], [3, 2, 3], [1, 1, 3], [1, 2, 4], [1, 1, 2], [2, 3, 4]])
