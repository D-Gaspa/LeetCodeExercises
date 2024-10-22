import bisect
from collections import defaultdict
from typing import List, Optional

from tabulate import tabulate

from Utils.trees_utils import BinaryTreeNode, BinaryTreeVisualizer


# June, 2024
# <--------------------------------------------------- Week 1, June --------------------------------------------------->
# 1940. Longest Common Subsequence Between Sorted Arrays

# Given an array of integer arrays `arrays` where each `arrays[i]` is sorted in strictly increasing order,
# return an integer array representing the longest common subsequence between all the arrays.


def longestCommonSubsequence1(arrays: List[List[int]]) -> List[int]:
    """
    Finds the longest common subsequence present in the given list of sorted arrays.

    This function performs a preliminary check to identify the smallest array, then checks for numbers in this
    array across all other arrays in the list using a helper binary search function.

    The time complexity of this solution is O(n * m * log(k)), skewed towards the smallest size of the arrays.
    Where n is the number of elements in the smallest array, m is the total number of arrays,
    and k is the average length of the arrays.

    The space complexity is O(L), where L is the length of the longest common subsequence.
    """
    print("\n--- Input Parameters ---")
    print(f"\tarrays = {arrays}")

    def binary_search(array, num):
        """Helper function to perform binary search for 'num' in 'array'."""
        print(f"\t\tBinary searching for {num} in {array}")
        index = bisect.bisect_left(array, num)
        result = index != len(array) and array[index] == num
        print(f"\t\tFound: {result}")  # Print search result
        return result

    print("\n--- Identifying Smallest Array ---")
    smallest_array = min(arrays, key=len)
    arrays.remove(smallest_array)
    print(f"\tSmallest array: {smallest_array}")

    common_subsequences = []
    iteration_data = []

    print("\n--- Main Loop (Finding Common Elements) ---")
    for i, num in enumerate(smallest_array):
        print(f"\nIteration {i + 1}:")
        print(f"\tnum = {num}")
        # The line below should be kept intact for functionality reasons:
        if all(binary_search(array, num) for array in arrays):  # Check if 'num' is present in all other arrays
            common_subsequences.append(num)
            print(f"\t\tAdded to common_subsequences: {common_subsequences}")
        iteration_data.append([i + 1, num, common_subsequences.copy()])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "num", "common_subsequences"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"longestCommonSubsequence1 = {common_subsequences}")
    return common_subsequences


def longestCommonSubsequence2(arrays: List[List[int]]) -> List[int]:
    """
    Finds the longest common subsequence present in the given list of sorted arrays.

    The main logic of the function is based on the two-pointer technique used in iterating through sorted lists
    for comparison.
    It iteratively finds the common elements of the first list (starting as the common subsequence) and the next ones.
    For each list, it runs through the elements of the current common subsequence and the new list side by side.
    If elements don't match, it advances the index of the smaller element.
    Every time a common value is found, it is added to the new_subsequence that ultimately replaces the current
    common_subsequence for next comparisons.

    The time complexity of this function is O(n * m), where n is the total number of lists,
    and m is the average size of these lists.
    This is because we are running through each list once, comparing and moving the pointers in a linear fashion.
    The space complexity is O(K), where K is the maximum length among the input arrays
    (used to store intermediate subsequences).
    """
    print("\n--- Input Parameters ---")
    print(f"\tarrays = {arrays}")

    common_subsequences = arrays[0]
    iteration_data = []  # To collect data for iteration summary

    print("\n--- Main Loop (Finding Common Subsequences) ---")
    for i, array in enumerate(arrays[1:], start=1):  # Start from the second array
        print(f"\nIteration {i}: Comparing with array {array}")

        new_subsequence = []
        array_index = 0
        common_subseq_index = 0
        array_length = len(array)
        common_subseq_length = len(common_subsequences)
        iteration_steps = []  # To collect data for each step within the iteration

        while common_subseq_index < common_subseq_length and array_index < array_length:
            print(
                f"\tComparing: common_subsequences[{common_subseq_index}] = {common_subsequences[common_subseq_index]}"
                f" vs. array[{array_index}] = {array[array_index]}")
            if array[array_index] == common_subsequences[common_subseq_index]:
                new_subsequence.append(array[array_index])
                print(f"\t\tMatch found! Added to new_subsequence: {new_subsequence}")
                print(f"\t\tAdvancing both pointers")
                common_subseq_index += 1
                array_index += 1
            elif array[array_index] < common_subsequences[common_subseq_index]:
                print(f"\t\tAdvancing array_index")
                array_index += 1
            else:
                print(f"\t\tAdvancing common_subseq_index")
                common_subseq_index += 1

            iteration_steps.append([common_subseq_index, array_index, new_subsequence.copy()])

        # Print Iteration Summary
        print(f"\n\t--- Iteration {i} Steps Summary ---")
        headers = ["common_subseq_index", "array_index", "new_subsequence"]
        print(tabulate(iteration_steps, headers=headers, tablefmt="fancy_grid"))

        common_subsequences = new_subsequence
        iteration_data.append([i, array, new_subsequence])  # Update summary for this iteration

    print("\n--- Iteration Summary (All Iterations) ---")
    headers = ["Iteration", "Array Compared", "Resulting Common Subsequence"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"longestCommonSubsequence2 = {common_subsequences}")
    return common_subsequences


# <--------------------------------------------------- Week 2, June --------------------------------------------------->
# 2083. Substrings That Begin and End With the Same Letter

# You are given a 0-indexed string 's' consisting of only lowercase English letters.
# Return the number of substrings in 's' that begin and end with the same character.
# A substring is a contiguous non-empty sequence of characters within a string.


def numberOfSubstrings1(s: str) -> int:
    """
    Calculates the number of substrings in a string 's' that begin and end with the same character.

    The function first creates a frequency dictionary 'letter_counts', which uses the defaultdict data structure to
    count the occurrences of each character in the string 's'.
    It then iterates over the values in the 'letter_counts' to calculate the total number of substrings that begin and
    end with the same character.
    The calculation count * (count + 1) // 2 derives from the formula to calculate the sum of the first 'n' integers,
    as each letter can form a substring with every other occurrence of the same letter, including itself.

    The time complexity of this solution is O(n), where 'n' is the length of the string 's'.
    This is because we iterate through the string once to count the occurrences of each character.
    The loop over the values in the 'letter_counts' dictionary has a constant number of iterations (26 at most).
    The space complexity is O(1) because the 'letter_counts' dictionary stores counts for at most 26
    different characters (lowercase English letters), so the space usage doesn't scale with the length of 's'.
    """
    print("\n--- Input Parameters ---")
    print(f"\ts: {s}")

    result = 0
    letter_counts = defaultdict(int)

    print("\n--- First Loop (Counting Letters) ---")
    iteration_data = []
    for letter in s:
        print(f"\tProcessing letter: {letter}")
        letter_counts[letter] += 1
        print(f"\t\t Updated letter_counts: {letter_counts}")
        iteration_data.append([letter, letter_counts.copy()])  # Store data for iteration summary

    print("\n--- Iteration Summary (Letter Counts) ---")
    print(tabulate(iteration_data, headers=["Letter", "Letter Counts"], tablefmt="fancy_grid"))

    print("\n--- Second Loop (Calculating Substrings) ---")
    for count in letter_counts.values():
        print(f"\tProcessing count: {count}")
        print(f"\t\tAdding ({count} * ({count} + 1) // 2 = {count * (count + 1) // 2}) to result ({result})")
        result += count * (count + 1) // 2
        print(f"\t\t Updated result: {result}")

    # Alternative one-liner solution (commented out):
    # return sum(count * (count + 1) // 2 for count in Counter(s).values())

    print("\n--- Function Returning ---")
    print(f"\tResult: {result}")
    return result


# <--------------------------------------------------- Week 3, June --------------------------------------------------->
# 1580. Put Boxes Into the Warehouse II

# Given two arrays of positive integers, `boxes` and `warehouse`, representing the heights of boxes and the heights of
# rooms in a warehouse respectively, determine the maximum number of boxes you can put into the warehouse.
# Boxes can be rearranged, pushed from either side, cannot be stacked, and if a room's height is less than a box's
# height, that box and all behind it will be stopped.

# Introductory problem:
# 1564. Put Boxes Into the Warehouse I

# We are asked to do the same thing as in the previous problem, but this time boxes can only be pushed from the left.


def maxBoxesInWarehouse1(boxes: List[int], warehouse: List[int]) -> int:
    """
    Calculates the maximum number of boxes that can be arranged in the warehouse given the
    heights of the boxes and the warehouse rooms from left to right only.

    Function sorts boxes and tries to fit the largest boxes first.
    If a box fits (its height is <= current warehouse room), the count increases, and it moves to the next room.
    If it doesn't fit, the box is discarded (continue to the next box).

    Time complexity is O(n log n) and space complexity is O(n), where `n` is the number of boxes, as we sort the boxes.
    """
    max_number = 0

    for box in sorted(boxes, reverse=True):
        if max_number < len(warehouse) and box <= warehouse[max_number]:
            max_number += 1

    return max_number


def maxBoxesInWarehouse2(boxes: List[int], warehouse: List[int]) -> int:
    """
    Determines the maximum number of boxes that can be placed in a warehouse.

    This function implements a two-pointer approach to maximize the number of boxes that can fit into the warehouse.
    It sorts the boxes in descending order and attempts to place each box from either the left or right end of the
    warehouse, whichever allows for placement.
    This greedy strategy ensures that larger boxes are given priority, maximizing the total number of boxes placed.

    The algorithm maintains two pointers, one at each end of the warehouse.
    For each box, it checks if it can be placed at either end, moving the respective pointer inward if placement is
    successful.
    This approach efficiently handles the constraint of pushing boxes from either side of the warehouse.

    The time complexity of this solution is O(n log n), where `n` is the number of boxes, due to the sorting operation.
    The warehouse traversal is linear, but dominated by the sorting complexity.
    The space complexity is O(n) due to the built-in sorting operation in Python.
    """
    print("\n--- Input Parameters ---")
    print(f"\tboxes = {boxes}")
    print(f"\twarehouse = {warehouse}")

    max_number = 0
    left_index, right_index = 0, len(warehouse) - 1

    print("\n--- Initialization ---")
    print(f"\tmax_number = {max_number}")
    print(f"\tleft_index = {left_index}")
    print(f"\tright_index = {right_index}")

    print("\n--- Sorting Boxes ---")
    sorted_boxes = sorted(boxes, reverse=True)
    print(f"\tSorted boxes: {sorted_boxes}")

    print("\n--- Main Loop (Placing Boxes) ---")
    iteration_data = []
    for i, box in enumerate(sorted_boxes):
        print(f"\nRoom {i + 1}/{len(sorted_boxes)}:")
        print(f"\tCurrent box height: {box}")
        print(f"\tLeft index {left_index}, Right index {right_index}")
        print(f"\tWarehouse left height: {warehouse[left_index]}, Warehouse right height: {warehouse[right_index]}")

        print(f"\tChecking if box fits from the left or right:")
        if box <= warehouse[left_index]:
            print(f"\t\tBox height {box} <= warehouse room height {warehouse[left_index]}")
            max_number += 1
            action = "Placed on the left, increase left_index"
            iteration_data.append([i + 1, box, f"warehouse[{left_index}] = {warehouse[left_index]}",
                                   f"warehouse[{right_index}] = {warehouse[right_index]}", action, max_number])
            left_index += 1
            print(f"\t\tBox placed on the left")
            print(f"\t\tIncreasing max_number: {max_number}")
            print(f"\t\tIncreasing left_index: {left_index}")

        elif box <= warehouse[right_index]:
            print(f"\t\tBox height {box} <= warehouse room height {warehouse[right_index]}")
            max_number += 1
            action = "Placed on the right, decrease right_index"
            iteration_data.append([i + 1, box, f"warehouse[{left_index}] = {warehouse[left_index]}",
                                   f"warehouse[{right_index}] = {warehouse[right_index]}", action, max_number])
            right_index -= 1
            print("\t\tBox placed on the right")
            print(f"\t\tIncreasing max_number: {max_number}")
            print(f"\t\tDecreasing right_index: {right_index}")
        else:
            print("\t\tBox cannot be placed")
            action = "Not placed"
            iteration_data.append([i + 1, box, f"warehouse[{left_index}] = {warehouse[left_index]}",
                                   f"warehouse[{right_index}] = {warehouse[right_index]}", action, max_number])

        if left_index == right_index:
            print("\n--- Early Termination ---")
            print("\tLeft index met right index. All rooms checked.")
            iteration_data.append(['-', '-', '-', '-', f"Early Termination ({left_index} == {right_index})",
                                   max_number])
            break

    print("\n--- Iteration Summary (Box Placement) ---")
    headers = ["Room", "Box Height", "Left Room Height", "Right Room Height", "Action", "Max Number"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"Maximum number of boxes placed: {max_number}")
    return max_number


# <--------------------------------------------------- Week 4, June --------------------------------------------------->
# 2743. Count Substrings Without Repeating Character

# Given a string s of lowercase English letters, the task is to count the number of substrings (contiguous) that do not
# contain any repeating characters, referred to as "special" substrings.


def numberOfSpecialSubstrings1(s: str) -> int:
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")

    print("\n--- Initialization ---")
    window_chars = set()
    left_index = 0
    n = len(s)
    potential_special_substrings = n * (n + 1) // 2
    print(f"\twindow_chars = {window_chars}")
    print(f"\tleft_index = {left_index}")
    print(f"\tn = {n}")
    print(f"\tpotential_special_substrings = {n} * ({n} + 1) // 2 = {potential_special_substrings}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for right_index, current_char in enumerate(s):
        print(f"\n--- Character {right_index + 1}/{n}: '{current_char}' ---")
        print(f"\tCurrent state: window_chars = {window_chars}, left_index = {left_index}")

        print(f"\tShrinking window if '{current_char}' is already in window:")
        if current_char not in window_chars:
            print(f"\t\tCharacter '{current_char}' not in window.")
        while current_char in window_chars:
            print(f"\t\tCharacter '{current_char}' already in window.")
            print(f"\t\tRemoving '{s[left_index]}' from window_chars")
            window_chars.remove(s[left_index])
            left_index += 1
            print(f"\t\tUpdated: window_chars = {window_chars}, left_index = {left_index}")

        print("\tCounting non-special substrings:")
        non_special = left_index
        print(f"\t\tNon-special substrings ending at current position: {non_special}")
        print(f"\t\tUpdated potential_special_substrings = {potential_special_substrings} - {non_special} = "
              f"{potential_special_substrings - non_special}")
        potential_special_substrings -= non_special

        print("\tUpdating window:")
        print(f"\t\tAdding '{current_char}' to window_chars")
        window_chars.add(current_char)
        print(f"\t\tUpdated window_chars: {window_chars}")

        iteration_data.append(
            [right_index + 1, current_char, left_index, list(window_chars), non_special, potential_special_substrings])

    print("\n--- Iteration Summary ---")
    headers = ["Position", "Character", "Left Index", "Window Chars", "Non-special Count", "Potential Special"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal count of special substrings: {potential_special_substrings}")
    return potential_special_substrings


def numberOfSpecialSubstrings2(s: str) -> int:
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")

    print("\n--- Initialization ---")
    special_substrings_count = 0
    last_occurrence = [-1] * 26
    start_index = 0
    n = len(s)
    print(f"\tspecial_substrings_count = {special_substrings_count}")
    print(f"\tlast_occurrence = {last_occurrence[:5]} + ['...']")
    print(f"\tstart_index = {start_index}")
    print(f"\tn = {n}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for current_index, char in enumerate(s):
        print(f"\n--- Character {current_index + 1}/{n}: '{char}' ---")
        print(f"\tCurrent state: start_index = {start_index}, special_substrings_count = {special_substrings_count}")

        print("\tCalculating character index:")
        char_index = ord(char) - ord('a')
        print(f"\t\tchar_index = ord('{char}') - ord('a') = {ord(char)} - {ord('a')} = {char_index}")

        print("\tChecking if character occurred within current window:")
        if last_occurrence[char_index] >= start_index:
            start_index = last_occurrence[char_index] + 1
            print(f"\t\tCharacter '{char}' occurred within current window.")
            print(f"\t\tUpdating start_index to {start_index}")
        else:
            print(f"\t\tNo adjustment needed. '{char}' not in current window.")

        print("\tCounting special substrings:")
        new_special_substrings = current_index - start_index + 1
        special_substrings_count += new_special_substrings
        print(f"\t\tNew special substrings ending at current position: {new_special_substrings}")
        print(
            f"\t\tCalculation: current_index ({current_index}) - start_index"
            f" ({start_index}) + 1 = {new_special_substrings}")
        print(f"\t\tUpdated special_substrings_count: {special_substrings_count}")

        print("\tUpdating last occurrence:")
        print(f"\t\tSetting last_occurrence[{char_index}] to {current_index}")
        last_occurrence[char_index] = current_index

        iteration_data.append(
            [current_index + 1, char, char_index, start_index, new_special_substrings, special_substrings_count,
             f'{last_occurrence.copy()[:5]} + ["..."]'])

    print("\n--- Iteration Summary ---")
    headers = ["Position", "Character", "Char Index", "Start Index", "New Special", "Total Special", "Last Occurrence"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal count of special substrings: {special_substrings_count}")
    return special_substrings_count


# <--------------------------------------------------- Week 5, June --------------------------------------------------->
# 776. Split BST

# Given a binary search tree (BST) `root` and an integer `target`, the task is to split the BST into two subtrees such
# that the first subtree contains nodes with values less than or equal to `target`, and the second subtree contains
# nodes with values greater than `target`.
# The original tree's structure should be largely preserved.
# The output should be an array of the two subtree roots in order.


def splitBST1(root: Optional[BinaryTreeNode], target: int) -> List[Optional[BinaryTreeNode]]:
    print("\n--- Input Parameters ---")
    print(f"\troot = {root.val if root else None}")
    print(f"\ttarget = {target}")

    # Visualize the initial tree
    BinaryTreeVisualizer.visualize(root, "initial_tree")
    print(f"\nInitial tree visualization saved as 'initial_tree.png'")

    print("\n--- Initialization ---")
    smaller_equal_subtree, greater_subtree = None, None
    path_stack = []
    print(f"\tsmaller_equal_subtree = {smaller_equal_subtree}")
    print(f"\tgreater_subtree = {greater_subtree}")
    print(f"\tpath_stack = {path_stack}")

    print("\n--- Traversing Down the Tree ---")
    current = root
    traversal_data = []
    iteration = 0
    while current:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"\tCurrent node value: {current.val}")
        path_stack.append(current)
        print(f"\tAdded to path_stack: {current.val}")
        print(f"\tChecking condition: current.val ({current.val}) <= target ({target})")
        if current.val <= target:
            print(f"\t\tCondition true. Moving to right child.")
            current = current.right
        else:
            print(f"\t\tCondition false. Moving to left child.")
            current = current.left
        print(f"\tNew current: {current.val if current else None}")
        traversal_data.append([iteration, current.val if current else None,
                               [node.val for node in path_stack],
                               "Right" if current and current.val <= target else "Left"])

    print("\n--- Traversal Summary ---")
    headers = ["Iteration", "Current Node", "Path Stack", "Direction"]
    print(tabulate(traversal_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Reconstructing Subtrees ---")
    reconstruction_data = []
    iteration = 0
    se_subtree_iter = 0
    g_subtree_iter = 0
    while path_stack:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"\tPath stack: {[node.val for node in path_stack]}")
        node = path_stack.pop()
        print(f"\tPopped node: {node.val}")
        print(f"\tChecking condition: node.val ({node.val}) <= target ({target})")
        if node.val <= target:
            print(f"\t\tCondition true. Adding to smaller_equal_subtree.")
            print(f"\t\tPrevious smaller_equal_subtree: {smaller_equal_subtree.val if smaller_equal_subtree else None}")
            node.right = smaller_equal_subtree
            smaller_equal_subtree = node
            is_added_to_smaller_equal_subtree = True
            is_added_to_greater_subtree = False
            print(f"\t\tNew smaller_equal_subtree root: {smaller_equal_subtree.val}")
        else:
            print(f"\t\tCondition false. Adding to greater_subtree.")
            print(f"\t\tPrevious greater_subtree: {greater_subtree.val if greater_subtree else None}")
            node.left = greater_subtree
            greater_subtree = node
            is_added_to_smaller_equal_subtree = False
            is_added_to_greater_subtree = True
            print(f"\t\tNew greater_subtree root: {greater_subtree.val}")
        reconstruction_data.append([iteration, node.val,
                                    "smaller_equal" if node.val <= target else "greater",
                                    smaller_equal_subtree.val if smaller_equal_subtree else None,
                                    greater_subtree.val if greater_subtree else None])

        # Visualize the current state of both subtrees
        if is_added_to_smaller_equal_subtree:
            se_subtree_iter += 1
            BinaryTreeVisualizer.visualize(smaller_equal_subtree, f"smaller_equal_subtree_step_{se_subtree_iter}")
            print(f"\tSmaller/Equal subtree visualization saved as 'smaller_equal_subtree_step_{se_subtree_iter}.png'")
        if is_added_to_greater_subtree:
            g_subtree_iter += 1
            BinaryTreeVisualizer.visualize(greater_subtree, f"greater_subtree_step_{g_subtree_iter}")
            print(f"\tGreater subtree visualization saved as 'greater_subtree_step_{g_subtree_iter}.png'")

    print("\n--- Reconstruction Summary ---")
    headers = ["Iteration", "Processed Node", "Subtree Added To", "Smaller/Equal Subtree Root", "Greater Subtree Root"]
    print(tabulate(reconstruction_data, headers=headers, tablefmt="fancy_grid"))

    return returnSplitBSTResult(smaller_equal_subtree, greater_subtree)


def splitBST2(root: Optional[BinaryTreeNode], target: int) -> List[Optional[BinaryTreeNode]]:
    print("\n--- Input Parameters ---")
    print(f"\troot = {root.val if root else None}")
    print(f"\ttarget = {target}")

    # Visualize the initial tree
    BinaryTreeVisualizer.visualize(root, "initial_tree")
    print(f"\nInitial tree visualization saved as 'initial_tree.png'")

    print("\n--- Initialization ---")
    reconstruction_data = []
    se_subtree_iter = 0
    g_subtree_iter = 0

    def split_tree(node: Optional[BinaryTreeNode], depth: int = 0) -> List[Optional[BinaryTreeNode]]:
        """Helper function to recursively split the tree."""
        nonlocal se_subtree_iter, g_subtree_iter

        print(f"\n{'  ' * depth}--- Recursive call at depth {depth} ---")
        print(f"{'  ' * depth}Current node: {node.val if node else None}")

        if not node:
            print(f"{'  ' * depth}Node is None, returning [None, None]")
            return [None, None]

        print(f"{'  ' * depth}Checking condition: node.val ({node.val}) <= target ({target})")
        if node.val <= target:
            print(f"{'  ' * depth}Condition true. Recursing on right child.")
            smaller_equal_subtree, greater_subtree = split_tree(node.right, depth + 1)
            print(f"\n{'  ' * depth}Returned from right child recursion of current node {node.val}.")
            print(f"{'  ' * depth}Setting node.right to smaller_equal_subtree: {smaller_equal_subtree.val 
                  if smaller_equal_subtree else None}")
            node.right = smaller_equal_subtree
            print(f"{'  ' * depth}Returning [node, greater_subtree] = [{node.val}, {greater_subtree.val if
                  greater_subtree else None}]")

            se_subtree_iter += 1
            BinaryTreeVisualizer.visualize(node, f"smaller_equal_subtree_step_{se_subtree_iter}")
            print(
                f"{'  ' * depth}Smaller/Equal subtree visualization saved as "
                f"'smaller_equal_subtree_step_{se_subtree_iter}.png'")

            reconstruction_data.append([depth, node.val, "smaller_equal",
                                        node.val,
                                        greater_subtree.val if greater_subtree else None])
            return [node, greater_subtree]
        else:
            print(f"{'  ' * depth}Condition false. Recursing on left child.")
            smaller_equal_subtree, greater_subtree = split_tree(node.left, depth + 1)
            print(f"\n{'  ' * depth}Returned from left child recursion of current node {node.val}.")
            print(f"{'  ' * depth}Setting node.left to greater_subtree: "
                  f"{greater_subtree.val if greater_subtree else None}")
            node.left = greater_subtree
            print(f"{'  ' * depth}Returning [smaller_equal_subtree, node] = [{smaller_equal_subtree.val if
                  smaller_equal_subtree else None}, {node.val}]")

            g_subtree_iter += 1
            BinaryTreeVisualizer.visualize(node, f"greater_subtree_step_{g_subtree_iter}")
            print(f"{'  ' * depth}Greater subtree visualization saved as 'greater_subtree_step_{g_subtree_iter}.png'")

            reconstruction_data.append([depth, node.val, "greater",
                                        smaller_equal_subtree.val if smaller_equal_subtree else None,
                                        node.val])
            return [smaller_equal_subtree, node]

    result = split_tree(root)

    print("\n--- Reconstruction Summary ---")
    headers = ["Depth", "Processed Node", "Subtree Added To", "Smaller/Equal Subtree Root", "Greater Subtree Root"]
    print(tabulate(reconstruction_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tsmaller_equal_subtree root: {result[0].val if result[0] else None}")
    print(f"\tgreater_subtree root: {result[1].val if result[1] else None}")

    # Final visualization of both subtrees
    if result[0]:
        BinaryTreeVisualizer.visualize(result[0], "final_smaller_equal_subtree")
        print(f"\nFinal Smaller/Equal subtree visualization saved as 'final_smaller_equal_subtree.png'")
    if result[1]:
        BinaryTreeVisualizer.visualize(result[1], "final_greater_subtree")
        print(f"\nFinal Greater subtree visualization saved as 'final_greater_subtree.png'")

    return result


def splitBST3(root: Optional[BinaryTreeNode], target: int) -> List[Optional[BinaryTreeNode]]:
    print("\n--- Input Parameters ---")
    print(f"\troot = {root.val if root else None}")
    print(f"\ttarget = {target}")

    # Visualize the initial tree
    BinaryTreeVisualizer.visualize(root, "initial_tree")
    print(f"\nInitial tree visualization saved as 'initial_tree.png'")

    print("\n--- Initialization ---")
    dummy_smaller_equal = BinaryTreeNode()
    dummy_greater = BinaryTreeNode()
    current_smaller_equal = dummy_smaller_equal
    current_greater = dummy_greater
    current_node = root

    print(f"\tdummy_smaller_equal = TreeNode()")
    print(f"\tdummy_greater = TreeNode()")
    print(f"\tcurrent_smaller_equal = dummy_smaller_equal")
    print(f"\tcurrent_greater = dummy_greater")
    print(f"\tcurrent_node = {current_node.val if current_node else None}")

    iteration_data = []
    iteration = 0

    while current_node is not None:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"\tCurrent node value: {current_node.val}")
        print(f"\tChecking condition: current_node.val ({current_node.val}) <= target ({target})")

        if current_node.val <= target:
            print(f"\t\tCondition true. Adding to smaller_equal subtree.")
            current_smaller_equal.right = current_node
            current_smaller_equal = current_node
            print(f"\t\tcurrent_smaller_equal updated to: {current_smaller_equal.val}")

            next_node = current_node.right
            current_smaller_equal.right = None  # Cut off right subtree
            print(f"\t\tCut off right subtree. Next node: "
                  f"{next_node.val if next_node else None}")

            BinaryTreeVisualizer.visualize(dummy_smaller_equal, f"smaller_equal_subtree_step_{iteration}")
            print(f"\t\tSmaller/Equal subtree visualization saved as 'smaller_equal_subtree_step_{iteration}.png'")
        else:
            print(f"\t\tCondition false. Adding to greater subtree.")
            current_greater.left = current_node
            current_greater = current_node
            print(f"\t\tcurrent_greater updated to: {current_greater.val}")

            next_node = current_node.left
            current_greater.left = None  # Cut off left subtree
            print(f"\t\tCut off left subtree. Next node: {next_node.val if next_node else None}")

            BinaryTreeVisualizer.visualize(dummy_greater, f"greater_subtree_step_{iteration}")
            print(f"\t\tGreater subtree visualization saved as 'greater_subtree_step_{iteration}.png'")

        current_node = next_node
        print(f"\tMoving to next node: {current_node.val if current_node else None}")

        iteration_data.append([iteration,
                               current_smaller_equal.val if current_smaller_equal != dummy_smaller_equal else None,
                               current_greater.val if current_greater != dummy_greater else None,
                               current_node.val if current_node else None])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Current Smaller/Equal", "Current Greater", "Next Node"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    smaller_equal_subtree = dummy_smaller_equal.right
    greater_subtree = dummy_greater.left

    return returnSplitBSTResult(greater_subtree, smaller_equal_subtree)


def returnSplitBSTResult(greater_subtree, smaller_equal_subtree) -> List[Optional[BinaryTreeNode]]:
    print("\n--- Function Returning ---")
    print(f"\tsmaller_equal_subtree root: {smaller_equal_subtree.val if smaller_equal_subtree else None}")
    print(f"\tgreater_subtree root: {greater_subtree.val if greater_subtree else None}")
    # Final visualization of both subtrees
    if smaller_equal_subtree:
        BinaryTreeVisualizer.visualize(smaller_equal_subtree, "final_smaller_equal_subtree")
        print(f"\nFinal Smaller/Equal subtree visualization saved as 'final_smaller_equal_subtree.png'")
    if greater_subtree:
        BinaryTreeVisualizer.visualize(greater_subtree, "final_greater_subtree")
        print(f"\nFinal Greater subtree visualization saved as 'final_greater_subtree.png'")
    return [smaller_equal_subtree, greater_subtree]


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for Week 1, June
# Expected output: [2, 3, 6]
# longestCommonSubsequence1(arrays=[[2, 3, 6, 8], [1, 2, 3, 5, 6, 7, 10], [2, 3, 4, 6, 9]])
# longestCommonSubsequence2(arrays=[[2, 3, 6, 8], [1, 2, 3, 5, 6, 7, 10], [2, 3, 4, 6, 9]])

# Test cases for Week 2, June
# Expected output: 11
# numberOfSubstrings1(s="abacabd")

# Test cases for Week 3, June
# Expected output: 4
# maxBoxesInWarehouse1(boxes=[5, 4, 3, 2, 1], warehouse=[4, 3, 1, 5, 2, 3])
# maxBoxesInWarehouse2(boxes=[5, 4, 3, 2, 1], warehouse=[4, 3, 1, 5, 2, 3])

# Test cases for Week 4, June
# Expected output: 7
# numberOfSpecialSubstrings1(s="abab")
# numberOfSpecialSubstrings2(s="abab")

# Test cases for Week 5, June
# Expected output: [(TreeNode(2, left=TreeNode(1))),
#                   TreeNode(4, left=TreeNode(3), right=TreeNode(6, left=TreeNode(5), right=TreeNode(7)))]
# splitBST1(TreeNode(val=4, left=TreeNode(val=2, left=TreeNode(val=1), right=TreeNode(val=3)),
#                    right=TreeNode(val=6, left=TreeNode(val=5), right=TreeNode(val=7))), target=2)
# splitBST2(TreeNode(val=4, left=TreeNode(val=2, left=TreeNode(val=1), right=TreeNode(val=3)),
#                    right=TreeNode(val=6, left=TreeNode(val=5), right=TreeNode(val=7))), target=2)
# splitBST3(TreeNode(val=4, left=TreeNode(val=2, left=TreeNode(val=1), right=TreeNode(val=3)),
#                    right=TreeNode(val=6, left=TreeNode(val=5), right=TreeNode(val=7))), target=2)
