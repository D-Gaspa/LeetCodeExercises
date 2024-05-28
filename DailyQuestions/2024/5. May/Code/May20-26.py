from collections import defaultdict
from pprint import pprint
from tabulate import tabulate
from typing import List


# TODO: Add function parameters and return types to the function definitions.
# TODO: Add thorough debug statements to improve example outputs.
# - To be completed:
# - May 23rd
# - May 24th
# - May 25th


# Week 4: May 20th - 26th

# <-------------------------------------------------- May 20th, 2024 -------------------------------------------------->
# 1863. Sum of All Subset XOR Totals

# Given an array nums, return the sum of all XOR totals for every subset of nums.

def subsetXORSum1(nums: List[int]) -> int:
    """
    Calculates the sum of XOR totals over all possible subsets of an integer list.

    This solution uses bit manipulation to generate all subsets and calculate their XOR totals.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """
    n = len(nums)
    result = 0
    table_data = []

    # Print the input parameters for debugging
    print(f"Input Parameters: nums = {nums}")

    print("\n--- Subset XOR Calculation ---")

    # Iterate over all possible subsets using bit representations (2^n subsets) [1 << n = 2^n]
    for i in range(1 << n):
        print(f"\n**Iteration {i}: Bit Representation = {bin(i)[2:].zfill(n)}**")
        xor = 0
        subset = []

        for j in range(n):  # Calculate XOR total for the current subset.
            if i & (1 << j):  # Bitwise trick to check if j-th element is in this subset
                xor ^= nums[j]
                subset.append(nums[j])
                print(f"  - Adding {nums[j]} (index {j}) to subset, XOR = {xor}")
            else:
                print(f"  - Skipping {nums[j]} (index {j}), not in subset")

        table_data.append([i, bin(i)[2:].zfill(n), subset, xor])
        result += xor
        print(f"  - Subset XOR total: {xor}")

    print("\n--- Subsets and XOR Totals ---")
    print(tabulate(table_data, headers=["Iteration (i)", "Bit Rep.", "Subset", "XOR Total"], tablefmt="fancy_grid"))

    print(f"\n--- Final Sum of XOR Totals: {result} ---")
    return result


def subsetXORSum2(nums: List[int]) -> int:
    """
    Calculates the sum of XOR totals over all possible subsets of an integer list.

    This solution uses a bitwise OR operation to combine all numbers and then calculates the sum of XOR totals.
    The time complexity of this solution is O(n) where n is the length of the input list.
    """
    combined_bits = 0
    table_data = []

    # Print the input parameters for debugging
    print(f"Input Parameters: nums = {nums}")

    print("\n--- Subset XOR Calculation (Bitwise OR Approach) ---")
    print("Num | Bit Representation | Combined Bits")
    print("----|-------------------|---------------")

    for i, num in enumerate(nums):
        combined_bits |= num
        table_data.append([num, bin(num)[2:].zfill(8), bin(combined_bits)[2:].zfill(8)])
        print(f"{num} | {bin(num)[2:].zfill(8)}   | {bin(combined_bits)[2:].zfill(8)}")

    print("\n--- Combined Bits ---")
    print(tabulate(table_data, headers=["Num", "Bit Rep.", "Combined Bits"]))

    n = len(nums)
    result = combined_bits * (1 << (n - 1))

    print(f"\nCombined Bits: {combined_bits} (decimal), {bin(combined_bits)} (binary)")
    print(f"Number of subsets: 2 ^ (n - 1) = 2 ^ {n - 1} = {1 << (n - 1)}")
    print(f"\n--- Final Sum of XOR Totals: {result} ---")
    return result


# <-------------------------------------------------- May 21st, 2024 -------------------------------------------------->
# 78. Subsets

# Given an integer array nums of unique elements, return all possible subsets (the power set).


def subsets1(nums: List[int]) -> List[List[int]]:
    """
    Generates all possible subsets of an integer list.

    This solution uses bit manipulation to generate all subsets.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """
    n = len(nums)
    result = []
    table_data = []

    # Print the input parameters for debugging
    print(f"Input Parameters: nums = {nums}")

    print("\n--- Subset Generation ---")

    for i in range(1 << n):
        subset = []
        included_elements = []

        for j in range(n):
            if i & (1 << j):  # Bitwise trick to check if j-th element is in this subset
                subset.append(nums[j])
                included_elements.append(j)

        result.append(subset)
        table_data.append([i, bin(i)[2:].zfill(n), included_elements, subset])

    print(tabulate(table_data, headers=["Iteration (i)", "Bit Rep.", "Included Indices", "Subset"],
                   tablefmt="fancy_grid"))

    print("\n--- Final Subsets ---")
    pprint(result)
    return result


def subsets2(nums: List[int]) -> List[List[int]]:
    """
    Generates all possible subsets of an integer list.

    This solution uses backtracking to generate all subsets.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """
    # Print the input parameters for debugging
    print(f"Input Parameters: nums = {nums}")

    def backtrack(start: int, current_subset: List[int], depth: int = 0):
        """Backtracking function to generate all possible subsets."""
        indent = "  " * depth  # For visual indentation based on recursion depth
        print(f"{indent}Exploring from index {start} with current subset:", current_subset)

        result.append(current_subset[:])
        print(f"{indent}Appending subset: {current_subset}")

        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            print(f"{indent}Adding element {nums[i]} (index {i}) to subset: {current_subset}")
            backtrack(i + 1, current_subset, depth + 2)  # Recursive call with increased depth
            current_subset.pop()
            print(f"{indent}Backtracking, removing {nums[i]} (index {i}) from subset: {current_subset}")

    result = []
    backtrack(0, [])

    print("\n--- All Subsets ---")
    pprint(result)
    return result


def subsets3(nums: List[int]) -> List[List[int]]:
    """
    Generates all possible subsets of an integer list.

    This solution uses an iterative approach to generate all subsets.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """
    result = [[]]

    # Print the input parameters for debugging
    print(f"Input Parameters: nums = {nums}")

    print("\n--- Subset Generation ---")

    for num in nums:
        new_subsets = [curr + [num] for curr in result]

        print(tabulate([[num, result, new_subsets]], headers=["Num", "Existing Subsets", "New Subsets"]))

        result += new_subsets  # Add new subsets to the result

    print("\n--- All Subsets ---")
    pprint(result)  # Use pprint for nicely formatted output
    return result


# <-------------------------------------------------- May 22nd, 2024 -------------------------------------------------->
# 131. Palindrome Partitioning

# Given a string 's', partition 's' such that every substring of the partition is a palindrome
# Return all possible palindrome partitions of 's'.


def partition1(s: str) -> List[List[str]]:
    """
    Generates all possible palindrome partitions of a string.

    This solution uses backtracking to generate all palindrome partitions.
    The time complexity of this solution is O(n * 2^n) where n is the length of the input string.
    """
    # Print the input parameters for debugging
    print(f"Input Parameters: s = {s}")

    print("\n--- Palindrome Partitioning ---")

    def is_palindrome(s, start_index, end_index, depth=0):
        """Checks if a substring is a palindrome."""
        indent = "  " * depth  # For visual indentation based on recursion depth
        print(f"{indent}Checking if '{s[start_index:end_index + 1]}' is a palindrome...")
        while start_index < end_index:
            if s[start_index] != s[end_index]:
                print(f"{indent}Not a palindrome. Characters at indices {start_index} and {end_index} don't match.")
                return False
            start_index += 1
            end_index -= 1
        print(f"{indent}Palindrome!")
        return True

    def backtrack(start_index: int, current_partition: List[str], depth: int = 0):
        """Backtracking function to generate all palindrome partitions."""
        indent = "  " * depth  # For visual indentation based on recursion depth
        print(f"{indent}Backtracking from index {start_index} with partition:", current_partition)

        if start_index == len(s):
            result.append(current_partition[:])
            print(f"{indent}Found a valid partition: {current_partition}")
            return

        for end_index in range(start_index, len(s)):
            if is_palindrome(s, start_index, end_index, depth + 1):  # Pass depth to is_palindrome
                current_partition.append(s[start_index:end_index + 1])
                print(f"{indent}Adding '{s[start_index:end_index + 1]}' to partition: {current_partition}")
                backtrack(end_index + 1, current_partition, depth + 2)  # Recursive call with increased depth
                current_partition.pop()
                print(f"{indent}Backtracking, removing '{s[start_index:end_index + 1]}' "
                      f"from partition: {current_partition}")
            else:
                print(f"{indent}Skipping '{s[start_index:end_index + 1]}', not a palindrome.")

    result = []
    backtrack(0, [])

    print("\n--- All Palindrome Partitions ---")
    table = tabulate(enumerate(result), headers=["Index", "Partition"], tablefmt="fancy_grid")
    print(table)
    return result


def partition2(s: str) -> List[List[str]]:
    """
    Generates all possible palindrome partitions of a string.

    This solution uses dynamic programming and backtracking to generate all palindrome partitions.
    The time complexity of this solution is O(n * 2^n) where n is the length of the input string.
    """
    # Print the input parameters for debugging
    print(f"Input Parameters: s = {s}")

    n = len(s)
    is_palindrome = [[False] * n for _ in range(n)]  # DP table to store palindrome substrings
    table_data = []

    print("\n--- Dynamic Programming: Palindrome Substrings ---")

    for start_index in range(n - 1, -1, -1):
        for end_index in range(start_index, n):
            if s[start_index] == s[end_index] and (
                    end_index - start_index < 2 or is_palindrome[start_index + 1][end_index - 1]):
                is_palindrome[start_index][end_index] = True
                table_data.append([s[start_index:end_index + 1], start_index, end_index, True])
            else:
                table_data.append([s[start_index:end_index + 1], start_index, end_index, False])

    print(tabulate(table_data, headers=["Substring", "Start", "End", "Is Palindrome?"], tablefmt="fancy_grid"))

    print("\n--- Backtracking: Generating Palindrome Partitions ---")

    def backtrack(start_index: int, current_partition: List[str], depth: int = 0):
        """Backtracking function to generate all palindrome partitions."""
        indent = "  " * depth  # For visual indentation based on recursion depth
        print(f"{indent}Backtracking from index {start_index} with partition: {current_partition}")

        if start_index == n:
            result.append(current_partition[:])
            print(f"{indent}Found a valid partition: {current_partition}")
            return

        for end_index in range(start_index, n):
            if is_palindrome[start_index][end_index]:
                current_partition.append(s[start_index:end_index + 1])
                print(f"{indent}Adding '{s[start_index:end_index + 1]}' to partition: {current_partition}")
                backtrack(end_index + 1, current_partition, depth + 1)
                current_partition.pop()
                print(
                    f"{indent}Backtracking, removing '{s[start_index:end_index + 1]}' "
                    f"from partition: {current_partition}")
            else:
                print(f"{indent}Skipping '{s[start_index:end_index + 1]}', not a palindrome.")

    result = []
    backtrack(0, [])

    print("\n--- All Palindrome Partitions ---")
    table = tabulate(enumerate(result), headers=["Index", "Partition"], tablefmt="fancy_grid")
    print(table)
    return result


# <-------------------------------------------------- May 23rd, 2024 -------------------------------------------------->
# 2597. The Number of Beautiful Subsets

# Count the number of non-empty subsets in an array of positive integers where no two elements have an absolute
# difference equal to a given positive integer k.


def beautifulSubsets1(nums: List[int], k: int) -> int:
    """
    Counts the number of beautiful subsets in an integer list.

    This solution uses backtracking to generate all subsets and check if they are beautiful.
    The time complexity of this solution is O(2^n) where n is the length of the input list.
    """
    n = len(nums)
    beautiful_count = 0
    current_subset = []
    table_data = []

    # Print the input parameters for debugging
    print(f"Input Parameters: nums = {nums}, k = {k}")

    print("\n--- Backtracking: Generating Beautiful Subsets ---")

    def backtrack(start_index: int, depth=0):
        """Backtracking function to generate all subsets and check if they are beautiful."""
        nonlocal beautiful_count  # Access the outer variable

        indent = "  " * depth  # For visual indentation based on recursion depth
        print(f"{indent}Exploring from index {start_index} with subset: {current_subset}")

        # Base case: a subset is found, check if it's beautiful
        if len(current_subset) > 0:
            for index in range(len(current_subset) - 1):
                if abs(current_subset[index] - current_subset[-1]) == k:
                    print(f"{indent}Not beautiful (difference of {k} found). Pruning.")
                    return
            beautiful_count += 1
            print(f"{indent}Beautiful subset found! Count updated: {beautiful_count}")

        table_data.append([depth, start_index, current_subset[:], beautiful_count])  # Record the exploration steps

        # Recursive case: try adding each remaining element
        for index in range(start_index, n):
            current_subset.append(nums[index])
            print(f"{indent}Adding element {nums[index]} (index {index}) to subset: {current_subset}")
            backtrack(index + 1, depth + 1)
            current_subset.pop()
            print(f"{indent}Backtracking, removing {nums[index]} (index {index}) from subset: {current_subset}")

    backtrack(0)  # Start backtracking from the beginning

    print("\n--- Exploration Steps ---")
    print(tabulate(table_data, headers=["Depth", "Start Index", "Subset", "Beautiful Count"], tablefmt="fancy_grid"))

    print(f"\n--- Final Count of Beautiful Subsets: {beautiful_count} ---")

    return beautiful_count


def beautifulSubsets2(nums: List[int], k: int) -> int:
    """
    Counts the number of beautiful subsets in the given list of numbers.

    This solution uses dynamic programming and memoization to count beautiful subsets.
    The time complexity of this solution is O(2^n) where n is the length of the input list.
    """
    total_count = 1
    remainder_groups = defaultdict(lambda: defaultdict(int))
    memo = {}
    table_data = []

    # Print the input parameters for debugging
    print(f"Input Parameters: nums = {nums}, k = {k}")

    def count_beautiful_subsets(subsets: List[tuple], current_index: int, depth=0) -> int:
        """Recursively counts beautiful subsets starting from the current index."""
        indent = "  " * depth  # For visual indentation based on recursion depth
        print(f"{indent}Exploring from index {current_index} with remaining subsets:")

        print(f"{indent}{subsets[current_index:]}")

        if current_index == len(subsets):
            print(f"{indent}Base case reached. Empty subset is beautiful. Returning 1.")
            return 1

        key = (current_index, tuple(subsets[current_index:]))
        if key in memo:
            print(f"{indent}Memoized result found: {memo[key]}")
            return memo[key]

        # Divide the subsets into two groups, excluding and including the current number, then count them
        print(f"{indent}Excluding current number {subsets[current_index]}")
        exclude_count = count_beautiful_subsets(subsets, current_index + 1, depth + 1)

        include_count = (1 << subsets[current_index][1]) - 1
        print(f"{indent}Including current number {subsets[current_index]}")

        if current_index + 1 < len(subsets) and subsets[current_index + 1][0] - subsets[current_index][0] == k:
            print(f"{indent}  Skipping next number (difference of k).")
            include_count *= count_beautiful_subsets(subsets, current_index + 2, depth + 1)
        else:
            print(f"{indent}  Including next number.")
            include_count *= count_beautiful_subsets(subsets, current_index + 1, depth + 1)

        total_count = exclude_count + include_count
        memo[key] = total_count
        print(f"{indent}Memoizing result: {total_count} for key: {key}")

        return total_count

    print("\n--- Grouping Numbers by Remainders (Modulo k) ---")
    for num in nums:
        remainder_groups[num % k][num] += 1
        table_data.append([num, num % k, remainder_groups[num % k][num]])  # Add data for table
    print(tabulate(table_data, headers=["Number", "Remainder", "Count"], tablefmt="fancy_grid"))

    table_data = []
    print("\n--- Calculating Beautiful Subsets for Each Remainder Group ---")
    for remainder, group in remainder_groups.items():
        print(f"\nRemainder Group: {remainder}")
        sorted_subsets = sorted(group.items())
        pprint(sorted_subsets)
        group_count = count_beautiful_subsets(sorted_subsets, 0)
        total_count *= group_count
        table_data.append([remainder, group_count, total_count])

    print("\n--- Beautiful Subset Counts per Remainder Group ---")
    print(tabulate(table_data, headers=["Remainder", "Group Count", "Total Count (Cumulative)"], tablefmt="fancy_grid"))

    print(f"\nFinal total count (excluding empty set): {total_count - 1}")
    return total_count - 1


def beautifulSubsets3(nums: List[int], k: int) -> int:
    """
    Counts the number of beautiful subsets in the given list of numbers.

    This solution uses dynamic programming with an iterative approach to count beautiful subsets.
    The time complexity of this solution is O(n log n) where n is the length of the input list.
    """
    beautiful_count = 1
    remainder_groups = defaultdict(dict)
    table_data = []

    # Print the input parameters for debugging
    print(f"Input Parameters: nums = {nums}, k = {k}")

    print("\n--- Grouping Numbers by Remainders (Modulo k) ---")
    for num in nums:
        remainder_groups[num % k][num] = remainder_groups[num % k].get(num, 0) + 1
        table_data.append([num, num % k, remainder_groups[num % k][num]])  # Add data for table
    print(tabulate(table_data, headers=["Number", "Remainder", "Count"], tablefmt="fancy_grid"))

    print("\n--- Iterating Over Remainder Groups ---")
    group_table_data = []
    for remainder, group in remainder_groups.items():
        print(f"\n**Remainder Group: {remainder}**")
        prev_num = -k  # Initialize with a number guaranteed not to be in nums
        count_excluding_prev, exclude_count = 1, 1

        # Sort the group items for consistent iteration
        sorted_group = sorted(group.items())
        print(tabulate(sorted_group, headers=["Number", "Frequency"], tablefmt="fancy_grid"))

        for num, frequency in sorted_group:
            include_count = (1 << frequency) - 1  # Count subsets with the current number
            print(f"- Number: {num}, Frequency: {frequency}, Include Count: {include_count}")

            if num - prev_num == k:
                print(f"- Difference with previous number is k. "
                      f"Multiplying include_count by count_excluding_prev ({count_excluding_prev}).")
                include_count *= count_excluding_prev
            else:
                print(
                    f"- Difference with previous number is not k. "
                    f"Multiplying include_count by exclude_count ({exclude_count}).")
                include_count *= exclude_count

            # Update counts for the next iteration
            count_excluding_prev, exclude_count = exclude_count, exclude_count + include_count
            prev_num = num

            # Add data for the group table
            group_table_data.append([remainder, num, frequency, include_count, exclude_count, count_excluding_prev])

        beautiful_count *= exclude_count  # Update the overall count

        print("\n--- Subset Counts for Current Group ---")
        print(tabulate(group_table_data, headers=["Remainder", "Number", "Frequency", "Include Count", "Exclude Count",
                                                  "Count Excluding Prev"], tablefmt="fancy_grid"))

    print(f"\nFinal total count (excluding empty set): {beautiful_count - 1}")
    return beautiful_count - 1  # Exclude the empty subset


# <-------------------------------------------------- May 24th, 2024 -------------------------------------------------->

# 1255. Maximum Score Words Formed by Letters

# Given a list of words, list of letters, and score of letters, find the maximum score of any valid set of words you can
# form using the given letters (words[i] cannot be used two or more times).

def maxScoreWords1(words: List[str], letters: List[str], score: List[int]) -> int:
    """
    Finds the maximum score of any valid set of words that can be formed using the given letters.

    This solution uses backtracking to generate all valid sets of words and calculate their scores.
    The time complexity of this solution is O(2^n * L) where n is the number of words, and L is the average length of
    the words in the input list.
    """

    def backtrack(start: int, current_score: int, available_letters: List[int]):
        """Backtracking function to generate all valid sets of words and calculate their scores."""
        nonlocal max_score  # Access the outer variable

        max_score = max(max_score, current_score)  # Update the maximum score

        for i in range(start, len(words)):
            word = words[i]
            word_score = 0
            valid_word = True

            for char in word:  # Calculate the score of the current word and check if it's valid
                available_letters[ord(char) - ord('a')] -= 1
                word_score += score[ord(char) - ord('a')]

                if available_letters[ord(char) - ord('a')] < 0:
                    valid_word = False  # The word is not valid if a letter count becomes negative

            if valid_word:  # If the word is valid, continue backtracking with the next word
                backtrack(i + 1, current_score + word_score, available_letters)

            # Backtrack by restoring the letter counts
            for char in word:
                available_letters[ord(char) - ord('a')] += 1

    max_score = 0
    available_letters = [letters.count(chr(i + ord('a'))) for i in range(26)]  # Compute the count of each letter
    backtrack(0, 0, available_letters)  # Start backtracking from the beginning
    return max_score


def maxScoreWords2(words: List[str], letters: List[str], score: List[int]) -> int:
    """
    Finds the maximum score of any valid set of words that can be formed using the given letters.

    This solution uses backtracking with memoization to generate all valid sets of words and calculate their scores.
    The time complexity of this solution is O(2^n * L) where n is the number of words, and L is the average length of
    the words in the input list.
    """

    n = len(words)
    max_scores_by_word_idx = [0] * n  # Memoized max scores for each starting word
    word_scores = [0] * n
    available_letter_counts = [0] * 26

    # Precalculate word scores and available letter counts
    for letter in letters:
        available_letter_counts[ord(letter) - ord('a')] += 1
    for i, word in enumerate(words):
        for char in word:
            char_code = ord(char) - ord('a')
            if available_letter_counts[char_code] > 0:
                word_scores[i] += score[char_code]
            else:  # Word cannot be formed with available letters
                word_scores[i] = -1
                break

    visited_states = set()  # Track combinations to avoid redundancy

    def backtrack(word_index: int, current_score: int, used_words_bitmask: int, letter_counts: List[int]):
        """Recursively explore valid word combinations and update max_scores_by_word_idx."""

        # Mark the current word as used (set the corresponding bit in the bitmask)
        used_words_bitmask |= (1 << word_index)
        if word_index == n:  # Base case: reached the end of the word list
            return
        if used_words_bitmask in visited_states:  # Skip redundant exploration
            return
        if word_scores[word_index] == -1:  # Skip words that cannot be formed
            return

        # Check if the current combination is valid (enough letters available)
        for i in range(26):
            if letter_counts[i] > available_letter_counts[i]:
                return

        visited_states.add(used_words_bitmask)  # Mark the current combination as visited
        current_score += word_scores[word_index]
        max_scores_by_word_idx[word_index] = max(max_scores_by_word_idx[word_index], current_score)

        # Recursively explore the next word combinations
        for next_word_index in range(word_index + 1, n):
            for char in words[next_word_index]:  # Update letter counts for the next word
                letter_counts[ord(char) - ord('a')] += 1
            backtrack(next_word_index, current_score, used_words_bitmask, letter_counts)  # Recursive call
            for char in words[next_word_index]:  # Restore letter counts before moving to the next word
                letter_counts[ord(char) - ord('a')] -= 1

    overall_max_score = 0

    # Start backtracking from each word index and update the overall maximum score
    for idx in range(n):
        available_letters = [0] * 26
        for char in words[idx]:  # Count letters in the starting word
            available_letters[ord(char) - ord('a')] += 1

        used_words_bitmask = 1 << idx  # Mark the starting word as used
        backtrack(idx, 0, used_words_bitmask, available_letters)
        overall_max_score = max(overall_max_score, max_scores_by_word_idx[idx])

    return overall_max_score


# <-------------------------------------------------- May 25th, 2024 -------------------------------------------------->

# 140. Word Break II

# Given a string s and a dictionary of strings word_dict, add spaces in s to construct a sentence where each word is a
# valid dictionary word.
# Return all such possible sentences in any order.


def wordBreak1(s: str, word_dict: List[str]) -> List[str]:
    """
    Returns all possible sentences formed by adding spaces to a string to construct valid dictionary words.

    This solution uses backtracking with memoization to generate all possible sentences.
    The time complexity of this solution is O(2^n) where n is the length of the input string.
    """

    def backtrack(start: int):
        """Backtracking function to generate all possible sentences."""
        if start == len(s):  # Base case: reached the end of the string
            return [[]]

        if start in memo:  # Return memoized result if available
            return memo[start]

        sentences = []
        for end in range(start + 1, len(s) + 1):  # Try all possible substrings starting from the current index
            word = s[start:end]
            if word in word_dict:
                # Recursively backtrack from the next index and append the current word to the sentences
                for sentence in backtrack(end):
                    sentences.append([word] + sentence)

        memo[start] = sentences
        return sentences

    memo = {}
    sentences = backtrack(0)  # Start backtracking from the beginning
    return [' '.join(words) for words in sentences]  # Convert the list of words to sentences


def wordBreak2(s: str, word_dict: List[str]) -> List[str]:
    """
    Returns all possible sentences formed by adding spaces to a string to construct valid dictionary words.

    This solution uses iterative dynamic programming to generate all possible sentences.
    The time complexity of this solution is O(n^2) where n is the length of the input string.
    """
    n = len(s)
    dp_table = [[] for _ in range(n + 1)]
    dp_table[n] = [[]]  # Base case: empty string has an empty list of sentences

    # Iterate backwards through the string to build up sentences
    for start in range(n - 1, -1, -1):
        for end in range(start + 1, n + 1):  # Try all possible substrings starting from the current index
            word = s[start:end]
            if word in word_dict:
                # Append the current word to all possible sentences starting from the end index
                for sentence in dp_table[end]:
                    dp_table[start].append([word] + sentence)

    return [' '.join(words) for words in dp_table[0]]  # Convert the list of words to sentences


def wordBreak3(s: str, word_dict: List[str]) -> List[str]:
    """
    Returns all possible sentences formed by adding spaces to a string to construct valid dictionary words.

    This solution uses the Trie data structure and iterative dynamic programming to generate all possible sentences.
    The time complexity of this solution is O(n^2) where n is the length of the input string.
    """

    class TrieNode:
        """A node in a Trie data structure."""

        def __init__(self):
            """Initializes a Trie node."""
            self.children = {}  # Dictionary to store child nodes
            self.is_word_end = False  # Flag to indicate the end of a valid word

    def build_trie(words: List[str]) -> TrieNode:
        """Builds a trie from a list of words."""
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                node = node.children.setdefault(char, TrieNode())  # Create child if not exists
            node.is_word_end = True
        return root

    n = len(s)
    sentences_at_index = [[] for _ in range(n + 1)]  # DP table: sentences found at each index
    sentences_at_index[n] = [[]]  # Base case: empty string has an empty list of sentences
    trie_root = build_trie(word_dict)

    # Iterate backwards, building sentences from the end of the string
    for start in range(n - 1, -1, -1):
        node = trie_root
        for end in range(start, n):
            char = s[end]
            if char not in node.children:
                break  # The current prefix is not a valid word start
            node = node.children[char]
            if node.is_word_end:
                # Add valid words found at 'end' to sentences starting at 'start'
                for sentence in sentences_at_index[end + 1]:
                    sentences_at_index[start].append([s[start:end + 1]] + sentence)

    return [' '.join(words) for words in sentences_at_index[0]]  # Convert the list of words to sentences


# <-------------------------------------------------- May 26th, 2024 -------------------------------------------------->

# 552. Student Attendance Record II

# Given an integer n, return the number of all possible attendance records with length 'n' that make a student
# eligible for an award.
# The attendance record may only contain the characters 'A' (absent), 'L' (late), or 'P' (present).
# A student is eligible for an award if they meet the following criteria:
# - No more than one 'A' (absence) in the entire attendance record.
# - No more than two continuous 'L' (late) days.
# - All remaining days are 'P' (present).
# The function should return the number of possible attendance records, modulo 10^9 + 7 due to the potential large
# result.


def checkRecord1(n: int) -> int:
    """
    Returns the number of all possible attendance records that make a student eligible for an award.

    This solution uses a top-down dynamic programming approach with memoization to count valid attendance records.
    The time complexity of this solution is O(n) where n is the length of the attendance record.
    """
    MOD = 10 ** 9 + 7  # Modulo to prevent integer overflow

    # Print the input parameters for debugging
    print(f"Input Parameters: n = {n}")

    def count_valid_records(day: int, absences: int, consecutive_late_days: int) -> int:
        """Recursively counts valid attendance records starting from the current day."""
        print(f"\n--- Day: {day}, Absences: {absences}, Late Days: {consecutive_late_days} ---")

        if day == n:
            print("Reached end of attendance record (Base Case). Returning 1.")
            return 1  # Base case: reached the end of the attendance record

        # Return the memoized result if available
        if (day, absences, consecutive_late_days) in memo:
            print(f"Memoized result found: {memo[(day, absences, consecutive_late_days)]}")
            return memo[(day, absences, consecutive_late_days)]

        total_valid_records = 0

        # Case 1: Student is absent today
        if absences < 1:
            print("Exploring absence for today...")
            total_valid_records += count_valid_records(day + 1, absences + 1, 0) % MOD
            print(f"Back from exploring absence. Valid records so far: {total_valid_records}")

        # Case 2: Student is late today
        if consecutive_late_days < 2:
            print("Exploring being late today...")
            total_valid_records += count_valid_records(day + 1, absences, consecutive_late_days + 1) % MOD
            print(f"Back from exploring being late. Valid records so far: {total_valid_records}")

        # Case 3: Student is present today
        print("Exploring being present today...")
        total_valid_records += count_valid_records(day + 1, absences, 0) % MOD
        print(f"Back from exploring being present. Valid records so far: {total_valid_records}")

        # Store the total count in memoization dictionary for future use, then return it
        memo[(day, absences, consecutive_late_days)] = total_valid_records % MOD
        print(f"Memoizing result: {total_valid_records % MOD}")
        return total_valid_records % MOD

    memo = {}
    result = count_valid_records(0, 0, 0)  # Start counting from the first day

    print("\n--- Final Memoization Table ---")
    pprint(memo)  # Print the final memoization table nicely formatted

    return result  # Return the final count of valid attendance records


def checkRecord2(n: int) -> int:
    """
    Returns the number of all possible attendance records with length 'n' that make a student eligible for an award.

    This solution uses an iterative bottom-up dynamic programming approach to count valid attendance records.
    The time complexity of this solution is O(n) where n is the length of the attendance record.
    """
    MOD = 10 ** 9 + 7  # Modulo to prevent integer overflow

    # Print the input parameters for debugging
    print(f"Input Parameters: n = {n}")

    dp_current_state = [[0] * 3 for _ in range(2)]  # Current day's states (absences: 0 or 1)
    dp_next_state = [[0] * 3 for _ in range(2)]  # Next day's states (initialized to 0)

    dp_current_state[0][0] = 1  # Base case: 1 valid empty record (no absences, no late days)

    for day in range(1, n + 1):
        print(f"\n--- Day {day} ---")
        print("Current State (before update):")
        pprint(dp_current_state)

        for absences in range(2):
            for consecutive_late_days in range(3):
                print(f"- Absences: {absences}, Late Days: {consecutive_late_days}")

                # Case 1: Add a 'P' (present) day to any existing valid record
                dp_next_state[absences][0] = (dp_next_state[absences][0] +
                                              dp_current_state[absences][consecutive_late_days]) % MOD
                print("  Adding 'P':", dp_next_state[absences][0])

                # Case 2: Add an 'A' (absence) day to a record with no absences
                if absences < 1:
                    dp_next_state[absences + 1][0] = (dp_next_state[absences + 1][0] +
                                                      dp_current_state[absences][consecutive_late_days]) % MOD
                    print("  Adding 'A':", dp_next_state[absences + 1][0])

                # Case 3: Add an 'L' (late) day to a record with less than 2 consecutive late days
                if consecutive_late_days < 2:
                    dp_next_state[absences][consecutive_late_days + 1] = (
                            (dp_next_state[absences][consecutive_late_days + 1] +
                             dp_current_state[absences][consecutive_late_days]) % MOD)
                    print("  Adding 'L':", dp_next_state[absences][consecutive_late_days + 1])

        # Move to the next day by swapping the current and next states
        dp_current_state, dp_next_state = dp_next_state, dp_current_state
        print("Next State (for next iteration):")
        pprint(dp_current_state)

        # Reset the next state to 0 for the next iteration
        dp_next_state = [[0] * 3 for _ in range(2)]

    # Sum all valid records for both cases of having 0 or 1 absences in the final current_state
    return sum(dp_current_state[absences][late_days] for absences in range(2) for late_days in range(3)) % MOD


def checkRecord3(n: int) -> int:
    """
    Returns the number of all possible attendance records with length 'n' that make a student eligible for an award.

    This solution uses matrix exponentiation to count valid attendance records.
    The time complexity of this solution is O(log n) where n is the length of the attendance record.
    """
    MOD = 10 ** 9 + 7  # Modulo to prevent integer overflow

    # Print the input parameters for debugging
    print(f"Input Parameters: n = {n}")

    # Define the transition matrix based on possible attendance states
    # Each state represents (absences, consecutive late days):
    #  - State 0: (0, 0)  # No absences, no late days
    #  - State 1: (0, 1)  # No absences, 1 late day
    #  - State 2: (0, 2)  # No absences, 2 consecutive late days
    #  - State 3: (1, 0)  # 1 absence, no late days
    #  - State 4: (1, 1)  # 1 absence, 1 late day
    #  - State 5: (1, 2)  # 1 absence, 2 consecutive late days
    transition_matrix = [
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0]
    ]

    print("\n--- Transition Matrix ---")
    pprint(transition_matrix)

    def matrix_multiply(matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
        """Multiplies two matrices and returns the result."""

        print("\n--- Matrix Multiplication ---")
        pprint(matrix1)
        print("x")
        pprint(matrix2)

        rows_A, cols_A = len(matrix1), len(matrix1[0])
        rows_B, cols_B = len(matrix2), len(matrix2[0])

        result = [[0] * cols_B for _ in range(rows_A)]

        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += matrix1[i][k] * matrix2[k][j] % MOD

        print("Result:")
        pprint(result)

        return result

    def matrix_power(matrix: List[List[int]], power: int) -> List[List[int]]:
        """Calculates the matrix exponentiation of a matrix."""
        print(f"\n--- Matrix Power (Power = {power}) ---")
        if power == 1:  # Base case: return the matrix
            print("Base Case Reached. Returning the matrix:")
            pprint(matrix)
            return matrix
        if power % 2 == 0:  # If the power is even, calculate the square of the matrix
            print("Power is even. Calculating half power...")
            half_power = matrix_power(matrix, power // 2)
            result = matrix_multiply(half_power, half_power)
            print("Squared Result:")
            pprint(result)
            return result
        print("Power is odd. Recursive call to matrix_power...")
        result = matrix_multiply(matrix, matrix_power(matrix, power - 1))
        print("Result:")
        pprint(result)
        return result

    # Calculate the matrix power for the given attendance record length
    print(f"\n--- Calculating Matrix Power for n = {n} ---")
    result_matrix = matrix_power(transition_matrix, n)

    print("\n--- Result Matrix ---")
    pprint(result_matrix)

    # The first row of the result matrix represents valid records ending in each state.
    # Sum those to get the total number of valid records.
    return sum(result_matrix[0]) % MOD


# <-------------------------------------------------- Test Cases -------------------------------------------------->

# May 20th
nums = [5, 1, 6]
# subsetXORSum1(nums)  # Expected output: 28
# subsetXORSum2(nums)  # Expected output: 28

# May 21st
nums_2 = [1, 2, 3]
# subsets1(nums_2)  # Expected output: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
# subsets2(nums_2)  # Expected output: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
# subsets3(nums_2)  # Expected output: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]

# May 22nd
s = "aab"
# partition1(s)  # Expected output: [["a", "a", "b"], ["aa", "b"]]
# partition2(s)  # Expected output: [["a", "a", "b"], ["aa", "b"]]

# May 23rd
nums_3 = [2, 4, 6]
k = 2
# beautifulSubsets1(nums_3, k)  # Expected output: 4
# beautifulSubsets2(nums_3, k)  # Expected output: 4
# beautifulSubsets3(nums_3, k)  # Expected output: 4

# May 24th
# ...

# May 25th
# ...

# May 26th
n = 2
# checkRecord1(n)  # Expected output: 8
# checkRecord2(n)  # Expected output: 8
# checkRecord3(n)  # Expected output: 8
