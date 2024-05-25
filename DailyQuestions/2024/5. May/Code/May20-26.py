from collections import defaultdict
from typing import List


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
    ans = 0

    # Iterate over all possible subsets using bit representations (2^n subsets) [1 << n = 2^n]
    for i in range(1 << n):
        xor = 0

        for j in range(n):  # Calculate XOR total for the current subset.
            if i & (1 << j):  # Bitwise trick to check if j-th element is in this subset
                xor ^= nums[j]
        ans += xor

    return ans


def subsetXORSum2(nums: List[int]) -> int:
    """Calculates the sum of XOR totals over all possible subsets of an integer list.

    This solution uses a bitwise OR operation to combine all numbers and then calculates the sum of XOR totals.
    The time complexity of this solution is O(n) where n is the length of the input list.
    """
    combined_bits = 0  # Accumulate bitwise OR of all numbers

    for num in nums:
        combined_bits |= num

    # Calculate the sum of XOR totals by multiplying the combined bits with 2^(n-1)
    return combined_bits * (1 << (len(nums) - 1))


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
    ans = []

    # Iterate over all possible subsets using bit representations (2^n subsets) [1 << n = 2^n]
    for i in range(1 << n):
        subset = []

        for j in range(n):
            if i & (1 << j):  # Bitwise trick to check if j-th element is in this subset
                subset.append(nums[j])

        ans.append(subset)

    return ans


def subsets2(nums: List[int]) -> List[List[int]]:
    """
    Generates all possible subsets of an integer list.

    This solution uses backtracking to generate all subsets.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """

    def backtrack(start: int, current_subset: List[int]):
        """Backtracking function to generate all subsets."""
        ans.append(current_subset[:])

        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()  # Backtrack by removing the last element

    ans = []
    backtrack(0, [])
    return ans


def subsets3(nums: List[int]) -> List[List[int]]:
    """
    Generates all possible subsets of an integer list.

    This solution uses an iterative approach to generate all subsets.
    The time complexity of this solution is O(2^n * n) where n is the length of the input list.
    """
    ans = [[]]

    for num in nums:
        ans += [curr + [num] for curr in ans]  # Add the current number to all existing subsets

    return ans


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

    def is_palindrome(s, start, end):
        """Checks if a substring is a palindrome."""
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    def backtrack(start: int, current_partition: List[str]):
        """Backtracking function to generate all palindrome partitions."""
        if start == len(s):
            result.append(current_partition[:])  # Found a valid partition
            return

        for end in range(start, len(s)):
            if is_palindrome(s, start, end):
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()  # Backtrack by removing the last element

    result = []
    backtrack(0, [])
    return result


def partition2(s: str) -> List[List[str]]:
    """
    Generates all possible palindrome partitions of a string.

    This solution uses dynamic programming and backtracking to generate all palindrome partitions.
    The time complexity of this solution is O(n * 2^n) where n is the length of the input string.
    """
    n = len(s)
    dp_table = [[False] * n for _ in range(n)]  # DP table to store palindrome substrings

    for start in range(n - 1, -1, -1):
        for end in range(start, n):
            # A substring is a palindrome if the start and end characters are the same,
            # and the substring between them is also a palindrome
            if s[start] == s[end] and (end - start < 2 or dp_table[start + 1][end - 1]):
                dp_table[start][end] = True

    def backtrack(start: int, current_partition: List[str]):
        """Backtracking function to generate all palindrome partitions."""
        if start == n:
            result.append(current_partition[:])  # Found a valid partition
            return

        for end in range(start, n):
            if dp_table[start][end]:
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()  # Backtrack by removing the last element

    result = []
    backtrack(0, [])
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

    def backtrack(start_index: int):
        nonlocal beautiful_count  # Access the outer variable

        # Base case: a subset is found, check if it's beautiful
        if len(current_subset) > 0:
            for i in range(len(current_subset) - 1):
                if abs(current_subset[i] - current_subset[-1]) == k:
                    return  # Not beautiful, prune the search
            beautiful_count += 1

        # Recursive case: try adding each remaining element
        for i in range(start_index, n):
            current_subset.append(nums[i])
            backtrack(i + 1)  # Explore subsets starting from the next index
            current_subset.pop()  # Remove the last added element (backtracking)

    backtrack(0)  # Start backtracking from the beginning
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

    def count_beautiful_subsets(subsets: List[tuple], current_index: int) -> int:
        """Recursively counts beautiful subsets starting from the current index."""
        if current_index == len(subsets):
            return 1  # Base case: empty subset is beautiful

        # Create a key for the memoization dictionary based on the current index and remaining subsets
        key = (current_index, tuple(subsets[current_index:]))
        if key in memo:
            return memo[key]  # Return memoized result if available

        # Divide the subsets into two groups, excluding and including the current number, then count them
        exclude_count = count_beautiful_subsets(subsets, current_index + 1)
        include_count = (1 << subsets[current_index][1]) - 1

        # If the next number is 'k' apart from the current number, it must be in a different subset, so skip it
        if current_index + 1 < len(subsets) and subsets[current_index + 1][0] - subsets[current_index][0] == k:
            include_count *= count_beautiful_subsets(subsets, current_index + 2)
        else:
            # Otherwise, the next number can be in the same subset, so include it
            include_count *= count_beautiful_subsets(subsets, current_index + 1)

        # Store the total count in memoization dictionary for future use, then return it
        total_count = exclude_count + include_count
        memo[key] = total_count
        return total_count

    # Group numbers by their remainders when divided by k
    for num in nums:
        remainder_groups[num % k][num] += 1

    # Calculate beautiful subsets for each remainder group
    for group in remainder_groups.values():
        sorted_subsets = sorted(group.items())
        total_count *= count_beautiful_subsets(sorted_subsets, 0)

    return total_count - 1  # Exclude the empty subset


def beautifulSubsets3(nums: List[int], k: int) -> int:
    """
    Counts the number of beautiful subsets in the given list of numbers.

    This solution uses dynamic programming with an iterative approach to count beautiful subsets.
    The time complexity of this solution is O(n log n) where n is the length of the input list.
    """
    beautiful_count = 1
    remainder_groups = defaultdict(dict)

    # Group numbers by their remainders when divided by k
    for num in nums:
        remainder_groups[num % k][num] = remainder_groups[num % k].get(num, 0) + 1

    # Iterate over each remainder group
    for group in remainder_groups.values():
        prev_num = -k  # Initialize with a number guaranteed not to be in nums
        count_excluding_prev, exclude_count = 1, 1

        # Iterate over sorted numbers in the group
        for num, frequency in sorted(group.items()):
            include_count = (1 << frequency) - 1  # Count subsets with the current number

            # If current and previous numbers differ by k, they must be in different subsets
            if num - prev_num == k:
                include_count *= count_excluding_prev
            else:
                # Otherwise, the previous number can be in the same subset, so include it
                include_count *= exclude_count

            # Update counts for the next iteration
            count_excluding_prev, exclude_count = exclude_count, exclude_count + include_count
            prev_num = num

        beautiful_count *= exclude_count  # Update the overall count

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
        for end in range(start + 1, len(s) + 1):  # Try adding spaces at different positions
            word = s[start:end]
            if word in word_dict:
                for sentence in backtrack(end):  # Recursively generate sentences for the remaining string
                    sentences.append([word] + sentence)

        memo[start] = sentences
        return sentences

    memo = {}
    sentences = backtrack(0)  # Start backtracking from the beginning
    return [' '.join(words) for words in sentences]  # Convert the list of words to sentences


def wordBreak2(s: str, word_dict: List[str]) -> List[str]:
    pass


def wordBreak3(s: str, word_dict: List[str]) -> List[str]:
    pass


s = "catsanddog"
word_dict = ["cat", "cats", "and", "sand", "dog"]

# Expected output: ["cats and dog", "cat sand dog"]
print(wordBreak1(s, word_dict))
print(wordBreak2(s, word_dict))
print(wordBreak3(s, word_dict))

# <-------------------------------------------------- May 26th, 2024 -------------------------------------------------->
