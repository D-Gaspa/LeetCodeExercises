from collections import Counter
from pprint import pprint
from typing import List

from tabulate import tabulate


# Week 2: June 3rd - June 9th, 2024

# <-------------------------------------------------- June 3rd, 2024 -------------------------------------------------->
# 2486. Append Characters to String to Make Subsequence

# You are given two strings s and t consisting of only lowercase English letters.
# Return the minimum number of characters that need to be appended to the end of 's'
# so that 't' becomes a subsequence of s


def appendCharacters1(s: str, t: str) -> int:
    """
    Calculates the minimum number of characters from string 't'
    that must be appended to string 's' to make 't' a subsequence of 's'.

    This function iterates through both strings, comparing characters at corresponding positions.
    When a match is found, it advances in both strings; otherwise, it only moves forward in the first string.
    The function effectively checks if 't' is a subsequence of 's'
    (meaning 't' can be formed by deleting zero or more characters from 's').
    The result is the number of characters remaining in 't' after the comparison,
    indicating how many need to be appended.

    The function operates in O(n) time complexity,
    since each character of the string 's' and 't' is visited at most once.
    The space complexity is O(1) as the solution here does not require additional space that scales with input size.
    """
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")
    print(f"\tt = {t}")

    s_index = 0
    t_index = 0
    s_length = len(s)
    t_length = len(t)
    iterations = []

    print("\n--- Main Loop (Comparing 's' and 't') ---")
    while s_index < s_length and t_index < t_length:
        print(f"\nIteration of Main Loop: {s_index + 1}")
        print(f"\tCurrent 's_index': {s_index}")
        print(f"\tCurrent 't_index': {t_index}")
        match = False

        # Summary for this iteration
        iterations.append([s_index, t_index, s[s_index], t[t_index], ])

        # Condition check
        print(f"\tComparing s[{s_index}] = {s[s_index]} with t[{t_index}] = {t[t_index]}")
        if s[s_index] == t[t_index]:
            print("\t\tTrue Branch: s and t characters match")
            t_index += 1
            match = True
        else:
            print("\t\tFalse Branch: s and t characters don't match")
        s_index += 1

        # Add the match status to the summary
        iterations[-1].append("Yes" if match else "No")

    print("\n--- Iteration Summary (Current Indexes and Matched Characters) ---")
    headers = ["s_index", "t_index", "s[s_index]", "t[t_index]", "Match?"]
    print(tabulate(iterations, headers=headers, tablefmt="fancy_grid"))

    final_result = t_length - t_index

    print("\n--- Function return value ---")
    print(f"Final Result: t_length - t_index = {t_length} - {t_index} = {final_result}")

    return final_result


# <-------------------------------------------------- June 4th, 2024 -------------------------------------------------->
# 409 - Longest Palindrome

# Given a string 's', which consists of lowercase or uppercase letters, return the length of the longest palindrome
# that can be built with those letters.


def longestPalindrome1(s: str) -> int:
    """
    Calculates the length of the longest palindrome that can be built with the characters in the input string 's'.

    First, the function counts the frequency of each character using the 'char_count' dictionary.
    Then, it iterates through the counts and if the count is even, it adds the count to the result.
    If the count is odd, it adds one less than the count to the result and sets 'odd_exists' flag to True.
    This is done because palindromes can have at most one character with an odd count (at the center of the palindrome);
    all other characters must occur an even number of times.
    Finally, if there was at least one character with an odd count,
    it adds 1 to the result, accounting for the possible center element in the palindrome.

    The total time complexity of this function is O(n) because it iterates over the string 's' once to count characters
    and iterates over every character frequency in 'char_count' once.
    The space complexity of this function is O(1) because the 'char_count' dictionary will at most contain entries
    equal to the number of different characters which are constant.
    """
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")
    print("\n--- Building Character Count Dictionary ---")
    char_count = {}
    for i, char in enumerate(s):
        print(f"\tIteration {i + 1}:")
        print(f"\t\tCharacter: {char}")
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
        print(f"\t\tUpdated char_count: {char_count}")
    print("\n--- Character Count Dictionary ---")
    pprint(char_count)

    print("\n--- Main Loop (Calculating Palindrome Length) ---")
    result = 0
    odd_exists = False
    for char, count in char_count.items():
        print(f"\tCharacter: {char}, Count: {count}")
        if count % 2 == 0:
            result += count
            print(f"\t\tCount is even, added {count} to result (result = {result})")
        else:
            result += count - 1
            odd_exists = True
            print(f"\t\tCount is odd, added count - 1 = {count - 1} to result (result = {result})")
            print("\t\todd_exists flag set to True")

    # If there was at least one character with an odd count, it can be used as the center of the palindrome
    print("\n--- Decision Point (Odd Character Exists?) ---")
    if odd_exists:
        result += 1
        print("\tOdd character exists, adding 1 to result (result = {result})")

    print("\n--- Function Returning ---")
    print(f"Length of longest possible palindrome: {result}")
    return result


def longestPalindrome2(s: str) -> int:
    """
    Calculates the length of the longest palindrome that can be built with the characters in the input string 's'.

    This function uses a set `character_set` to keep track of characters encountered.
    For each character, if it is already in the set, it can be paired with its existing counterpart,
    contributing 2 to the palindrome length.
    If not in the set, it is added to the set as it may be paired with a future character.
    In the end, if `character_set` still contains characters,
    it means a palindrome can still fit one more character in its middle.
    Therefore, the result is incremented by 1 if `character_set` is not empty.

    The time complexity is O(n) where n is the length of the input string due to the single pass through the string.
    The space complexity is O(1) since the set will contain at most 52 characters (26 lowercase and 26 uppercase).
    """
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")

    print("\n--- Main Loop (Character Processing) ---")
    character_set = set()
    result = 0
    iteration_data = []

    for i, char in enumerate(s):
        print(f"\nIteration {i + 1}:")
        print(f"\tCharacter: {char}")
        print(f"\tcharacter_set (before): {character_set}")
        if char in character_set:
            result += 2
            character_set.remove(char)
            print(f"\t\tCharacter found in set, removed from set and result updated to {result}")
            iteration_data.append([i + 1, char, f"Remove '{char}', result += 2", character_set.copy(), result])
        else:
            character_set.add(char)
            print(f"\t\tCharacter not in set, added to set")
            iteration_data.append([i + 1, char, f"Add '{char}' to set", character_set.copy(), result])

    print("\n--- Final character_set ---")
    pprint(character_set)

    print("\n--- Decision Point (Remaining Characters?) ---")
    if character_set:
        result += 1
        print(f"\tCharacters remaining in set, adding 1 to result (result = {result})")

    print("\n--- Iteration Summary (Palindrome Length Calculation) ---")
    headers = ["Iteration", "Character", "Action", "Character Set", "Result"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"Length of longest possible palindrome: {result}")
    return result


# <-------------------------------------------------- June 5th, 2024 -------------------------------------------------->
# 1002. Find Common Characters

# Given a string array `words`, return an array of all characters that show up in all strings within the `words`
# (including duplicates).

def commonChars1(words: List[str]) -> List[str]:
    """
    Finds the common characters (including duplicates) that appear in all strings within the input list 'words'.

    This function starts with a list of characters from the first word and iteratively updates it
    by checking with each later word.
    Each character found in both the 'common_chars' and the current word is retained.
    This operation is performed by 'new_common_chars',
    which is then assigned back to 'common_chars' at the end of each iteration.

    The time complexity of this solution is O(n*m^2) where n is the number of words,
    and m is the average length of each word.
    The outer loop runs n times and the inner loop m times.
    Within the inner loop, the 'remove' operation is performed which can take up to m operations in the worst-case
    scenario (hence, m squared).
    The space complexity is O(p), where p is the length of the first word.
    The initial size of the common_chars list is determined by the size of the first word,
    and while exploring other words in the list, the size of this list would only decrease or remain the same.
    """
    print("\n--- Input Parameters ---")
    print(f"\twords = {words}")

    print("\n--- Initialization ---")
    common_chars = list(words[0])  # Initial common characters from the first word
    print(f"\tInitial common_chars: {common_chars}")

    print("\n--- Main Loop (Comparing with Each Word) ---")
    iteration_data = []  # To store data for iteration summary
    for i, word in enumerate(words[1:], start=1):  # Start from the second word
        print(f"\nIteration {i}: Comparing with word '{word}'")
        new_common_chars = []

        print("\tInner Loop (Checking Characters in Word) ---")
        for char in word:
            if char in common_chars:
                new_common_chars.append(char)
                common_chars.remove(char)  # Remove to handle duplicates correctly
                print(f"\t\tFound common char '{char}': {new_common_chars}")
            else:
                print(f"\t\t'{char}' not found in common_chars")

        common_chars = new_common_chars
        print(f"\tUpdated common_chars after iteration {i}: {common_chars}")

        iteration_data.append([i, word, new_common_chars])  # Store data for this iteration

    print("\n--- Iteration Summary (Common Characters per Word) ---")
    headers = ["Iteration", "Word", "New Common Chars"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal common_chars: {common_chars}")
    return common_chars


def commonChars2(words: List[str]) -> List[str]:
    """
    Finds the common characters (including duplicates) that appear in all strings within the input list 'words'.

    The function uses python's built-in data structure 'Counter' to generate a count
    of each character in a word.
    Initially, the 'Counter' of the first word is taken,
    and then applied with bitwise 'AND' operation with the 'Counter' of each later word.
    This bitwise 'AND' operation results in the intersection of characters of both words,
    keeping the count as the minimum of counts in both words.
    This ensures that 'common_chars' always holds the common characters with the least count among all processed words,
    thus effectively finding the common characters.
    Finally, 'elements()' method is used to generate the list of common characters from the updated 'Counter'.

    The time complexity of this solution is O(n*m), where n is the number of words,
    and m is the average length of each word.
    This is because for each word, the function runs through each character to update the Counter.
    The space complexity is O(min(m, k)), where m is the length of the longest word, and k is the size of the alphabet
    (26 for English), as 'Counter' holds the count of each character appearing in the word.
    """
    common_chars = Counter(words[0])

    for word in words[1:]:
        common_chars &= Counter(word)

    return list(common_chars.elements())


# <-------------------------------------------------- June 6th, 2024 -------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <-------------------------------------------------- June 7th, 2024 -------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <-------------------------------------------------- June 8th, 2024 -------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <-------------------------------------------------- June 9th, 2024 -------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for june 3rd, 2024
s = "coaching"
t = "coding"
# appendCharacters1(s, t)  # Expected output: 4

# Test cases for june 4th, 2024
s_2 = "abccccdd"
# longestPalindrome1(s_2)  # Expected output: 7
# longestPalindrome2(s_2)  # Expected output: 7

# Test cases for june 5th, 2024
words = ["bella", "label", "roller"]
commonChars1(words)  # Expected output: ["e", "l", "l"]
# commonChars2(words)  # Expected output: ["e", "l", "l"]

# Test cases for june 6th, 2024

# Test cases for june 7th, 2024

# Test cases for june 8th, 2024

# Test cases for june 9th, 2024
