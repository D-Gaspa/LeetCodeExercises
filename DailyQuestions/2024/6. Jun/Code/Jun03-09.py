import heapq
from collections import Counter, defaultdict
from pprint import pprint
from typing import List

from tabulate import tabulate

from Utils.trie_utils import TrieVisualizer, TrieNode


# June 2024, Week 2: June 3rd - June 9th

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
    char_count = defaultdict(int)
    for i, char in enumerate(s):
        print(f"\tIteration {i + 1}:")
        print(f"\t\tCharacter: {char}")
        char_count[char] += 1
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
    print("\n--- Input Parameters ---")
    print(f"\twords = {words}")

    print("\n--- Initialization ---")
    common_chars = []
    print(f"\tcommon_chars = {common_chars}")

    print("\n--- Main Loop: Iterating through unique characters of the first word ---")
    iteration_data = []
    unique_chars = set(words[0])
    total_iterations = len(unique_chars)

    for i, char in enumerate(unique_chars):
        print(f"\n--- Character {i + 1}/{total_iterations}: '{char}' ---")

        print("\tCounting occurrences in each word:")
        char_counts = []
        for j, word in enumerate(words):
            count = word.count(char)
            char_counts.append(count)
            print(f"\t\tword[{j}] ('{word}'): count = {count}")

        print("\n\tCalculating minimum count:")
        min_count = min(char_counts)
        print(f"\t\tmin_count = min({char_counts}) = {min_count}")

        print("\n\tAdding characters to common_chars:")
        chars_to_add = [char] * min_count
        common_chars.extend(chars_to_add)
        print(f"\t\tAdding: {chars_to_add}")
        print(f"\t\tUpdated common_chars: {common_chars}")

        iteration_data.append([i + 1, char, char_counts, min_count, chars_to_add])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Character", "Counts in Words", "Min Count", "Added to Result"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal Result: {common_chars}")
    return common_chars


def commonChars2(words: List[str]) -> List[str]:
    print("\n--- Input Parameters ---")
    print(f"\twords = {words}")

    print("\n--- Initialization ---")
    common_chars = Counter(words[0])  # Initialize with character counts from the first word
    print(f"\tInitial common_chars (Counter):")
    print(f"\t\t{common_chars}")

    print("\n--- Main Loop (Intersecting Character Counts) ---")
    iteration_data = []
    for i, word in enumerate(words[1:], start=1):  # Iterate from the second word onwards
        print(f"\nIteration {i}: Comparing with word '{word}'")
        print(f"\tWord's Counter:")
        print(f"\t\t{Counter(word)}")

        common_chars &= Counter(word)  # Intersection of character counts
        print(f"\tUpdated common_chars after intersection:")
        print(f"\t\t{common_chars}")
        iteration_data.append([i, word, common_chars.copy()])

    print("\n--- Iteration Summary (Common Character Counts After Each Word) ---")
    headers = ["Iteration", "Word", "Common Chars Counter"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    result = list(common_chars.elements())  # Convert Counter to a list of characters
    print(f"\tFinal common_chars: {result}")
    return result


# <-------------------------------------------------- June 6th, 2024 -------------------------------------------------->
# 846. Hand of Straights

# Given an integer array hand representing the values on Alice's cards and an integer groupSize,
# determine if Alice can rearrange her cards into groups of groupSize consecutive cards.
# Return true if possible, otherwise false.


def isNStraightHand1(hand: List[int], group_size: int) -> bool:
    print("\n--- Input Parameters ---")
    print(f"\thand = {hand}")
    print(f"\tgroup_size = {group_size}")

    print("\n--- Initialization ---")
    print(f"\tChecking if len(hand) % group_size == 0")
    if len(hand) % group_size != 0:
        print(f"\t\tCondition false: {len(hand)} % {group_size} = {len(hand) % group_size}")
        print("\tReturning False")
        return False
    print(f"\t\tCondition true: {len(hand)} % {group_size} = 0")

    print("\n\tInitializing card_count (Counter)")
    card_count = Counter(hand)
    print(f"\tcard_count = {dict(card_count)}")

    print("\n\tInitializing card_min_heap")
    card_min_heap = list(card_count.keys())
    print(f"\tcard_min_heap (before heapify) = {card_min_heap}")
    heapq.heapify(card_min_heap)
    print(f"\tcard_min_heap (after heapify) = {card_min_heap}")

    print("\n--- Main Loop ---")
    iteration_data = []
    while card_min_heap:
        print(f"\n--- While card_min_heap is not empty ---")
        print(f"\tCurrent card_min_heap: {card_min_heap}")
        start_card = card_min_heap[0]
        print(f"\tstart_card = {start_card}")

        print(f"\n\tForming group of size {group_size}")
        for offset in range(group_size):
            current_card = start_card + offset
            print(f"\n\t--- Processing card: {current_card} ---")
            print(f"\t\tChecking if card_count[{current_card}] == 0")
            if card_count[current_card] == 0:
                print(f"\t\t\tCondition true: card_count[{current_card}] == 0")
                print("\t\tReturning False")
                return False
            print(f"\t\t\tCondition false: card_count[{current_card}] = {card_count[current_card]}")

            print(f"\t\tDecrementing card_count[{current_card}]")
            card_count[current_card] -= 1
            print(f"\t\t\tNew value: card_count[{current_card}] = {card_count[current_card]}")

            print(f"\t\tChecking if card_count[{current_card}] == 0")
            if card_count[current_card] == 0:
                print(f"\t\t\tCondition true: Removing {current_card} from card_min_heap")
                removed_card = heapq.heappop(card_min_heap)
                print(f"\t\t\tRemoved card: {removed_card}")
            else:
                print(f"\t\t\tCondition false: card_count[{current_card}] = {card_count[current_card]}")

        iteration_data.append([start_card, group_size, card_count.copy(), card_min_heap.copy()])

    print("\n--- Iteration Summary ---")
    headers = ["Start Card", "Group Size", "Card Count", "Min Heap"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("\tAll groups formed successfully")
    print("\tReturning True")
    return True


def isNStraightHand2(hand: List[int], group_size: int) -> bool:
    print("\n--- Input Parameters ---")
    print(f"\thand = {hand}")
    print(f"\tgroup_size = {group_size}")

    print("\n--- Initialization ---")
    print(f"\tChecking if len(hand) % group_size == 0")
    if len(hand) % group_size != 0:
        print(f"\t\tCondition false: "
              f"{len(hand)} % {group_size} = {len(hand) % group_size}")
        print("\tReturning False")
        return False
    print(f"\t\tCondition true: {len(hand)} % {group_size} = 0")

    print("\n\tInitializing card_count (Counter)")
    card_count = Counter(hand)
    print(f"\tcard_count = {dict(card_count)}")

    def find_group_start(card):
        print(f"\n\t--- Finding group start for card {card} ---")
        group_start = card
        while card_count[group_start - 1]:
            print(f"\t\tcard_count[{group_start - 1}] = {card_count[group_start - 1]}")
            print(f"\t\tMoving group_start from {group_start} to {group_start - 1}")
            group_start -= 1
        print(f"\t\tGroup start found: {group_start}")
        return group_start

    def remove_group(group_start):
        print(f"\n\t--- Removing group starting from {group_start} ---")
        for card in range(group_start, group_start + group_size):
            print(f"\t\tChecking card {card}")
            if not card_count[card]:
                print(f"\t\t\tcard_count[{card}] = 0, returning False")
                return False
            print(f"\t\t\tDecrementing card_count[{card}] from {card_count[card]} to {card_count[card] - 1}")
            card_count[card] -= 1
        print("\t\tGroup removed successfully")
        return True

    print("\n--- Main Loop ---")
    iteration_data = []
    for card in hand:
        print(f"\n--- Processing card {card} ---")
        group_start = find_group_start(card)
        print(f"\tInitial group_start: {group_start}")

        while group_start <= card:
            print(f"\n\t--- While group_start ({group_start}) <= card ({card}) ---")
            inner_loop_data = []
            while card_count[group_start]:
                print(f"\n\t\t--- While card_count[{group_start}] = {card_count[group_start]} ---")
                print(f"\t\tAttempting to remove group starting at {group_start}")
                if not remove_group(group_start):
                    print("\t\tGroup removal failed, returning False")
                    return False
                inner_loop_data.append(card_count.copy())
            print(f"\t\tMoving group_start from {group_start} to {group_start + 1}")
            group_start += 1

            if inner_loop_data:
                iteration_data.append([card, group_start - 1, inner_loop_data])

    print("\n--- Iteration Summary ---")
    headers = ["Processed Card", "Group Start", "Card Count After Each Removal"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("\tAll groups formed successfully")
    print("\tReturning True")
    return True


# <-------------------------------------------------- June 7th, 2024 -------------------------------------------------->
# 648. Replace Words

# Given a dictionary of roots and a sentence, replace all derivatives in the sentence with their corresponding root.
# If multiple roots can form a derivative, use the shortest root.
# Return the modified sentence.


def replaceWords1(dictionary: List[str], sentence: str) -> str:
    print("\n--- Input Parameters ---")
    print(f"\tdictionary = {dictionary}")
    print(f"\tsentence = {sentence}")

    print("\n--- Initialization ---")
    words = sentence.split()
    print(f"\tWords in sentence: {words}")

    print("\n--- Helper Function: find_shortest_root ---")

    def find_shortest_root(word: str) -> str:
        print(f"\n\t--- Processing word: '{word}' ---")
        matching_roots = [root for root in dictionary if word.startswith(root)]
        print(f"\t\tMatching roots: {matching_roots}")

        if matching_roots:
            shortest_root = min(matching_roots, key=len)
            print(f"\t\tShortest root found: '{shortest_root}'")
        else:
            shortest_root = word
            print(f"\t\tNo matching root found. Using original word: '{shortest_root}'")

        return shortest_root

    result = replace_words_main_loop(find_shortest_root, words)
    return result


def replaceWords2(dictionary: List[str], sentence: str) -> str:
    print("\n--- Input Parameters ---")
    print(f"\tdictionary = {dictionary}")
    print(f"\tsentence = {sentence}")

    print("\n--- Initialization ---")
    words = sentence.split()
    print(f"\tWords in sentence: {words}")

    print("\n--- Building Trie ---")

    def build_trie(words: List[str]) -> TrieNode:
        root = TrieNode()
        for word in words:
            print(f"\tAdding word: '{word}'")
            node = root
            for char in word:
                if char not in node.children:
                    print(f"\t\tAdding new node for character: '{char}'")
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word_end = True
            print(f"\t\tMarking end of word: '{word}'")
        return root

    root = build_trie(dictionary)
    print(f"\tTrie visualization: {TrieVisualizer.visualize(root, file_name='trie_visualization')}")

    print("\n--- Helper Function: find_shortest_root ---")

    def find_shortest_root(word: str) -> str:
        print(f"\n\t--- Processing word: '{word}' ---")
        node = root
        for index, char in enumerate(word):
            print(f"\t\tChecking character: '{char}'")
            if char not in node.children:
                print(f"\t\t\tNo matching child node. Returning original word: '{word}'")
                return word
            node = node.children[char]
            if node.is_word_end:
                shortest_root = word[:index + 1]
                print(f"\t\t\tFound shortest root: '{shortest_root}'")
                return shortest_root
        print(f"\t\tNo shorter root found. Returning original word: '{word}'")
        return word

    result = replace_words_main_loop(find_shortest_root, words)
    return result


def replace_words_main_loop(find_shortest_root, words):
    print("\n--- Main Processing Loop ---")
    iteration_data = []
    for i, word in enumerate(words):
        print(f"\n--- Word {i + 1}/{len(words)}: '{word}' ---")

        replaced_word = find_shortest_root(word)

        print(f"\tResult: {'Replaced' if replaced_word != word else 'Unchanged'}")
        print(f"\tOriginal word: '{word}'")
        print(f"\tFinal word: '{replaced_word}'")

        iteration_data.append([i + 1, word, replaced_word, 'Replaced' if replaced_word != word else 'Unchanged'])
    print("\n--- Iteration Summary ---")
    headers = ["Word #", "Original Word", "Replaced Word", "Action"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))
    print("\n--- Function Returning ---")
    result = " ".join([row[2] for row in iteration_data])
    print(f"\tFinal replaced sentence: '{result}'")
    return result


def replaceWords3(dictionary: List[str], sentence: str) -> str:
    print("\n--- Input Parameters ---")
    print(f"\tdictionary = {dictionary}")
    print(f"\tsentence = {sentence}")

    print("\n--- Initialization ---")
    root_lengths = {root: len(root) for root in dictionary}
    print("\tRoot lengths dictionary:")
    print(tabulate(root_lengths.items(), headers=["Root", "Length"], tablefmt="fancy_grid"))

    min_length = min(root_lengths.values())
    max_length = max(root_lengths.values())
    print(f"\tMinimum root length: {min_length}")
    print(f"\tMaximum root length: {max_length}")

    words = sentence.split()
    print(f"\tWords in sentence: {words}")

    print("\n--- Main Processing Loop ---")
    replaced_words = []
    iteration_data = []
    for i, word in enumerate(words):
        print(f"\n--- Word {i+1}/{len(words)}: '{word}' ---")
        replacement = word
        prefix_checks = []

        for length in range(min_length, min(max_length, len(word)) + 1):
            prefix = word[:length]
            prefix_checks.append(prefix)
            print(f"\tChecking prefix: '{prefix}'")
            if prefix in root_lengths:
                replacement = prefix
                print(f"\t\tMatch found! Replacing with: '{replacement}'")
                break
            else:
                print(f"\t\tNo match for this prefix")

        if replacement == word:
            print(f"\tNo shorter root found. Keeping original word: '{word}'")
        else:
            print(f"\tReplaced '{word}' with '{replacement}'")

        replaced_words.append(replacement)
        iteration_data.append([i+1, word, replacement, ", ".join(prefix_checks),
                               'Replaced' if replacement != word else 'Unchanged'])

    print("\n--- Iteration Summary ---")
    headers = ["Word #", "Original Word", "Replaced Word", "Prefixes Checked", "Action"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    result = " ".join(replaced_words)
    print(f"\tFinal replaced sentence: '{result}'")
    return result


# <-------------------------------------------------- June 8th, 2024 -------------------------------------------------->
# 523. Continuous Subarray Sum

# Given an integer array 'nums' and an integer 'k', return true if 'nums' has a continuous subarray of size at least two
# whose elements sum up to a multiple of 'k', or false otherwise.


def checkSubarraySum1(nums: List[int], k: int) -> bool:
    """
    Determines if there exists a continuous subarray of at least two elements in the given list `nums`
    whose sum is a multiple of `k`.

    The function uses a hashmap to store the remainders of cumulative sums when divided by `k`.
    By storing the first index where each remainder is seen, we can check if a later occurrence
    of the same remainder indicates a valid subarray.
    This is because if two cumulative sums have the same remainder when divided by `k`,
    their difference is a multiple of `k`.

    The time complexity of this solution is O(n) where n is the length of the input array.
    This is because every element in the array is processed exactly once.
    The space complexity is O(min(n, k)) for the dictionary as it can store at most k remainders.
    If k is larger than n, it stores at most n remainders.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tk = {k}")

    # initialization of two variables
    was_found = False
    accumulated_mod_k = 0
    mod_k_seen_map = {0: -1}
    print("\n--- Initialization ---")
    print(f"\taccumulated_mod_k = {accumulated_mod_k}")
    print(f"\tmod_k_seen_map = {mod_k_seen_map} ")

    print("\n--- Main Loop (Processing Nums Array) ---")
    iteration_data = []
    for index in range(len(nums)):
        print(f"\nBeginning of Iteration {index + 1}:\n")
        print(f"\tCurrent Index: {index}")
        print(f"\tCurrent Number: {nums[index]}")

        print("\tCalculating Accumulated Remainder:")
        previous_accumulated_mod_k = accumulated_mod_k
        accumulated_mod_k += nums[index]
        accumulated_mod_k %= k
        print(f"\tUpdated accumulated_mod_k: "
              f"({previous_accumulated_mod_k} + {nums[index]}) % {k} = {accumulated_mod_k}")

        if accumulated_mod_k not in mod_k_seen_map:
            mod_k_seen_map[accumulated_mod_k] = index
            print(f"\tNew remainder found. Updating mod_k_seen_map to: {mod_k_seen_map}")
        elif index - mod_k_seen_map[accumulated_mod_k] >= 2:
            print(f"\tA Valid subarray found. Returning True.")
            was_found = True
        else:
            print(f"\tThe subarray is not of size at least 2.")

        iteration_data.append([index + 1, nums[index], accumulated_mod_k,
                               mod_k_seen_map.copy()])  # add copy so that reference doesn't get updated

        if was_found:
            break

    print("\n--- Iteration Summary (Nums Processing) ---")
    headers = ["Iteration", "Number", "Accumulated Mod k", "Remainder Seen Map"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    if was_found:
        print("A valid subarray was found. Returning True.")
        return True

    print("No valid subarray found. Returning False.")
    return False


# <-------------------------------------------------- June 9th, 2024 -------------------------------------------------->
# 974. Subarray Sums Divisible by K

# Given an integer array nums and an integer k, return the number of non-empty contiguous subarrays that have a sum
# divisible by k.


def subarraysDivByK1(nums: List[int], k: int) -> int:
    """
    Counts the number of contiguous subarrays in a given list whose sum is divisible by k.

    The function uses a hashmap to keep track of the frequencies of remainders of the cumulative sums when divided by k.
    By maintaining the count of these remainders,
    it efficiently counts the number of subarrays that have a sum divisible by k.
    The key insight is that if two prefix sums have the same remainder when divided by k, their difference is a
    multiple of k, which implies that the subarray between these two sums has a sum divisible by k.

    The time complexity of this solution is O(n), where n is the length of the input array.
    This is because each element in the array is processed exactly once.
    The space complexity is O(min(n, k)) for the dictionary as it can store at most k remainders.
    If k is larger than n, it stores at most n remainders.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")
    print(f"\tk = {k}")

    print("\n--- Main Loop (Building Count Dictionary) ---")
    remainder_counts = {0: 1}
    prefix_sum = 0
    subarray_count = 0

    iteration_data = []  # To collect data for iteration summary table
    for index, num in enumerate(nums):
        print(f"\nIteration {index + 1}:")
        print(f"\tNumber: {num}")

        prefix_sum += num
        print(f"\tPrefix Sum (Updated): {prefix_sum}")
        print(f"\tCurrent remainder count dictionary: {remainder_counts}")

        remainder = prefix_sum % k
        print(f"\tRemainder ({prefix_sum} % {k}): {remainder}")

        if remainder in remainder_counts:
            print(f"\t\tRemainder found in dictionary with count: {remainder_counts[remainder]}")
            print(f"\t\tTotal subarrays divisible by {k} so far ({subarray_count} + {remainder_counts[remainder]} = "
                  f"{subarray_count + remainder_counts[remainder]})")
            subarray_count += remainder_counts[remainder]
            remainder_counts[remainder] += 1
            print(f"\t\tUpdated count for remainder {remainder}: {remainder_counts[remainder]}")
        else:
            print(f"\t\tRemainder not found. Adding to dictionary.")
            remainder_counts[remainder] = 1

        iteration_data.append([index + 1, num, prefix_sum, remainder, subarray_count, remainder_counts.copy()])

    print("\n--- Iteration Summary (Remainder Counts & Subarray Count) ---")
    headers = ["Iteration", "Number", "Prefix Sum", "Remainder", "Subarray Count", "Remainder Counts"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Final Remainder Counts Dictionary ---")
    pprint(remainder_counts)  # Use pprint for better formatting

    print("\n--- Function Returning ---")
    print(f"Total subarrays divisible by {k}: {subarray_count}")
    return subarray_count


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for june 3rd, 2024
# Expected output: 4
# appendCharacters1(s="coaching", t="coding")

# Test cases for june 4th, 2024
# Expected output: 7
# longestPalindrome1(s="abccccdd")
# longestPalindrome2(s="abccccdd")

# Test cases for june 5th, 2024
# Expected output: ["e", "l", "l"]
# commonChars1(words=["bella", "label", "roller"])
# commonChars2(words=["bella", "label", "roller"])

# Test cases for june 6th, 2024
# Expected output: True
# isNStraightHand1(hand=[1, 2, 3, 6, 2, 3, 4, 7, 8], group_size=3)
# isNStraightHand2(hand=[1, 2, 3, 6, 2, 3, 4, 7, 8], group_size=3)

# Test cases for june 7th, 2024
# Expected output: "the cat was rat by the bat"
# replaceWords1(dictionary=["pre", "post", "anti", "pro"],
#               sentence="The prewar and postwar eras had both prowar and antiwar sentiments")
# replaceWords2(dictionary=["pre", "post", "anti", "pro"],
#               sentence="The prewar and postwar eras had both prowar and antiwar sentiments")
# replaceWords3(dictionary=["pre", "post", "anti", "pro"],
#               sentence="The prewar and postwar eras had both prowar and antiwar sentiments")

# Test cases for june 8th, 2024
# Expected output: True
# checkSubarraySum1(nums=[23, 2, 6, 4, 7], k=6)

# Test cases for june 9th, 2024
# Expected output: 7
# subarraysDivByK1(nums=[4, 5, 0, -2, -3, 1], k=5)
