import heapq
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

    The time complexity of this solution is O(n * m^2) where n is the number of words,
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
        common_chars_copy = common_chars.copy()

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

        iteration_data.append([i, word, common_chars_copy, new_common_chars.copy()])

    print("\n--- Iteration Summary (Common Characters per Word) ---")
    headers = ["Iteration", "Word", "Common Chars Before Iteration", "New Common Chars"]
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

    The time complexity of this solution is O(n * m), where n is the number of words,
    and m is the average length of each word.
    This is because for each word, the function runs through each character to update the Counter.
    The space complexity is O(1) as the size of the `Counter` used to store the character counts is limited by the
    fixed alphabet size (26).
    """
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
    """
    This function checks if it's possible to divide the 'hand' into several groups of 'group_size',
    where each group consists of consecutive cards.

    The function checks if the total number of cards is divisible by the group size, returning False if not.
    Then the function sorts 'hand' in increasing order.
    It repeatedly selects the smallest card left ('start_card'),
    then looks for group_size - 1 cards that follow 'start_card' consecutively.
    If it finds all of them, it removes them from 'hand' along with 'start_card';
    if any card is not in 'hand', it returns False right away.
    The function keeps forming groups in such a way until no cards are left, in which case it returns True.

    The time complexity of this solution is O(n^2) because it uses list.remove() inside a loop.
    Specifically, 'list.remove(x)' involves scanning the list until finding 'x' and then removing it,
    which can take up to O(n) time where n is the length of the list.
    Given that 'remove' is called in a loop, this will result in an overall quadratic time complexity.
    The space complexity is O(n) due to the storage requirements for the 'hand' list.
    """
    print("\n--- Input Parameters ---")
    print(f"\thand = {hand}")
    print(f"\tgroup_size = {group_size}")

    if len(hand) % group_size != 0:
        print("\n--- Decision Point (Divisibility Check) ---")
        print("\tThe number of cards is not divisible by the group size.")
        print("\tReturning False")
        return False

    hand.sort()
    print("\n--- After Sorting ---")
    print(f"\thand = {hand}")

    iteration_data = []  # To collect data for the iteration summary table
    iteration_count = 0

    print("\n--- Main Loop (Building Groups) ---")
    while hand:
        iteration_count += 1
        print(f"\nIteration {iteration_count}:")
        start_card = hand[0]
        print(f"\tStart Card: {start_card}")

        print(f"\tCard before iteration {iteration_count}: {hand}")

        group = [start_card]
        for i in range(1, group_size):
            card_to_find = start_card + i
            print(f"\t\tLooking for Card: {card_to_find}")
            if card_to_find not in hand:
                print("\n--- Decision Point (Consecutive Group Check) ---")
                print(f"\t\tCard {card_to_find} not found in the remaining hand.")
                print("\t\tReturning False")
                return False
            group.append(card_to_find)  # Build the group

        for card in group:
            hand.remove(card)

        iteration_data.append([iteration_count, group, hand.copy()])  # Add to summary table data

        print(f"\tCard after iteration {iteration_count}: {hand}")

    print("\n--- Iteration Summary (Formed Groups) ---")
    headers = ["Iteration", "Group Formed", "Remaining Hand"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("\tAll cards successfully formed groups.")
    print("\tReturning True")
    return True


def isNStraightHand2(hand: List[int], group_size: int) -> bool:
    """
    This function checks if it's possible to divide the 'hand' into several groups of 'group_size',
    where each group consists of consecutive cards.

    The function checks if the total number of cards is divisible by the group size, returning False if not.
    It counts each card's frequency and uses a min-heap to track the smallest card values.
    It iterates over the heap, checking for group_size - 1 consecutive cards with non-zero counts.
    If any are missing, it returns False.
    Otherwise, it decrements the counts and removes cards with a count of 0.
    If all cards form valid groups, it returns True.

    The time complexity of this solution is O(n log n) due to the use of a heap, where n is the size of the hand.
    The heap is created with all the unique card values and heappop is used (which has logarithmic time complexity).
    The space complexity of this solution is O(n), for the card_count and min_heap storage,
    where n is the number of unique cards.
    """
    print("\n--- Input Parameters ---")
    print(f"\thand = {hand}")
    print(f"\tgroup_size = {group_size}")

    print("\n--- Preprocessing (Counting Cards) ---")
    if len(hand) % group_size != 0:
        print("\tThe number of cards is not divisible by the group size.")
        print("\tReturning False")
        return False

    card_count = Counter(hand)
    print(f"\tCard Counts: {card_count}")

    min_heap = list(card_count.keys())
    heapq.heapify(min_heap)
    print(f"\tMin-Heap (Initial): {min_heap}")

    iteration_data = []  # To collect data for the iteration summary table
    iteration_count = 0

    print("\n--- Main Loop (Building Groups) ---")
    while min_heap:
        iteration_count += 1
        print(f"\nIteration {iteration_count}:")

        current_card = min_heap[0]
        print(f"\tCurrent Card (Min-Heap Top): {current_card}")
        print(f"\t\tMin-Heap: {min_heap}")

        group_formed = True  # Assume the group can be formed unless proven otherwise
        for i in range(group_size):
            card_to_find = current_card + i
            if card_count[card_to_find] == 0:
                group_formed = False  # Group cannot be formed
                print(f"\t\t\tCard {card_to_find} missing or count is zero")
                break
            else:
                print(f"\t\t\tCard {card_to_find} found in counts")

        if group_formed:
            for i in range(group_size):
                card_count[current_card + i] -= 1
                if card_count[current_card + i] == 0:
                    del card_count[current_card + i]
                    print(f"\t\t\tRemoved card {current_card + i} from counts")
                    if current_card + i != heapq.heappop(min_heap):
                        print("\t\t\tError: Heap inconsistency detected")
                        return False

        iteration_data.append([iteration_count, current_card, group_formed, min_heap.copy(), card_count.copy()])

    print("\n--- Iteration Summary (Group Formation Attempts) ---")
    headers = ["Iteration", "Current Card", "Group Formed?", "Min-Heap", "Card Counts"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("\tAll cards successfully formed groups.")
    print("\tReturning True")
    return True


def isNStraightHand3(hand: List[int], group_size: int) -> bool:
    """
    This function checks if it's possible to divide the 'hand' into several groups of 'group_size',
    where each group consists of consecutive cards.

    The function checks if the total number of cards is divisible by the group size, returning False if not.
    It then creates a Counter dictionary 'card_count' to count each unique card in the 'hand'.
    For each 'card' in the 'hand', it looks for the smallest card 'start_card'
    in the potential group that the 'card' belongs to.
    It then reduces by 1 the count of 'start_card' and any other card in the same group in 'card_count',
    and continues doing so until 'start_card' doesn't exist in the 'hand'.
    If all cards from 'start_card' to the 'card' can be decremented successfully,
    it moves on to the next potential 'start_card';
    if any card doesn't exist in the 'hand', it will return False.
    After all cards are processed and no False has been returned, it will return True meaning that the hand can be
    divided into groups as required.

    The time complexity of this solution is O(n) because every card is processed at most three times:
    once when constructing the dictionary, and potentially twice more when forming the groups.
    We might visit each card twice in the worst case: once on the way down, locating the start card and once on the way
    up, decrementing and removing the cards of a group, resulting in O(3n) operations, which simplifies to O(n).
    The space complexity is O(n) because of the Counter dictionary used to count the cards,
    where n is the total number of cards in the 'hand'.
    """
    print("\n--- Input Parameters ---")
    print(f"\thand = {hand}")
    print(f"\tgroup_size = {group_size}")

    print("\n--- Initial Check (Divisibility) ---")
    if len(hand) % group_size != 0:
        print(f"\tHand length ({len(hand)}) not divisible by group_size ({group_size}). Returning False.")
        return False

    print("\n--- Card Counting ---")
    card_count = Counter(hand)
    pprint(card_count)

    print("\n--- Main Loop (Processing Cards) ---")
    iteration_data = []  # Data collection for iteration summary
    for card in hand:
        print(f"\n\tProcessing card in hand: {card}")
        start_card = card
        print(f"\t\tInitial start_card: {start_card}")

        # Find the smallest card in the potential group
        print(f"\t\tFinding smallest start_card:")
        while card_count[start_card - 1]:
            start_card -= 1
            print(f"\t\t\tstart_card updated to: {start_card}")

        # Process cards in the group
        print(f"\t\tProcessing group starting from {start_card}:")
        while start_card <= card:
            while card_count[start_card]:
                # Iterate over cards in the group
                for next_card in range(start_card, start_card + group_size):
                    if not card_count[next_card]:  # Card not found
                        print(f"\t\t\tMissing card: {next_card}. Returning False.")
                        return False
                    card_count[next_card] -= 1
                    print(f"\t\t\tDecremented count of {next_card} to {card_count[next_card]}")
                    iteration_data.append([card, next_card, card_count.copy()])  # Collect data
            start_card += 1
            print(f"\t\t\tMoved to next start_card: {start_card}")

    # Iteration summary
    print("\n--- Iteration Summary (Card Processing) ---")
    headers = ["Card", "Processed Card", "Card Count (After)"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print("All cards successfully formed groups. Returning True.")
    return True


# <-------------------------------------------------- June 7th, 2024 -------------------------------------------------->
# 648. Replace Words

# Given a dictionary of roots and a sentence, replace all derivatives in the sentence with their corresponding root.
# If multiple roots can form a derivative, use the shortest root.
# Return the modified sentence.


def replaceWords1(dictionary: List[str], sentence: str) -> str:
    """
    Replaces all words in a sentence with their shortest root available in the given dictionary.

    The function is divided into two parts, first with the nested function 'shortest_root';
    in this function, a check is performed for each word to see if it starts with a root found in the dictionary.
    If multiple roots are found, it returns the shortest one.
    If no roots are found, it returns the original word.
    The main function then takes a sentence, splits it into words, and calls 'shortest_root' for each word.
    The result is then rejoined into a sentence with the roots replacing the original words.

    The time complexity of this solution is O(n * d * m) where n is the number of words in the sentence,
    d is the number of words in the dictionary, and m is the average length of these words.
    This is due to the nested loop structure where each word is checked against each root, and each root is checked
    against the word using the startswith method, which in the worst case can take O(m) time.
    The space complexity is O(n) as the function stores the modified words in a list before joining them.
    """
    print("\n--- Input Parameters ---")
    print(f"\tdictionary = {dictionary}")
    print(f"\tsentence = {sentence}")

    print("\n--- Splitting Sentence into Words ---")
    words = sentence.split()
    print(f"\twords = {words}")

    def shortest_root(word: str) -> str:
        print(f"\n--- Processing Word: '{word}' ---")

        print("\tChecking for Replacements:")
        replacements = []
        for root in dictionary:
            print(f"\t\tTesting Root: '{root}'")
            if word.startswith(root):
                replacements.append(root)
                print(f"\t\t\tWord starts with '{root}'. Adding to replacements: {replacements}")

        print(f"\tPossible Replacements Found: {replacements}")
        if replacements:
            shortest = min(replacements, key=len)
            print(f"\tShortest Replacement Selected: '{shortest}'")
            return shortest
        else:
            print("\tNo Replacements Found. Returning Original Word.")
            return word

    print("\n--- Replacing Words with Shortest Roots ---")
    replaced_words = []
    for word in words:
        replaced = shortest_root(word)
        replaced_words.append(replaced)

    print("\n--- Iteration Summary (Word Replacements) ---")
    headers = ["Original Word", "Replaced Word"]
    table_data = [[original, replaced] for original, replaced in zip(words, replaced_words)]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Joining Replaced Words into Sentence ---")
    result = " ".join(replaced_words)
    print(f"\tResult: '{result}'")

    print("\n--- Function Returning ---")
    return result


def replaceWords2(dictionary: List[str], sentence: str) -> str:
    """
    Replaces all words in a sentence with their shortest root available in the given dictionary.

    The first part of the function creates a Trie (prefix tree) using the dictionary.
    Each character forms a node and word completions are marked using 'is_word_end' flag.
    The initial building of the Trie allows for efficient prefix searches later on.

    In the 'shortest_root' function, the Trie is traversed character by character for each word in the sentence.
    If a word end is encountered during traversal, it returns the prefix up to that point.
    If no matching root is found, it returns the original word.
    The main function splits the sentence into words, replaces each word with the shortest root (if available)
    and reassembles the sentence.

    The time complexity is O(d * m + n * w) where d is the number of words in the dictionary, m is the average length
    of these words, n is the number of words in the sentence, and w is the average length of the sentence words.
    This is because, when building the Trie, each word is traversed character by character making it O(d * m).
    Then, we call the 'shortest_root' function for each word in the sentence, and the Trie is traversed over the
    characters of the sentence words, making it O(n * w).

    The space complexity is (d * m + n * w), as we're storing all characters in the Trie and the new sentence.
    """
    print("\n--- Input Parameters ---")
    print(f"\tdictionary = {dictionary}")
    print(f"\tsentence = {sentence}")

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_word_end = False

    def build_trie(words: List[str]) -> TrieNode:
        print("\n--- Building Trie ---")

        root = TrieNode()

        for word in words:
            print(f"\tAdding word: '{word}'")
            node = root
            for char in word:
                print(f"\t\tProcessing character: '{char}'")
                if char not in node.children:
                    print(f"\t\t\tAdding new node for '{char}'")
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word_end = True
            print(f"\t\tMarked end of word: '{word}'")

        return root

    root = build_trie(dictionary)

    def shortest_root(word: str) -> str:
        print(f"\n--- Finding Shortest Root for '{word}' ---")

        node = root
        for index, char in enumerate(word):
            print(f"\tChecking character: '{char}'")
            if char not in node.children:
                print(f"\t\tCharacter not found in Trie. Returning original word: '{word}'")
                return word
            node = node.children[char]
            if node.is_word_end:
                print(f"\t\tFound root: '{word[:index + 1]}'")
                return word[:index + 1]  # Shortest root found

        print(f"\tNo root found. Returning original word: '{word}'")
        return word

    print("\n--- Splitting Sentence into Words ---")
    words = sentence.split()
    print(f"\tInitial words: {words}")

    print("\n--- Replacing Words with Shortest Roots ---")
    replaced_words = []
    for word in words:
        replaced_word = shortest_root(word)
        replaced_words.append(replaced_word)

    print("\n--- Iteration Summary (Word Replacements) ---")
    headers = ["Word", "Replaced With"]
    iteration_data = [[words[i], replaced_words[i]] for i in range(len(words))]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    result = " ".join(replaced_words)

    print("\n--- Function Returning ---")
    print(f"\tFinal result: '{result}'")

    return result


def replaceWords3(dictionary: List[str], sentence: str) -> str:
    """
    Replaces all words in a sentence with their shortest root available in the given dictionary.

    The main context of the function is that it generates a dictionary 'word_lengths'
    where key-value pairs are the words in the dictionary and their lengths respectively.
    It then identifies and stores the minimum and maximum lengths of words in the dictionary for later use.
    The function then splits the sentence into individual words.

    For each word in the sentence, it checks each substring of the word from the length of the shortest to either the
    length of the longest word in the dictionary or the length of the original word; whichever is smallest.
    If it finds a matching substring in the dictionary, it stores the substring as the replacement for the word and
    breaks out of the loop to avoid longer roots.
    It then appends the replacement word to 'modified_sentence_words' list.
    Finally, it joins all the words in the 'modified_sentence_words' list to recreate the sentence.

    The time complexity of this solution is O(d * m + n * t) where d is the number of words in the dictionary,
    m is the average length of these words, n is the number of words in the sentence, and t is the maximum possible
    length of relevant substring checks for a word in the sentence.
    For each word in the sentence, we are only creating substrings from the minimum length of dictionary words to the
    minimum length between the maximum length of dictionary words and the length of the current word,
    which is substantially fewer operations than checking every character.
    Hence, the time complexity is O(n * t).
    Building the 'word_lengths' dictionary takes O(d * m) time as we iterate over each word and store its length.

    The space complexity is O(n + d) because we are storing n words from the sentence and d words from the dictionary.
    """
    print("\n--- Input Parameters ---")
    print(f"\tdictionary = {dictionary}")
    print(f"\tsentence = {sentence}")

    print("\n--- Building Word Length Dictionary ---")
    word_lengths = {word: len(word) for word in dictionary}
    print(f"\tword_lengths = {word_lengths}")

    print("\n--- Determining Minimum and Maximum Word Lengths ---")
    min_length = min(word_lengths.values())
    max_length = max(word_lengths.values())
    print(f"\tmin_length = {min_length}, max_length = {max_length}")

    print("\n--- Splitting Sentence into Words ---")
    words = sentence.split()
    print(f"\twords = {words}")

    modified_sentence_words = []
    for i, word in enumerate(words):
        print(f"\n--- Processing Word {i+1}: '{word}' ---")

        replacement = word
        print(f"\tInitial replacement = '{replacement}'")

        print("\tChecking Substrings:")
        for index in range(min_length, min(max_length, len(word)) + 1):
            substring = word[:index]
            print(f"\t\tSubstring: '{substring}'")

            if substring in word_lengths:
                replacement = substring
                print(f"\t\t\tFound in dictionary! Updating replacement to '{replacement}'")
                break
            else:
                print(f"\t\t\tNot found in dictionary.")

        modified_sentence_words.append(replacement)
        print(f"\tFinal replacement for '{word}': '{replacement}'")

    print("\n--- Iteration Summary (Word Replacements) ---")
    headers = ["Original Word", "Replaced Word"]
    table_data = [[original, replaced] for original, replaced in zip(words, modified_sentence_words)]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    result = " ".join(modified_sentence_words)
    print(f"\tResult: '{result}'")

    print("\n--- Function Returning ---")
    return result


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
# commonChars1(words)  # Expected output: ["e", "l", "l"]
# commonChars2(words)  # Expected output: ["e", "l", "l"]

# Test cases for june 6th, 2024
hand = [1, 2, 3, 6, 2, 3, 4, 7, 8]
groupSize = 3
# isNStraightHand1(hand, groupSize)  # Expected output: True
# isNStraightHand2(hand, groupSize)  # Expected output: True
# isNStraightHand3(hand, groupSize)  # Expected output: True

# Test cases for june 7th, 2024
dictionary = ["cat", "bat", "rat"]
sentence = "the cattle was rattled by the battery"
# replaceWords1(dictionary, sentence)  # Expected output: "the cat was rat by the bat"
# replaceWords2(dictionary, sentence)  # Expected output: "the cat was rat by the bat"
# replaceWords3(dictionary, sentence)  # Expected output: "the cat was rat by the bat"

# Test cases for june 8th, 2024

# Test cases for june 9th, 2024
