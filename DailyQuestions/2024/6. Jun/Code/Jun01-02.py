from typing import List

from tabulate import tabulate


# June 2024, Week 1: June 1st - June 2nd

# <-------------------------------------------------- June 1st, 2024 -------------------------------------------------->
# 3110. Score of a String

# The score of a string is defined as the sum of the absolute difference between the ASCII values
# of adjacent characters.
# Given a string s, return the score of s.


def scoreOfString(s: str) -> int:
    """
    Calculates the total score of a string based on absolute differences between adjacent character ASCII values.

    This function iterates through each pair of adjacent characters in the string,
    calculates the absolute difference between their ASCII values, and accumulates this into a total score.

    The time complexity of this function is O(n), where n is the length of the string.
    This is because it performs a single iteration over the string's characters.
    The space complexity is O(1) as it uses a constant amount of extra space to store variables.
    """
    print("\n--- Input Parameters ---")
    print(f"\ts = '{s}'")  # Display the input string

    n = len(s)
    print(f"\tn (string length) = {n}")  # Display calculated length

    total_score = 0
    print(f"\ttotal_score (initialized) = {total_score}")  # Show initial score

    print("\n--- Main Loop (Calculating Character Pair Scores) ---")
    iteration_data = []  # Collect data for iteration summary
    for index in range(n - 1):
        print(f"\n\tIteration {index + 1}:")
        char1 = s[index]
        char2 = s[index + 1]
        print(f"\t\tCharacters: '{char1}' and '{char2}'")

        char1_ascii = ord(char1)
        char2_ascii = ord(char2)
        print(f"\t\tASCII Values: {char1_ascii} and {char2_ascii}")

        pair_score = abs(char1_ascii - char2_ascii)
        print(f"\t\tPair Score: {pair_score}")

        total_score += pair_score
        print(f"\t\tUpdated Total Score: {total_score}")

        # Store data for iteration summary
        iteration_data.append([index + 1, char1, char2, char1_ascii, char2_ascii, pair_score, total_score])

    print("\n--- Iteration Summary (Character Pair Scores) ---")
    headers = ["Iteration", "Char 1", "Char 2", "ASCII 1", "ASCII 2", "Pair Score", "Total Score"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"Final Total Score: {total_score}")
    return total_score


# <-------------------------------------------------- June 2nd, 2024 -------------------------------------------------->
# 344. Reverse String

# Given a character array s, reverse the order of its elements.
# You must do this by modifying the input array in-place with O(1) extra memory.


def reverseString(s: List[str]) -> None:
    """
    This function reverses the order of elements in a character array in-place.

    This function uses a two-pointer approach, where one pointer starts from the beginning (left_index),
    and the other one starts from the end of the array (right_index).
    The characters at these two pointers are swapped, and the pointers are moved towards each other.
    This carries on until the two pointers meet or pass each other, which suggests that the array is now reversed.

    The time complexity of this function is O(n), where n is the number of elements in the list.
    This is because the function iterates through roughly half of the elements in the list to reverse it.
    The space complexity is O(1) because it operates directly on the input list
    and uses a constant amount of additional memory for the index variables.
    """
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")

    print("\n--- Main Loop (Reversing In-Place) ---")
    left_index = 0
    right_index = len(s) - 1

    # Data collection for iteration table
    iteration_data = []

    while left_index < right_index:
        print(f"\nIteration: {left_index + 1}")
        print(f"\tBefore swap: s = {s}, left_index = {left_index}, right_index = {right_index}")

        s[left_index], s[right_index] = s[right_index], s[left_index]  # Swap

        print(f"\tAfter swap: s = {s}, left_index = {left_index}, right_index = {right_index}")

        # Collect data for iteration summary table
        iteration_data.append([left_index + 1, s.copy(), left_index, right_index])

        left_index += 1
        right_index -= 1

    print("\n--- Iteration Summary (Swaps and List States) ---")
    headers = ["Iteration", "List (s)", "Left Index", "Right Index"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Complete ---")
    print(f"Final reversed list: s = {s}")


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for June 1st, 2024
test_input_1 = "hello"
# scoreOfString(test_input_1)  # Expected output: 13

# Test cases for June 2nd, 2024
test_input_2 = ["h", "e", "l", "l", "o"]
# reverseString(test_input_2)  # Expected output: ["o", "l", "l", "e", "h"]
