"""Module with functions to manipulate a Sudoku class object"""
import itertools
import time
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def get_nine_blocks(np_Sudoku: np.ndarray) -> List[np.ndarray]:
    """split 9x9 sudoku in np representation into a list of nine 3 x 3 np.ndarray blocks

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku or its mask

    Returns:
        List[np.ndarray]: a list of nine 3 x 3 np.ndarray blocks
    """
    nine_blocks = [
        sudoku_piece[:, j : j + 3]
        for sudoku_piece in [np_Sudoku[i : i + 3, :] for i in [0, 3, 6]]
        for j in [0, 3, 6]
    ]
    return nine_blocks


def is_full_configure_valid(np_Sudoku: np.ndarray) -> bool:
    """_summary_

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku

    Returns:
        bool: whether this configuration is correct
    """
    nine_blocks = get_nine_blocks(np_Sudoku)
    # check if each block adds up to 45
    for block in range(9):
        block_sum = sum(nine_blocks[block].ravel())
        if block_sum != 45:
            return False
    # check if each row adds up to 45:
    for row in range(9):
        row_sum = sum(np_Sudoku[row])
        if row_sum != 45:
            return False
    # check if each col adds up to 45:
    for col in range(9):
        col_sum = sum(np_Sudoku.T[col])
        if col_sum != 45:
            return False
    return True


def is_pos_valid(np_Sudoku: np.ndarray, val: int, row_pos: int, col_pos: int) -> bool:
    """Return if a value added to a certain position of sudoku is valid or not

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku that needs to be updated
        val (int): value to be filled in
        row_pos (int): row position associated to val
        col_pos (int): col position associated to val

    Returns:
        bool: whether val could be filled int to np_Sudoku[row_pos][col_pos]
    """
    nine_blocks = get_nine_blocks(np_Sudoku)
    block_num = which_block_it_is(row_pos, col_pos)
    # print(block_num)
    # val in block?
    if val in set(list(nine_blocks[block_num].ravel())):
        return False
    # val in row?
    if val in np_Sudoku[row_pos]:
        return False
    if val in np_Sudoku.T[col_pos]:
        return False
    return True


def generate_empty_coord(np_Sudoku: np.ndarray) -> List[Tuple]:
    """Geneate a list of (row, col) coordinates for unfilled positions on the input sudoku

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku

    Returns:
        List[Tuple]: [(row1,col1), (row2,col2)..]
    """
    # generate position of empty values
    coord = list()
    for row in range(9):
        for col in range(9):
            if np_Sudoku[row][col] == 0:
                coord.append((row, col))
    return coord


def convert_nine_blocks_to_sukudo(nine_blocks: List[np.ndarray]) -> np.ndarray:
    """convert nine 3 x 3 blocks back to one np.ndarray representation

    Args:
        nine_blocks (_type_): a list of nine 3 x 3 np.ndarray blocks

    Returns:
        np.ndarray: one np.ndarray representation of the corresponding input Sudoku
    """
    row3 = np.concatenate(([nine_blocks[i] for i in range(3)]), axis=1)
    row6 = np.concatenate(([nine_blocks[i] for i in range(3, 6)]), axis=1)
    row9 = np.concatenate(([nine_blocks[i] for i in range(6, 9)]), axis=1)
    updated_Sukudo = np.concatenate(([row3, row6, row9]), axis=0)
    return updated_Sukudo


def which_block_it_is(row: int, col: int) -> int:
    """given the coordinates of a position, return which block this position belongs to

    Args:
        row (int): row idx of np_sudoku, can be of value [0,1,2,3,4,5,6,7,8]
        col (int): col idx of np_sudoku, can be of value [0,1,2,3,4,5,6,7,8]

    Returns:
        int: the block number, top 3 rows of np_sudoku will be block 0,1,2,
            the middle 3 rows will be block 3,4,5, and last 3 rows will be block 6,7,8
    """
    if row in [0, 1, 2]:
        if col in [0, 1, 2]:
            return 0
        elif col in [3, 4, 5]:
            return 1
        else:
            return 2
    elif row in [3, 4, 5]:
        if col in [0, 1, 2]:
            return 3
        elif col in [3, 4, 5]:
            return 4
        else:
            return 5

    else:
        if col in [0, 1, 2]:
            return 6
        elif col in [3, 4, 5]:
            return 7
        else:
            return 8


def show_sudoku_with_known_truth(
    original_puzzle: np.ndarray,
    solved: np.ndarray = None,
    pred: np.ndarray = None,
    ax=None,
):
    """Simple plotting statement that ingests a 9x9 array (n), and plots a sudoku-style grid around it.
        taken from https://gitlab.com/ostrokach/proteinsolver/-/blob/master/notebooks/20_sudoku_demo.ipynb

    Args:
        original_puzzle (np.ndarray): original puzzle with missing values
        solved (np.ndarray, optional): original puzzle with true answers
        pred (np.ndarray, optional): original puzzle with pred answers
        ax (_type_, optional): _description_. Defaults to None.
    """
    #
    if ax is None:
        _, ax = plt.subplots()

    for y in range(10):
        ax.plot([-0.05, 9.05], [y, y], color="black", linewidth=1)

    for y in range(0, 10, 3):
        ax.plot([-0.05, 9.05], [y, y], color="black", linewidth=3)

    for x in range(10):
        ax.plot([x, x], [-0.05, 9.05], color="black", linewidth=1)

    for x in range(0, 10, 3):
        ax.plot([x, x], [-0.05, 9.05], color="black", linewidth=3)

    ax.axis("image")
    ax.axis("off")  # drop the axes, they're not important here

    for x in range(9):
        for y in range(9):
            puzzle_element = original_puzzle[8 - y][
                x
            ]  # need to reverse the y-direction for plotting
            if puzzle_element > 0:  # ignore the zeros
                T = f"{puzzle_element}"
                ax.text(x + 0.3, y + 0.2, T, fontsize=20)
            elif solved is not None and pred is not None:
                solved_element = solved[8 - y][x]
                pred_element = pred[8 - y][x]
                if solved_element == pred_element:
                    T = f"{solved_element}"
                    ax.text(x + 0.3, y + 0.2, T, fontsize=20, color="b")
                else:
                    ax.text(x + 0.1, y + 0.3, f"{pred_element}", fontsize=13, color="r")
                    ax.text(
                        x + 0.55, y + 0.3, f"{solved_element}", fontsize=13, color="g"
                    )


def show_sudoku_filled(original_puzzle: np.ndarray, pred: np.ndarray = None, ax=None):
    """Simple plotting statement that ingests a 9x9 array (n), and plots a sudoku-style grid around it.
        adapted from https://gitlab.com/ostrokach/proteinsolver/-/blob/master/notebooks/20_sudoku_demo.ipynb

    Args:
        original_puzzle (np.ndarray): original puzzle with missing values
        pred (np.ndarray, optional): original puzzle with pred answers
        ax (_type_, optional): _description_. Defaults to None.
    """
    #
    if ax is None:
        _, ax = plt.subplots()

    for y in range(10):
        ax.plot([-0.05, 9.05], [y, y], color="black", linewidth=1)

    for y in range(0, 10, 3):
        ax.plot([-0.05, 9.05], [y, y], color="black", linewidth=3)

    for x in range(10):
        ax.plot([x, x], [-0.05, 9.05], color="black", linewidth=1)

    for x in range(0, 10, 3):
        ax.plot([x, x], [-0.05, 9.05], color="black", linewidth=3)

    ax.axis("image")
    ax.axis("off")  # drop the axes, they're not important here

    for x in range(9):
        for y in range(9):
            puzzle_element = original_puzzle[8 - y][
                x
            ]  # need to reverse the y-direction for plotting
            if puzzle_element > 0:  # ignore the zeros
                T = f"{puzzle_element}"
                ax.text(x + 0.3, y + 0.2, T, fontsize=20)
            elif pred is not None:
                pred_element = pred[8 - y][x]

                T = f"{pred_element}"
                ax.text(x + 0.3, y + 0.2, T, fontsize=20, color="grey")
