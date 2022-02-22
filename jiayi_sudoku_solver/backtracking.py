"""Module with code related to non-machine learning approach, such as backtracking and fill in numbers with certainty"""
import itertools
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import jiayi_sudoku_solver.Sudoku as Sudoku
from jiayi_sudoku_solver.util import (generate_empty_coord, get_nine_blocks,
                                      is_full_configure_valid, is_pos_valid,
                                      show_sudoku_filled, which_block_it_is)


def fill_in_num_one_pass(
    np_Sudoku: np.ndarray, coord: List[Tuple]
) -> Tuple[np.ndarray, int, Dict[Tuple, List]]:
    """Given a np representation of sudoku puzzle and a list of coord with positions that needs to be updated,
    fill the sudoku puzzle up with unique positional solution

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku
        coord (List[Tuple]): a list of tuple, e.g [(row1,col1), (row2,col2)..] indicating the position
                             that needs to be filled

    Returns:
        Tuple[np.ndarray, int, Dict[Tuple, List]]:
            np.ndarray: an updated sudoku puzzle by evaluating the input once and fill in the certain number
            int: an int on how many blocks have been updated
            Dict[Tuple, List]: position (key) and its possible values (val) for uncertain positions
    """
    # return the number of block updated
    block_updated = 0
    pos_val_dict = defaultdict(list, {k: [] for k in coord})
    nine_blocks = get_nine_blocks(np_Sudoku)
    for key in pos_val_dict:
        block_num = which_block_it_is(key[0], key[1])
        # print(block_num)
        nums_to_choose = list(range(1, 10))
        # remove block specific numbers
        nums_to_choose = [
            i
            for i in nums_to_choose
            if i not in set(list(nine_blocks[block_num].ravel()))
        ]
        # remove row specific numbers
        nums_to_choose = [i for i in nums_to_choose if i not in np_Sudoku[key[0]]]
        # remove col specific numbers
        nums_to_choose = [i for i in nums_to_choose if i not in np_Sudoku.T[key[1]]]
        pos_val_dict[key] = nums_to_choose
        if len(nums_to_choose) == 1:
            np_Sudoku[key[0]][key[1]] = nums_to_choose[0]
            block_updated += 1
    # sort pos_val_dict by length of possible numbers to fill
    pos_val_dict = dict(sorted(pos_val_dict.items(), key=lambda x: len(x[1])))
    return np_Sudoku, block_updated, pos_val_dict


def get_pos_val_dict(np_Sudoku: np.ndarray, coord: List[Tuple]) -> Dict[Tuple, List]:
    """Given a np representation of sudoku puzzle and a list of coord with positions that needs to be updated,
    return a dict with position and value to be filled

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku
        coord (List[Tuple]): a list of tuple, e.g [(row1,col1), (row2,col2)..] indicating the position
                             that needs to be filled

    Returns:
            Dict[Tuple, List]: position (key) and its possible values (val) for uncertain positions
    """
    # return the number of block updated
    block_updated = 0
    pos_val_dict = defaultdict(list, {k: [] for k in coord})
    nine_blocks = get_nine_blocks(np_Sudoku)
    for key in pos_val_dict:
        block_num = which_block_it_is(key[0], key[1])
        # print(block_num)
        nums_to_choose = list(range(1, 10))
        # remove block specific numbers
        nums_to_choose = [
            i
            for i in nums_to_choose
            if i not in set(list(nine_blocks[block_num].ravel()))
        ]
        # remove row specific numbers
        nums_to_choose = [i for i in nums_to_choose if i not in np_Sudoku[key[0]]]
        # remove col specific numbers
        nums_to_choose = [i for i in nums_to_choose if i not in np_Sudoku.T[key[1]]]
        pos_val_dict[key] = nums_to_choose
    # sort pos_val_dict by length of possible numbers to fill
    pos_val_dict = dict(sorted(pos_val_dict.items(), key=lambda x: len(x[1])))
    return pos_val_dict


def fill_val_with_certainties(np_Sudoku: np.ndarray) -> Dict[Tuple, List]:
    """Go through the given np_Sudoku many times and search for the unique positional
    solutions, fill in the solutions with 100% certainty until there is no certainly left

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku

    Returns:
        Dict[Tuple, List]: position (key) and its possible values (val) for missing positions
    """
    np_Sudoku_original = np_Sudoku.copy()
    start = time.time()

    coord = generate_empty_coord(np_Sudoku)
    while len(coord) > 0:
        np_Sudoku, block_updated, pos_val_dict = fill_in_num_one_pass(np_Sudoku, coord)
        if block_updated > 0:
            coord = generate_empty_coord(np_Sudoku)
        else:
            break
    end = time.time()
    total_time = end - start
    print(
        f"total time used for filling in sure values was {round(total_time,2)} seconds"
    )
    if is_full_configure_valid(np_Sudoku):
        print(f"Solved!")
        show_sudoku_filled(np_Sudoku_original, np_Sudoku)

    else:
        print(f"Filling positions with certainties was not enough to solve it!")
        show_sudoku_filled(np_Sudoku_original, np_Sudoku)
        return pos_val_dict


def back_tracking_solve(
    np_Sudoku: np.ndarray, i: int, pos_val_dict: Dict[Tuple, List]
) -> bool:
    """_summary_

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku
        i (int): the ith missing element, ranked by how many possible outcomes there are
        pos_val_dict (Dict[Tuple, List]): _description_

    Returns:
        _type_: position (key) and its possible values (val) for missing positions
    """

    all_keys = list(pos_val_dict.keys())
    if i == len(all_keys):
        return True
    else:
        coord = all_keys[i]
        values = pos_val_dict[coord]

        for val in values:
            if is_pos_valid(np_Sudoku, val, coord[0], coord[1]):
                np_Sudoku[coord[0]][coord[1]] = val
                # back tracking:
                # back_tracking_solve(np_Sudoku, i + 1)
                if back_tracking_solve(np_Sudoku, i + 1, pos_val_dict):
                    #    end = time.time()
                    #    total_time = end - start
                    #    print(f'Total time cost is {total_time}')
                    return True
                np_Sudoku[coord[0]][coord[1]] = 0
        return False


def solve_sudoku_with_conventional_methods(startingSudoku: str):
    """solve sudoku with a combination of filling in certainties and
    backtracking

    Args:
        startingSudoku (str): a string representation of sudoku puzzle,
                             e.g
                             ""
                            430260700
                            682070493
                            107804500
                            820190047
                            004602910
                            950703028
                            509306070
                            240057106
                            703018250
                            ""
    """
    sudoku_obj = Sudoku.Sudoku(startingSudoku)
    pos_val_dict = fill_val_with_certainties(sudoku_obj.np_Sudoku)

    if pos_val_dict:
        # back tracking mode on
        start = time.time()
        back_tracking_solve(sudoku_obj.np_Sudoku, 0, pos_val_dict)
        end = time.time()
        total_time = end - start
        print(f"Total time cost for backtracking is {round(total_time,2)} seconds")

        show_sudoku_filled(sudoku_obj.np_Sudoku_original, sudoku_obj.np_Sudoku)
