"""Module for init a sudoku class"""
from typing import List, Tuple

import numpy as np


class Sudoku:
    def __init__(self, startingSudoku: str) -> np.ndarray:
        """From a string representation of an input Sudoku puzzle to a numpy representation

        Args:
            startingSudoku (str): a string representation of sudoku puzzle
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
                        inspired by https://github.com/challengingLuck/youtube/blob/master/sudoku/sudoku.py

        Returns:
            np.ndarray: A 9X9 np.ndarray representation of the original sudoku
                        e.g array([[4, 3, 0, 2, 6, 0, 7, 0, 0],
                                    [6, 8, 2, 0, 7, 0, 4, 9, 3],
                                    [1, 0, 7, 8, 0, 4, 5, 0, 0],
                                    [8, 2, 0, 1, 9, 0, 0, 4, 7],
                                    [0, 0, 4, 6, 0, 2, 9, 1, 0],
                                    [9, 5, 0, 7, 0, 3, 0, 2, 8],
                                    [5, 0, 9, 3, 0, 6, 0, 7, 0],
                                    [2, 4, 0, 0, 5, 7, 1, 0, 6],
                                    [7, 0, 3, 0, 1, 8, 2, 5, 0]]
        """
        np_Sudoku = np.array(
            [list(j) for j in [i for i in startingSudoku.split()]], dtype=int
        )
        self.np_Sudoku_original = np_Sudoku.copy()
        self.np_Sudoku = np_Sudoku

    def get_mask_nine_blocks(self) -> List[np.ndarray]:
        """split 9x9 sudoku mask in np representation into a list of nine 3 x 3 np.ndarray blocks


        Returns:
            List[np.ndarray]: a list of nine 3 x 3 np.ndarray blocks
        """

        self.nine_block_mask = [
            sudoku_piece[:, j : j + 3]
            for sudoku_piece in [
                self.original_puzzle_mask[i : i + 3, :] for i in [0, 3, 6]
            ]
            for j in [0, 3, 6]
        ]

    def mask_for_given_sudoku_val(self) -> np.ndarray:
        """
        Create a 9 x 9 mask of the entire puzzle

        Returns:
            np.ndarray:shape as the original sudoku but with 1 indicate the position's value is given and 0 otherwise
        """

        self.original_puzzle_mask = np.where(self.np_Sudoku > 0, 1, self.np_Sudoku)
