"""
Author: Florian Jaksik
Matr.Nr.: K0702862
Exercise 4
"""

from typing import Tuple

import numpy as np


def ex4(image_array: np.ndarray, border_x: Tuple[int, int], border_y: Tuple[int, int]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param image_array: A 2D numpy array representing an image
    :param border_x: A Tuple of two integers determining the width of the borders
    :param border_y: A Tuple of two integers determining the height of the borders
    :return: A Tuple of three numpy arrays. First the image_array with borders set to zero.
             Second an array of the same dimension with the borders set to zero and the rest
             to one.
             Third a 1D array with the original values of the borders in a sequence.
    :exception NotImplementedError: If image_array is not a numpy array or is not two dimensional
    :exception ValueError: If border_x|y values can't be cast to int or are smaller 1
                           If the shape of known pixels would be smaller than 16x16

    The function takes image_array and border_x|y and determines areas of borders and known pixels in
    image array. According to this information it returns the specified values.
    """

    check_input(border_x, border_y, image_array)

    known_array = create_known_array(border_x, border_y, image_array)

    image_array_cpy = create_input_array(image_array, known_array)

    target_array = image_array.copy()[known_array == 0.]

    return image_array_cpy, known_array, target_array


# ----------Create the output arrays------------------------------------------------------------------------------------

def create_input_array(image_array, known_array):
    image_array_cpy = image_array.copy()
    border_mask = known_array != 1
    np.putmask(image_array_cpy, border_mask, 0)

    return image_array_cpy


def create_known_array(border_x, border_y, image_array):
    known_array = np.zeros_like(image_array, dtype=image_array.dtype)
    fst_border_x_end, snd_border_x_start = border_x[0], known_array.shape[0] - border_x[1]
    fst_border_y_end, snd_border_y_start = border_y[0], known_array.shape[1] - border_y[1]
    should_be_ones = known_array[fst_border_x_end: snd_border_x_start, fst_border_y_end:snd_border_y_start]
    should_be_ones.fill(1)

    return known_array


# ----------Checks for input--------------------------------------------------------------------------------------------

def check_input(border_x, border_y, image_array):
    check_for_not_implemented(image_array)
    check_for_value_error(border_x, border_y, image_array)


def check_for_value_error(border_x, border_y, image_array):
    def will_there_still_be_16_times_16_pixels(_image_array, _border_x, _border_y):
        return _image_array.shape[0] - (_border_x[0] + _border_x[1]) < 16 or _image_array.shape[1] - (
                _border_y[0] + _border_y[1]) < 16

    try:
        # TODO this is not very nice, because it mixes checks and assignments
        border_x = int(border_x[0]), int(border_x[1])
        border_y = int(border_y[0]), int(border_y[1])
    except ValueError:
        raise ValueError("some border values can't be cast to int")

    if not all([i >= 1 for i in border_x + border_y]):
        raise ValueError("no border value is allowed to be smaller than 1")

    if will_there_still_be_16_times_16_pixels(image_array, border_x, border_y):
        raise ValueError("the resulting known pixels would be smaller than 16x16 pixels")


def check_for_not_implemented(image_array):
    if type(image_array) is not np.ndarray:
        raise NotImplementedError("image_array is not a numpy array")
    if image_array.ndim != 2:
        raise NotImplementedError("image_array is not a 2D array")