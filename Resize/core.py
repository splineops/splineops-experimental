# core.py
import numpy as np
from .utils import calculate_final_size
from .interpolation import (
    beta, get_interpolation_coefficients, get_samples, symmetric_fir,
    get_initial_causal_coefficient, get_initial_anti_causal_coefficient,
    do_integ, integ_sa, integ_as, do_diff, diff_sa, diff_as
)
from .utils import calculate_final_size, border

class Resize:
    def __init__(self):
        # Initialization of parameters
        self.interp_degree = None
        self.analy_degree = None
        self.synthe_degree = None
        self.zoom_y = None
        self.zoom_x = None
        self.inversable = None
        self.analy_even = 0
        self.corr_degree = None
        self.half_support = None
        self.spline_array_height = None
        self.spline_array_width = None
        self.index_min_height = None
        self.index_max_height = None
        self.index_min_width = None
        self.index_max_width = None
        self.tolerance = 1e-9

    def compute_zoom(self, input_img, output_img, analy_degree, synthe_degree,
                     interp_degree, zoom_y, zoom_x, shift_y, shift_x, inversable):
        self.interp_degree = interp_degree
        self.analy_degree = analy_degree
        self.synthe_degree = synthe_degree
        self.zoom_y = zoom_y
        self.zoom_x = zoom_x
        self.inversable = inversable

        ny, nx = input_img.shape

        size = calculate_final_size(inversable, ny, nx, zoom_y, zoom_x)
        working_size_y, working_size_x = size[:2]
        final_size_y, final_size_x = size[2:]

        if ((analy_degree + 1) / 2) * 2 == analy_degree + 1:
            self.analy_even = 1

        total_degree = interp_degree + analy_degree + 1
        self.corr_degree = analy_degree + synthe_degree + 1
        self.half_support = (total_degree + 1) / 2.0

        add_border_height = max(border(final_size_y, self.corr_degree), total_degree)
        final_total_height = final_size_y + add_border_height
        length_total_height = working_size_y + int(np.ceil(add_border_height / zoom_y))

        self.index_min_height = np.zeros(final_total_height, dtype=int)
        self.index_max_height = np.zeros(final_total_height, dtype=int)
        length_array_spln_height = final_total_height * (2 + total_degree)
        self.spline_array_height = np.zeros(length_array_spln_height)

        shift_y += ((analy_degree + 1.0) / 2.0 - np.floor((analy_degree + 1.0) / 2.0)) * (1.0 / zoom_y - 1.0)
        fact_height = np.power(zoom_y, analy_degree + 1)

        l_range_height = np.arange(final_total_height)
        affine_indices_height = l_range_height / zoom_y + shift_y
        self.index_min_height = np.ceil(affine_indices_height - self.half_support).astype(int)
        self.index_max_height = np.floor(affine_indices_height + self.half_support).astype(int)

        i = 0
        for l in range(final_total_height):
            for k in range(self.index_min_height[l], self.index_max_height[l] + 1):
                self.spline_array_height[i] = fact_height * beta(affine_indices_height[l] - k, total_degree)
                i += 1

        add_border_width = max(border(final_size_x, self.corr_degree), total_degree)
        final_total_width = final_size_x + add_border_width
        length_total_width = working_size_x + int(np.ceil(add_border_width / zoom_x))

        self.index_min_width = np.zeros(final_total_width, dtype=int)
        self.index_max_width = np.zeros(final_total_width, dtype=int)
        length_array_spln_width = final_total_width * (2 + total_degree)
        self.spline_array_width = np.zeros(length_array_spln_width)

        shift_x += ((analy_degree + 1.0) / 2.0 - np.floor((analy_degree + 1.0) / 2.0)) * (1.0 / zoom_x - 1.0)
        fact_width = np.power(zoom_x, analy_degree + 1)

        l_range_width = np.arange(final_total_width)
        affine_indices_width = l_range_width / zoom_x + shift_x
        self.index_min_width = np.ceil(affine_indices_width - self.half_support).astype(int)
        self.index_max_width = np.floor(affine_indices_width + self.half_support).astype(int)

        i = 0
        for l in range(final_total_width):
            for k in range(self.index_min_width[l], self.index_max_width[l] + 1):
                self.spline_array_width[i] = fact_width * beta(affine_indices_width[l] - k, total_degree)
                i += 1

        output_column = np.zeros(final_size_y)
        output_row = np.zeros(final_size_x)
        working_row = np.zeros(working_size_x)
        working_column = np.zeros(working_size_y)

        add_vector_height = np.zeros(length_total_height)
        add_output_vector_height = np.zeros(final_total_height)
        add_vector_width = np.zeros(length_total_width)
        add_output_vector_width = np.zeros(final_total_width)

        period_column_sym = 2 * working_size_y - 2
        period_row_sym = 2 * working_size_x - 2
        period_column_asym = 2 * working_size_y - 3
        period_row_asym = 2 * working_size_x - 3

        image = np.zeros((working_size_y, final_size_x))

        if inversable:
            inver_image = np.zeros((working_size_y, working_size_x))
            inver_image[:ny, :nx] = input_img

            if working_size_x > nx:
                inver_image[:, nx:] = inver_image[:, nx - 1: nx]
            if working_size_y > ny:
                inver_image[ny:, :] = inver_image[ny - 1: ny, :]

            for y in range(working_size_y):
                working_row = inver_image[y, :]
                get_interpolation_coefficients(working_row, interp_degree)
                self.resampling_row(working_row, output_row, add_vector_width, add_output_vector_width, period_row_sym, period_row_asym)
                image[y, :] = output_row

            for y in range(final_size_x):
                working_column = image[:, y]
                get_interpolation_coefficients(working_column, interp_degree)
                self.resampling_column(working_column, output_column, add_vector_height, add_output_vector_height, period_column_sym, period_column_asym)
                output_img[:, y] = output_column
        else:
            for y in range(working_size_y):
                working_row = input_img[y, :]
                get_interpolation_coefficients(working_row, interp_degree)
                self.resampling_row(working_row, output_row, add_vector_width, add_output_vector_width, period_row_sym, period_row_asym)
                image[y, :] = output_row

            for y in range(final_size_x):
                working_column = image[:, y]
                get_interpolation_coefficients(working_column, interp_degree)
                self.resampling_column(working_column, output_column, add_vector_height, add_output_vector_height, period_column_sym, period_column_asym)
                output_img[:, y] = output_column

    def resampling_row(self, input_vector, output_vector, add_vector, add_output_vector, max_sym_boundary, max_asym_boundary):
        length_input = len(input_vector)
        length_output = len(output_vector)
        length_total = len(add_vector)
        length_output_total = len(add_output_vector)
        average = 0

        # Projection Method
        if self.analy_degree != -1:
            average = do_integ(input_vector, self.analy_degree + 1)

        add_vector[:length_input] = input_vector

        l = np.arange(length_input, length_total)
        
        if self.analy_even == 1:
            l2 = np.where(l >= max_sym_boundary, np.abs(l % max_sym_boundary), l)
            l2 = np.where(l2 >= length_input, max_sym_boundary - l2, l2)
            add_vector[length_input:length_total] = input_vector[l2]
        else:
            l2 = np.where(l >= max_asym_boundary, np.abs(l % max_asym_boundary), l)
            l2 = np.where(l2 >= length_input, max_asym_boundary - l2, l2)
            add_vector[length_input:length_total] = -input_vector[l2]

        add_output_vector.fill(0.0)  # Initialize the add_output_vector with zeros

        index_min_width = np.array(self.index_min_width)
        index_max_width = np.array(self.index_max_width)
        spline_array_width = np.array(self.spline_array_width)

        i = 0
        for l in range(length_output_total):
            for k in range(index_min_width[l], index_max_width[l] + 1):
                index = k
                sign = 1
                if k < 0:
                    index = -k
                    if self.analy_even == 0:
                        index -= 1
                        sign = -1
                if k >= length_total:
                    index = length_total - 1
                # Geometric transformation and resampling
                add_output_vector[l] += sign * add_vector[index] * spline_array_width[i]
                i += 1

        # Projection Method
        if self.analy_degree != -1:
            # Differentiation analy_degree + 1 times of the signal
            do_diff(add_output_vector, self.analy_degree + 1)
            add_output_vector[:length_output_total] += average
            # IIR filtering
            get_interpolation_coefficients(add_output_vector, self.corr_degree)
            # Samples
            get_samples(add_output_vector, self.synthe_degree)

        output_vector[:length_output] = add_output_vector[:length_output]

    def resampling_column(self, input_vector, output_vector, add_vector, add_output_vector, max_sym_boundary, max_asym_boundary):
        length_input = len(input_vector)
        length_output = len(output_vector)
        length_total = len(add_vector)
        length_output_total = len(add_output_vector)
        average = 0

        # Projection Method
        if self.analy_degree != -1:
            average = do_integ(input_vector, self.analy_degree + 1)

        add_vector[:length_input] = input_vector

        l = np.arange(length_input, length_total)
        
        if self.analy_even == 1:
            l2 = np.where(l >= max_sym_boundary, np.abs(l % max_sym_boundary), l)
            l2 = np.where(l2 >= length_input, max_sym_boundary - l2, l2)
            add_vector[length_input:length_total] = input_vector[l2]
        else:
            l2 = np.where(l >= max_asym_boundary, np.abs(l % max_asym_boundary), l)
            l2 = np.where(l2 >= length_input, max_asym_boundary - l2, l2)
            add_vector[length_input:length_total] = -input_vector[l2]

        add_output_vector.fill(0.0)  # Initialize the add_output_vector with zeros
        i = 0

        index_min_height = np.array(self.index_min_height)
        index_max_height = np.array(self.index_max_height)
        spline_array_height = np.array(self.spline_array_height)

        for l in range(length_output_total):
            for k in range(index_min_height[l], index_max_height[l] + 1):
                index = k
                sign = 1
                if k < 0:
                    index = -k
                    if self.analy_even == 0:
                        index -= 1
                        sign = -1
                if k >= length_total:
                    index = length_total - 1
                # Geometric transformation and resampling
                add_output_vector[l] += sign * add_vector[index] * spline_array_height[i]
                i += 1

        # Projection Method
        if self.analy_degree != -1:
            # Differentiation analy_degree + 1 times of the signal
            do_diff(add_output_vector, self.analy_degree + 1)
            add_output_vector[:length_output_total] += average
            # IIR filtering
            get_interpolation_coefficients(add_output_vector, self.corr_degree)
            # Samples
            get_samples(add_output_vector, self.synthe_degree)

        output_vector[:length_output] = add_output_vector[:length_output]

def resize_image(input_img_normalized, output_size=None, zoom_factors=None,
                 method='Least-Squares', interpolation='Linear', inversable=False):
    # Determine the zoom factors
    if output_size is not None:
        zoom_y = output_size[0] / input_img_normalized.shape[0]
        zoom_x = output_size[1] / input_img_normalized.shape[1]
    elif zoom_factors is not None:
        zoom_y, zoom_x = zoom_factors
    else:
        raise ValueError("Either output_size or zoom_factors must be provided.")

    ##########################################

    # Set degrees based on interpolation method
    if interpolation == "Linear":
        interp_degree = 1
        synthe_degree = 1
        analy_degree = 1
    elif interpolation == "Quadratic":
        interp_degree = 2
        synthe_degree = 2
        analy_degree = 2
    else:  # Cubic
        interp_degree = 3
        synthe_degree = 3
        analy_degree = 3

    # Interpolation method must fulfill requirement: analy_degree = -1
    # Least-Squares method must fulfill requirement: analy_degree = interp_degree
    # Oblique projection method must fulfill requirement: -1 < analy_degree < interp_degree

    if method == "Interpolation":
        analy_degree = -1
    elif method == "Oblique projection":
        if interpolation == "Linear":
            analy_degree = 0
        elif interpolation == "Quadratic":
            analy_degree = 1
        else:  # Cubic
            analy_degree = 2

    # Define the output image size
    output_height = int(np.round(input_img_normalized.shape[0] * zoom_y))
    output_width = int(np.round(input_img_normalized.shape[1] * zoom_x))
    output_image = np.zeros((output_height, output_width), dtype=np.float64)

    # Create instance of Resize class
    resizer = Resize()

    # Perform resizing with a copy of the input image
    input_image_copy = input_img_normalized.copy()

    # Perform the resizing operation
    resizer.compute_zoom(
        input_image_copy,
        output_image,
        analy_degree,
        synthe_degree,
        interp_degree,
        zoom_y,
        zoom_x,
        shift_y=0,
        shift_x=0,
        inversable=inversable
    )

    return output_image
