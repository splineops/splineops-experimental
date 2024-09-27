import numpy as np

class Resize:
    def __init__(self):
        """Initialize the Resize class with default values."""
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

    def compute_zoom(self, input_img, output_img, analy_degree, synthe_degree, interp_degree, zoom_y, zoom_x, shift_y, shift_x, inversable):
        self.interp_degree = interp_degree
        self.analy_degree = analy_degree
        self.synthe_degree = synthe_degree
        self.zoom_y = zoom_y
        self.zoom_x = zoom_x
        self.inversable = inversable

        ny, nx = input_img.shape

        size = self.calculate_final_size(inversable, ny, nx, zoom_y, zoom_x)
        working_size_y, working_size_x = size[:2]
        final_size_y, final_size_x = size[2:]

        if ((analy_degree + 1) / 2) * 2 == analy_degree + 1:
            self.analy_even = 1

        total_degree = interp_degree + analy_degree + 1
        self.corr_degree = analy_degree + synthe_degree + 1
        self.half_support = (total_degree + 1) / 2.0

        add_border_height = max(self.border(final_size_y, self.corr_degree), total_degree)
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
                self.spline_array_height[i] = fact_height * self.beta(affine_indices_height[l] - k, total_degree)
                i += 1

        add_border_width = max(self.border(final_size_x, self.corr_degree), total_degree)
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
                self.spline_array_width[i] = fact_width * self.beta(affine_indices_width[l] - k, total_degree)
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
                self.get_interpolation_coefficients(working_row, interp_degree)
                self.resampling_row(working_row, output_row, add_vector_width, add_output_vector_width, period_row_sym, period_row_asym)
                image[y, :] = output_row

            for y in range(final_size_x):
                working_column = image[:, y]
                self.get_interpolation_coefficients(working_column, interp_degree)
                self.resampling_column(working_column, output_column, add_vector_height, add_output_vector_height, period_column_sym, period_column_asym)
                output_img[:, y] = output_column
        else:
            for y in range(working_size_y):
                working_row = input_img[y, :]
                self.get_interpolation_coefficients(working_row, interp_degree)
                self.resampling_row(working_row, output_row, add_vector_width, add_output_vector_width, period_row_sym, period_row_asym)
                image[y, :] = output_row

            for y in range(final_size_x):
                working_column = image[:, y]
                self.get_interpolation_coefficients(working_column, interp_degree)
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
            average = self.do_integ(input_vector, self.analy_degree + 1)

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
            self.do_diff(add_output_vector, self.analy_degree + 1)
            add_output_vector[:length_output_total] += average
            # IIR filtering
            self.get_interpolation_coefficients(add_output_vector, self.corr_degree)
            # Samples
            self.get_samples(add_output_vector, self.synthe_degree)

        output_vector[:length_output] = add_output_vector[:length_output]

    def resampling_column(self, input_vector, output_vector, add_vector, add_output_vector, max_sym_boundary, max_asym_boundary):
        length_input = len(input_vector)
        length_output = len(output_vector)
        length_total = len(add_vector)
        length_output_total = len(add_output_vector)
        average = 0

        # Projection Method
        if self.analy_degree != -1:
            average = self.do_integ(input_vector, self.analy_degree + 1)

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
            self.do_diff(add_output_vector, self.analy_degree + 1)
            add_output_vector[:length_output_total] += average
            # IIR filtering
            self.get_interpolation_coefficients(add_output_vector, self.corr_degree)
            # Samples
            self.get_samples(add_output_vector, self.synthe_degree)

        output_vector[:length_output] = add_output_vector[:length_output]

    def beta(self, x, degree):
        betan = 0.0
        if degree == 0:
            if abs(x) < 0.5 or x == -0.5:
                betan = 1.0
        elif degree == 1:
            x = abs(x)
            if x < 1.0:
                betan = 1.0 - x
        elif degree == 2:
            x = abs(x)
            if x < 0.5:
                betan = 3.0 / 4.0 - x * x
            elif x < 1.5:
                x -= 3.0 / 2.0
                betan = x * x * (1.0 / 2.0)
        elif degree == 3:
            x = abs(x)
            if x < 1.0:
                betan = x * x * (x - 2.0) * (1.0 / 2.0) + 2.0 / 3.0
            elif x < 2.0:
                x -= 2.0
                betan = x * x * x * (-1.0 / 6.0)
        elif degree == 4:
            x = abs(x)
            if x < 0.5:
                x *= x
                betan = x * (x * (1.0 / 4.0) - 5.0 / 8.0) + 115.0 / 192.0
            elif x < 1.5:
                betan = x * (x * (x * (5.0 / 6.0 - x * (1.0 / 6.0)) - 5.0 / 4.0) + 5.0 / 24.0) + 55.0 / 96.0
            elif x < 2.5:
                x -= 5.0 / 2.0
                x *= x
                betan = x * x * (1.0 / 24.0)
        elif degree == 5:
            x = abs(x)
            if x < 1.0:
                a = x * x
                betan = a * (a * (1.0 / 4.0 - x * (1.0 / 12.0)) - 1.0 / 2.0) + 11.0 / 20.0
            elif x < 2.0:
                betan = x * (x * (x * (x * (x * (1.0 / 24.0) - 3.0 / 8.0) + 5.0 / 4.0) - 7.0 / 4.0) + 5.0 / 8.0) + 17.0 / 40.0
            elif x < 3.0:
                a = 3.0 - x
                x = a * a
                betan = a * x * x * (1.0 / 120.0)
        elif degree == 6:
            x = abs(x)
            if x < 0.5:
                x *= x
                betan = x * (x * (7.0 / 48.0 - x * (1.0 / 36.0)) - 77.0 / 192.0) + 5887.0 / 11520.0
            elif x < 1.5:
                betan = x * (x * (x * (x * (x * (x * (1.0 / 48.0) - 7.0 / 48.0) + 21.0 / 64.0) - 35.0 / 288.0) - 91.0 / 256.0) - 7.0 / 768.0) + 7861.0 / 15360.0
            elif x < 2.5:
                betan = x * (x * (x * (x * (x * (7.0 / 60.0 - x * (1.0 / 120.0)) - 21.0 / 32.0) + 133.0 / 72.0) - 329.0 / 128.0) + 1267.0 / 960.0) + 1379.0 / 7680.0
            elif x < 3.5:
                x -= 7.0 / 2.0
                x *= x * x
                betan = x * x * (1.0 / 720.0)
        elif degree == 7:
            x = abs(x)
            if x < 1.0:
                a = x * x
                betan = a * (a * (a * (x * (1.0 / 144.0) - 1.0 / 36.0) + 1.0 / 9.0) - 1.0 / 3.0) + 151.0 / 315.0
            elif x < 2.0:
                betan = x * (x * (x * (x * (x * (x * (1.0 / 20.0 - x * (1.0 / 240.0)) - 7.0 / 30.0) + 1.0 / 2.0) - 7.0 / 18.0) - 1.0 / 10.0) - 7.0 / 90.0) + 103.0 / 210.0
            elif x < 3.0:
                betan = x * (x * (x * (x * (x * (x * (x * (1.0 / 720.0) - 1.0 / 36.0) + 7.0 / 30.0) - 19.0 / 18.0) + 49.0 / 18.0) - 23.0 / 6.0) + 217.0 / 90.0) - 139.0 / 630.0
            elif x < 4.0:
                a = 4.0 - x
                x = a * a * a
                betan = x * x * a * (1.0 / 5040.0)
        return betan


    def do_integ(self, c, nb):
        size = len(c)
        m = 0.0
        average = 0.0

        if nb == 1:
            average = np.sum(c)
            average = (2.0 * average - c[size - 1] - c[0]) / (2.0 * size - 2)
            self.integ_sa(c, average)

        elif nb == 2:
            average = np.sum(c)
            average = (2.0 * average - c[size - 1] - c[0]) / (2.0 * size - 2)
            self.integ_sa(c, average)
            self.integ_as(c, c)

        elif nb == 3:
            average = np.sum(c)
            average = (2.0 * average - c[size - 1] - c[0]) / (2.0 * size - 2)
            self.integ_sa(c, average)
            self.integ_as(c, c)
            m = np.sum(c)
            m = (2.0 * m - c[size - 1] - c[0]) / (2.0 * size - 2)
            self.integ_sa(c, m)

        elif nb == 4:
            average = np.sum(c)
            average = (2.0 * average - c[size - 1] - c[0]) / (2.0 * size - 2)
            self.integ_sa(c, average)
            self.integ_as(c, c)
            m = np.sum(c)
            m = (2.0 * m - c[size - 1] - c[0]) / (2.0 * size - 2)
            self.integ_sa(c, m)
            self.integ_as(c, c)

        return average

    def integ_sa(self, c, m):
        c -= m
        c[0] *= 0.5
        c[1:] += np.cumsum(c[:-1])

    def integ_as(self, c, y):
        z = c.copy()
        y[0] = z[0]
        y[1] = 0
        y[2:] = -np.cumsum(z[1:-1])

    def do_diff(self, c, nb):
        size = len(c)
        if nb == 1:
            self.diff_as(c)
        elif nb == 2:
            self.diff_sa(c)
            self.diff_as(c)
        elif nb == 3:
            self.diff_as(c)
            self.diff_sa(c)
            self.diff_as(c)
        elif nb == 4:
            self.diff_sa(c)
            self.diff_as(c)
            self.diff_sa(c)
            self.diff_as(c)

    def diff_sa(self, c):
        old = c[-2]
        c[:-1] -= c[1:]  # Perform the element-wise subtraction
        c[-1] -= old     # Update the last element

    def diff_as(self, c):
        c[1:] -= c[:-1]  # Perform the element-wise subtraction for differentiation
        c[0] *= 2.0      # Update the first element

    @staticmethod
    def border(size, degree):
        if degree in [0, 1]:
            return 0

        tolerance = 1e-10
        if degree == 2:
            z = np.sqrt(8.0) - 3.0
        elif degree == 3:
            z = np.sqrt(3.0) - 2.0
        elif degree == 4:
            z = np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0
        elif degree == 5:
            z = (np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) + np.sqrt(105.0 / 4.0) - 13.0 / 2.0)
        elif degree == 6:
            z = -0.488294589303044755130118038883789062112279161239377608394
        elif degree == 7:
            z = -0.5352804307964381655424037816816460718339231523426924148812
        else:
            raise ValueError("Invalid degree (should be [0..7])")

        horizon = 2 + int(np.log(tolerance) / np.log(abs(z)))
        horizon = min(horizon, size)
        return horizon

    @staticmethod
    def calculate_final_size(inversable, height, width, zoom_y, zoom_x):
        size = [height, width, 0, 0]

        if inversable:
            w2 = int(round(round((size[0] - 1) * zoom_y) / zoom_y))
            while size[0] - 1 - w2 != 0:
                size[0] += 1
                w2 = int(round(round((size[0] - 1) * zoom_y) / zoom_y))

            h2 = int(round(round((size[1] - 1) * zoom_x) / zoom_x))
            while size[1] - 1 - h2 != 0:
                size[1] += 1
                h2 = int(round(round((size[1] - 1) * zoom_x) / zoom_x))

            size[2] = int(round((size[0] - 1) * zoom_y) + 1)
            size[3] = int(round((size[1] - 1) * zoom_x) + 1)
        else:
            size[2] = int(round(size[0] * zoom_y))
            size[3] = int(round(size[1] * zoom_x))

        return size

    def get_interpolation_coefficients(self, c, degree):
        tolerance = 1e-10
        z = []
        lambda_ = 1.0

        if degree == 0 or degree == 1:
            return
        elif degree == 2:
            z = [np.sqrt(8.0) - 3.0]
        elif degree == 3:
            z = [np.sqrt(3.0) - 2.0]
        elif degree == 4:
            z = [np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0,
                np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0]
        elif degree == 5:
            z = [np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) + np.sqrt(105.0 / 4.0) - 13.0 / 2.0,
                np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0)) - np.sqrt(105.0 / 4.0) - 13.0 / 2.0]
        elif degree == 6:
            z = [-0.488294589303044755130118038883789062112279161239377608394,
                -0.081679271076237512597937765737059080653379610398148178525368,
                -0.00141415180832581775108724397655859252786416905534669851652709]
        elif degree == 7:
            z = [-0.5352804307964381655424037816816460718339231523426924148812,
                -0.122554615192326690515272264359357343605486549427295558490763,
                -0.0091486948096082769285930216516478534156925639545994482648003]
        else:
            raise ValueError("Invalid spline degree (should be [0..7])")

        if len(c) == 1:
            return

        z = np.array(z)
        lambda_ = np.prod((1.0 - z) * (1.0 - 1.0 / z))

        c *= lambda_

        for zk in z:
            c[0] = self.get_initial_causal_coefficient(c, zk, tolerance)
            for n in range(1, len(c)):
                c[n] += zk * c[n - 1]

            c[-1] = self.get_initial_anti_causal_coefficient(c, zk, tolerance)
            for n in range(len(c) - 2, -1, -1):
                c[n] = zk * (c[n + 1] - c[n])

    def get_samples(self, c, degree):
        if degree == 0 or degree == 1:
            return
        elif degree == 2:
            h = [3.0 / 4.0, 1.0 / 8.0]
        elif degree == 3:
            h = [2.0 / 3.0, 1.0 / 6.0]
        elif degree == 4:
            h = [115.0 / 192.0, 19.0 / 96.0, 1.0 / 384.0]
        elif degree == 5:
            h = [11.0 / 20.0, 13.0 / 60.0, 1.0 / 120.0]
        elif degree == 6:
            h = [5887.0 / 11520.0, 10543.0 / 46080.0, 361.0 / 23040.0, 1.0 / 46080.0]
        elif degree == 7:
            h = [151.0 / 315.0, 397.0 / 1680.0, 1.0 / 42.0, 1.0 / 5040.0]
        else:
            raise ValueError("Invalid spline degree (should be [0..7])")

        s = np.zeros_like(c)
        self.symmetric_fir(h, c, s)
        np.copyto(c, s)

    @staticmethod
    def get_initial_anti_causal_coefficient(c, z, tolerance):
        return (z * c[-2] + c[-1]) * z / (z * z - 1.0)

    @staticmethod
    def get_initial_causal_coefficient(c, z, tolerance):
        z1 = z
        zn = z ** (len(c) - 1)
        sum_ = c[0] + zn * c[-1]
        horizon = len(c)

        if tolerance > 0.0:
            horizon = 2 + int(np.log(tolerance) / np.log(np.abs(z)))
            horizon = min(horizon, len(c))

        n = np.arange(1, horizon - 1)
        z1_array = z ** n
        zn_array = (zn / z ** n) * z ** n  # simplifies to zn since (zn / z**n) * z**n = zn
        sum_ += np.sum((z1_array + zn_array) * c[1:horizon-1])

        return sum_ / (1.0 - z ** (2 * len(c) - 2))

    @staticmethod
    def symmetric_fir(h, c, s):
        if len(c) != len(s):
            raise IndexError("Incompatible size")

        if len(h) == 2:
            if len(c) >= 2:
                s[0] = h[0] * c[0] + 2.0 * h[1] * c[1]
                s[1:-1] = h[0] * c[1:-1] + h[1] * (c[:-2] + c[2:])
                s[-1] = h[0] * c[-1] + 2.0 * h[1] * c[-2]
            else:
                if len(c) == 1:
                    s[0] = (h[0] + 2.0 * h[1]) * c[0]
                else:
                    raise ValueError("Invalid length of data")
        
        elif len(h) == 3:
            if len(c) >= 4:
                s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2]
                s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3])
                s[2:-2] = h[0] * c[2:-2] + h[1] * (c[1:-3] + c[3:-1]) + h[2] * (c[0:-4] + c[4:])
                s[-2] = h[0] * c[-2] + h[1] * (c[-3] + c[-1]) + h[2] * (c[-4] + c[-2])
                s[-1] = h[0] * c[-1] + 2.0 * h[1] * c[-2] + 2.0 * h[2] * c[-3]
            else:
                if len(c) == 3:
                    s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2]
                    s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + 2.0 * h[2] * c[1]
                    s[2] = h[0] * c[2] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[0]
                elif len(c) == 2:
                    s[0] = (h[0] + 2.0 * h[2]) * c[0] + 2.0 * h[1] * c[1]
                    s[1] = (h[0] + 2.0 * h[2]) * c[1] + 2.0 * h[1] * c[0]
                elif len(c) == 1:
                    s[0] = (h[0] + 2.0 * (h[1] + h[2])) * c[0]
                else:
                    raise ValueError("Invalid length of data")

        elif len(h) == 4:
            if len(c) >= 6:
                s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2] + 2.0 * h[3] * c[3]
                s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3]) + h[3] * (c[2] + c[4])
                s[2] = h[0] * c[2] + h[1] * (c[1] + c[3]) + h[2] * (c[0] + c[4]) + h[3] * (c[1] + c[5])
                s[3:-3] = (h[0] * c[3:-3] + h[1] * (c[2:-4] + c[4:-2]) + h[2] * (c[1:-5] + c[5:-1]) + h[3] * (c[0:-6] + c[6:]))
                s[-3] = h[0] * c[-3] + h[1] * (c[-4] + c[-2]) + h[2] * (c[-5] + c[-1]) + h[3] * (c[-6] + c[-2])
                s[-2] = h[0] * c[-2] + h[1] * (c[-3] + c[-1]) + h[2] * (c[-4] + c[-2]) + h[3] * (c[-5] + c[-3])
                s[-1] = h[0] * c[-1] + 2.0 * h[1] * c[-2] + 2.0 * h[2] * c[-3] + 2.0 * h[3] * c[-4]
            else:
                if len(c) == 5:
                    s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2] + 2.0 * h[3] * c[3]
                    s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3]) + h[3] * (c[2] + c[4])
                    s[2] = h[0] * c[2] + (h[1] + h[3]) * (c[1] + c[3]) + h[2] * (c[0] + c[4])
                    s[3] = h[0] * c[3] + h[1] * (c[2] + c[4]) + h[2] * (c[1] + c[3]) + h[3] * (c[0] + c[2])
                    s[4] = h[0] * c[4] + 2.0 * h[1] * c[3] + 2.0 * h[2] * c[2] + 2.0 * h[3] * c[1]
                elif len(c) == 4:
                    s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2] + 2.0 * h[3] * c[3]
                    s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3]) + 2.0 * h[3] * c[2]
                    s[2] = h[0] * c[2] + h[1] * (c[1] + c[3]) + h[2] * (c[0] + c[2]) + 2.0 * h[3] * c[1]
                    s[3] = h[0] * c[3] + 2.0 * h[1] * c[2] + 2.0 * h[2] * c[1] + 2.0 * h[3] * c[0]
                elif len(c) == 3:
                    s[0] = h[0] * c[0] + 2.0 * (h[1] + h[3]) * c[1] + 2.0 * h[2] * c[2]
                    s[1] = h[0] * c[1] + (h[1] + h[3]) * (c[0] + c[2]) + 2.0 * h[2] * c[1]
                    s[2] = h[0] * c[2] + 2.0 * (h[1] + h[3]) * c[1] + 2.0 * h[2] * c[0]
                elif len(c) == 2:
                    s[0] = (h[0] + 2.0 * h[2]) * c[0] + 2.0 * (h[1] + h[3]) * c[1]
                    s[1] = (h[0] + 2.0 * h[2]) * c[1] + 2.0 * (h[1] + h[3]) * c[0]
                elif len(c) == 1:
                    s[0] = (h[0] + 2.0 * (h[1] + h[2] + h[3])) * c[0]
                else:
                    raise ValueError("Invalid length of data")
        else:
            raise ValueError("Invalid filter half-length (should be [2..4])")
        

def resize_image(input_img, output_size=None, zoom_factors=None, method='Least-Squares', degree='Linear'):
    """
    Resize an image using Least-Squares or Oblique interpolation.

    Parameters:
    - input_img: numpy.ndarray
        The input image to resize.
    - output_size: tuple of int (output_height, output_width), optional
        The desired output size. If provided, zoom_factors are ignored.
    - zoom_factors: tuple of float (zoom_y, zoom_x), optional
        The zoom factors for the y and x dimensions.
    - method: str, optional
        The interpolation method to use. Options are 'Least-Squares' or 'Oblique'.
    - degree: str, optional
        The degree of the interpolation. Options are 'Linear' or 'Cubic'.

    Returns:
    - output_img: numpy.ndarray
        The resized image.
    """
    # Determine the zoom factors
    if output_size is not None:
        zoom_y = output_size[0] / input_img.shape[0]
        zoom_x = output_size[1] / input_img.shape[1]
    elif zoom_factors is not None:
        zoom_y, zoom_x = zoom_factors
    else:
        raise ValueError("Either output_size or zoom_factors must be provided.")

    # Map the method to analy_degree and synthe_degree
    if method == 'Least-Squares':
        analy_degree = -1
    elif method == 'Oblique':
        if degree == 'Linear':
            analy_degree = 1
        elif degree == 'Cubic':
            analy_degree = 3
        else:
            raise ValueError("Invalid degree for Oblique method. Choose 'Linear' or 'Cubic'.")
    else:
        raise ValueError("Invalid method. Choose 'Least-Squares' or 'Oblique'.")

    # Map degree to interp_degree and synthe_degree
    if degree == 'Linear':
        interp_degree = 1
        synthe_degree = 1
    elif degree == 'Cubic':
        interp_degree = 3
        synthe_degree = 3
    else:
        raise ValueError("Invalid degree. Choose 'Linear' or 'Cubic'.")

    # Shifts (can be adjusted if needed)
    shift_y = 0.0
    shift_x = 0.0

    # Create the output image array
    output_shape = (int(round(input_img.shape[0] * zoom_y)), int(round(input_img.shape[1] * zoom_x)))
    output_img = np.zeros(output_shape, dtype=input_img.dtype)

    # Create an instance of the Resize class
    resizer = Resize()

    # Perform the resizing operation
    resizer.compute_zoom(
        input_img,
        output_img,
        analy_degree,
        synthe_degree,
        interp_degree,
        zoom_y,
        zoom_x,
        shift_y,
        shift_x,
        inversable=False
    )

    return output_img