import numpy as np

class Resize:
    """
    A class to perform image resizing using various interpolation methods.

    Attributes
    ----------
    interp_degree : int
        Degree of the interpolation.
    analy_degree : int
        Degree of the analysis.
    synthe_degree : int
        Degree of the synthesis.
    zoom_y : float
        Vertical zoom factor.
    zoom_x : float
        Horizontal zoom factor.
    inversable : bool
        Indicates if the transformation is inversable.
    analy_even : int
        Indicates if the analysis degree is even.
    corr_degree : int
        Degree of the correlation.
    half_support : float
        Half support size for the spline interpolation.
    spline_array_height : np.ndarray
        Array of spline coefficients for height.
    spline_array_width : np.ndarray
        Array of spline coefficients for width.
    index_min_height : np.ndarray
        Minimum index for height.
    index_max_height : np.ndarray
        Maximum index for height.
    index_min_width : np.ndarray
        Minimum index for width.
    index_max_width : np.ndarray
        Maximum index for width.
    tolerance : float
        Tolerance for numerical calculations.
    """

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

        nx, ny = input_img.shape[1], input_img.shape[0]

        size = self.calculate_final_size(inversable, ny, nx, zoom_y, zoom_x)
        working_size_x, working_size_y = size[1], size[0]
        final_size_x, final_size_y = size[3], size[2]

        self.analy_even = int((analy_degree + 1) % 2 == 0)

        total_degree = interp_degree + analy_degree + 1
        self.corr_degree = analy_degree + synthe_degree + 1
        self.half_support = (total_degree + 1) / 2.0

        add_border_height = self.border(final_size_y, self.corr_degree)
        if add_border_height < total_degree:
            add_border_height += total_degree

        final_total_height = final_size_y + add_border_height
        length_total_height = working_size_y + int(np.ceil(add_border_height / zoom_y))

        self.index_min_height = np.zeros(final_total_height, dtype=int)
        self.index_max_height = np.zeros(final_total_height, dtype=int)
        length_array_spln_height = final_total_height * (2 + total_degree)
        self.spline_array_height = np.zeros(length_array_spln_height)

        shift_y += ((analy_degree + 1.0) / 2.0 - np.floor((analy_degree + 1.0) / 2.0)) * (1.0 / zoom_y - 1.0)
        fact_height = np.float64(np.power(zoom_y, analy_degree + 1))

        i = 0
        for l in range(final_total_height):
            affine_index = 1.0 * l / zoom_y + shift_y
            self.index_min_height[l] = int(np.ceil(affine_index - self.half_support))
            self.index_max_height[l] = int(np.floor(affine_index + self.half_support))
            for k in range(self.index_min_height[l], self.index_max_height[l] + 1):
                self.spline_array_height[i] = fact_height * self.beta(affine_index - 1.0 * k, total_degree)
                i += 1

        add_border_width = self.border(final_size_x, self.corr_degree)
        if add_border_width < total_degree:
            add_border_width += total_degree

        final_total_width = final_size_x + add_border_width
        length_total_width = working_size_x + int(np.ceil(add_border_width / zoom_x))

        self.index_min_width = np.zeros(final_total_width, dtype=int)
        self.index_max_width = np.zeros(final_total_width, dtype=int)
        length_array_spln_width = final_total_width * (2 + total_degree)
        self.spline_array_width = np.zeros(length_array_spln_width)

        shift_x += ((analy_degree + 1.0) / 2.0 - np.floor((analy_degree + 1.0) / 2.0)) * (1.0 / zoom_x - 1.0)
        fact_width = np.power(zoom_x, analy_degree + 1)

        i = 0
        for l in range(final_total_width):
            affine_index = 1.0 * l / zoom_x + shift_x
            self.index_min_width[l] = int(np.ceil(affine_index - self.half_support))
            self.index_max_width[l] = int(np.floor(affine_index + self.half_support))
            for k in range(self.index_min_width[l], self.index_max_width[l] + 1):
                self.spline_array_width[i] = fact_width * self.beta(affine_index - 1.0 * k, total_degree)
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
        """
        Perform resampling of a row vector.

        Parameters
        ----------
        input_vector : np.ndarray
            The input row vector.
        output_vector : np.ndarray
            The output row vector.
        add_vector : np.ndarray
            Auxiliary vector for processing.
        add_output_vector : np.ndarray
            Auxiliary output vector for processing.
        max_sym_boundary : int
            Maximum symmetric boundary.
        max_asym_boundary : int
            Maximum asymmetric boundary.
        """
        length_input = len(input_vector)
        length_output = len(output_vector)
        length_total = len(add_vector)
        length_output_total = len(add_output_vector)
        average = 0

        # Projection Method
        if self.analy_degree != -1:
            average = self.do_integ(input_vector, self.analy_degree + 1)

        add_vector[:length_input] = input_vector

        for l in range(length_input, length_total):
            l2 = 2 * length_input - l - 2
            if l2 < 0:
                l2 = -l2
            if self.analy_even == 1:
                add_vector[l] = add_vector[l2]
            else:
                add_vector[l] = -add_vector[l2]

        i = 0

        for l in range(length_output_total):
            add_output_vector[l] = 0.0
            for k in range(self.index_min_width[l], self.index_max_width[l] + 1):
                index = k
                sign = 1
                if k < 0:
                    index = -k
                    if self.analy_even == 0:
                        index -= 1
                        sign = -1
                if k >= length_total:
                    index = 2 * length_total - k - 2
                    if index >= length_total:
                        index %= length_total
                    if self.analy_even == 0:
                        sign = -sign

                # Geometric transformation and resampling
                add_output_vector[l] += sign * add_vector[index] * self.spline_array_width[i]
                i += 1

        # Projection Method
        if self.analy_degree != -1:
            # Differentiation analy_degree + 1 times of the signal
            self.do_diff(add_output_vector, self.analy_degree + 1)
            for i in range(length_output_total):
                add_output_vector[i] += average
            # IIR filtering
            self.get_interpolation_coefficients(add_output_vector, self.corr_degree)
            # Samples
            self.get_samples(add_output_vector, self.synthe_degree)

        output_vector[:length_output] = add_output_vector[:length_output]

    def resampling_column(self, input_vector, output_vector, add_vector, add_output_vector, max_sym_boundary, max_asym_boundary):
        """
        Perform resampling of a column vector.

        Parameters
        ----------
        input_vector : np.ndarray
            The input column vector.
        output_vector : np.ndarray
            The output column vector.
        add_vector : np.ndarray
            Auxiliary vector for processing.
        add_output_vector : np.ndarray
            Auxiliary output vector for processing.
        max_sym_boundary : int
            Maximum symmetric boundary.
        max_asym_boundary : int
            Maximum asymmetric boundary.
        """
        length_input = len(input_vector)
        length_output = len(output_vector)
        length_total = len(add_vector)
        length_output_total = len(add_output_vector)
        average = 0

        # Projection Method
        if self.analy_degree != -1:
            average = self.do_integ(input_vector, self.analy_degree + 1)

        add_vector[:length_input] = input_vector

        for l in range(length_input, length_total):
            l2 = 2 * length_input - l - 2
            if l2 < 0:
                l2 = -l2
            if self.analy_even == 1:
                add_vector[l] = add_vector[l2]
            else:
                add_vector[l] = -add_vector[l2]

        i = 0

        for l in range(length_output_total):
            add_output_vector[l] = 0.0
            for k in range(self.index_min_height[l], self.index_max_height[l] + 1):
                index = k
                sign = 1
                if k < 0:
                    index = -k
                    if self.analy_even == 0:
                        index -= 1
                        sign = -1
                if k >= length_total:
                    index = 2 * length_total - k - 2
                    if index >= length_total:
                        index %= length_total
                    if self.analy_even == 0:
                        sign = -sign

                # Geometric transformation and resampling
                add_output_vector[l] += sign * add_vector[index] * self.spline_array_height[i]
                i += 1

        # Projection Method
        if self.analy_degree != -1:
            # Differentiation analy_degree + 1 times of the signal
            self.do_diff(add_output_vector, self.analy_degree + 1)
            for i in range(length_output_total):
                add_output_vector[i] += average
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
        """
        Perform integration on a vector.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        nb : int
            Number of integration steps.

        Returns
        -------
        average : float
            The average value after integration.
        """
        size = len(c)
        m = 0.0
        average = np.float64(0.0)

        if nb == 1:
            for f in range(size):
                average += np.float64(c[f])
            average = (2.0 * average - c[size - 1] - c[0]) / (2 * size - 2)
            self.integ_sa(c, average)

        elif nb == 2:
            for f in range(size):
                average += c[f]
            average = (2.0 * average - c[size - 1] - c[0]) / (2 * size - 2)
            self.integ_sa(c, average)
            self.integ_as(c, c)

        elif nb == 3:
            for f in range(size):
                average += c[f]
            average = (2.0 * average - c[size - 1] - c[0]) / (2 * size - 2)
            self.integ_sa(c, average)
            self.integ_as(c, c)
            for f in range(size):
                m += c[f]
            m = (2.0 * m - c[size - 1] - c[0]) / (2 * size - 2)
            self.integ_sa(c, m)

        elif nb == 4:
            for f in range(size):
                average += c[f]
            average = (2.0 * average - c[size - 1] - c[0]) / (2 * size - 2)
            self.integ_sa(c, average)
            self.integ_as(c, c)
            for f in range(size):
                m += c[f]
            m = (2.0 * m - c[size - 1] - c[0]) / (2 * size - 2)
            self.integ_sa(c, m)
            self.integ_as(c, c)

        return average

    def integ_sa(self, c, m):
        """
        Perform semi-analytical integration.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        m : float
            The average value.
        """
        size = len(c)
        c[0] = (c[0] - m) * 0.5
        for i in range(1, size):
            c[i] = c[i] - m + c[i - 1]

    def integ_as(self, c, y):
        """
        Perform analytical integration.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        y : np.ndarray
            The output vector after integration.
        """
        size = len(c)
        z = c.copy()
        y[0] = z[0]
        y[1] = 0
        for i in range(2, size):
            y[i] = y[i - 1] - z[i - 1]

    def do_diff(self, c, nb):
        """
        Perform differentiation on a vector.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        nb : int
            Number of differentiation steps.
        """
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
        """
        Perform semi-analytical differentiation.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        """
        size = len(c)
        old = c[size - 2]
        for i in range(size - 1):
            c[i] = np.float64(c[i] - c[i + 1])
        c[size - 1] = c[size - 1] - old

    def diff_as(self, c):
        """
        Perform analytical differentiation.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        """
        size = len(c)
        for i in range(size - 1, 0, -1):
            c[i] = c[i] - c[i - 1]
        c[0] = 2.0 * c[0]

    @staticmethod
    def border(size, degree):
        """
        Calculate the border size for a given degree.

        Parameters
        ----------
        size : int
            The size of the dimension.
        degree : int
            The degree of the spline.

        Returns
        -------
        horizon : int
            The calculated border size.
        """
        if degree in [0, 1]:
            return 0

        tolerance = 1e-10
        if degree == 2:
            z = (8.0 ** 0.5) - 3.0
        elif degree == 3:
            z = (3.0 ** 0.5) - 2.0
        elif degree == 4:
            z = (664.0 - (438976.0 ** 0.5)) ** 0.5 + (304.0 ** 0.5) - 19.0
        elif degree == 5:
            z = ((135.0 / 2.0 - (17745.0 / 4.0) ** 0.5) ** 0.5
                + (105.0 / 4.0) ** 0.5 - 13.0 / 2.0)
        elif degree == 6:
            z = -0.488294589303044755130118038883789062112279161239377608394
        elif degree == 7:
            z = -0.5352804307964381655424037816816460718339231523426924148812
        else:
            raise ValueError("Invalid degree (should be [0..7])")

        horizon = np.int64(2 + int(np.log(tolerance) / np.log(abs(z))))
        horizon = min(horizon, size)
        return horizon

    @staticmethod
    def calculate_final_size(inversable, height, width, zoom_y, zoom_x):
        """
        Calculate the final size of the image after zooming.

        Parameters
        ----------
        inversable : bool
            Indicates if the transformation is inversable.
        height : int
            The height of the input image.
        width : int
            The width of the input image.
        zoom_y : float
            Vertical zoom factor.
        zoom_x : float
            Horizontal zoom factor.

        Returns
        -------
        size : list
            A list containing the sizes [height, width, final_height, final_width].
        """
        size = [height, width, 0, 0]

        if inversable:
            w2 = round(round((size[0] - 1) * zoom_y) / zoom_y)
            while size[0] - 1 - w2 != 0:
                size[0] += 1
                w2 = round(round((size[0] - 1) * zoom_y) / zoom_y)

            h2 = round(round((size[1] - 1) * zoom_x) / zoom_x)
            while size[1] - 1 - h2 != 0:
                size[1] += 1
                h2 = round(round((size[1] - 1) * zoom_x) / zoom_x)

            size[2] = round((size[0] - 1) * zoom_y) + 1
            size[3] = round((size[1] - 1) * zoom_x) + 1
        else:
            size[2] = round(size[0] * zoom_y)
            size[3] = round(size[1] * zoom_x)

        return size

    def get_interpolation_coefficients(self, c, degree):
        """
        Get the interpolation coefficients for a given vector and degree.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        degree : int
            The degree of the spline.
        """
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

        for zk in z:
            lambda_ *= (1.0 - zk) * (1.0 - 1.0 / zk)

        for n in range(len(c)):
            c[n] *= lambda_

        for zk in z:
            c[0] = np.float64(self.get_initial_causal_coefficient(c, zk, tolerance))
            for n in range(1, len(c)):
                c[n] = np.float64(c[n] + zk * c[n - 1])
            c[-1] = self.get_initial_anti_causal_coefficient(c, zk, tolerance)
            for n in range(len(c) - 2, -1, -1):
                c[n] = np.float64(zk * (c[n + 1] - c[n]))

    def get_samples(self, c, degree):
        """
        Get the samples for a given vector and degree.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        degree : int
            The degree of the spline.
        """
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
        """
        Get the initial anti-causal coefficient for a given vector and z value.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        z : float
            The z value.
        tolerance : float
            The tolerance for numerical calculations.

        Returns
        -------
        float
            The initial anti-causal coefficient.
        """
        return (z * c[-2] + c[-1]) * z / (z * z - 1.0)

    @staticmethod
    def get_initial_causal_coefficient(c, z, tolerance):
        """
        Get the initial causal coefficient for a given vector and z value.

        Parameters
        ----------
        c : np.ndarray
            The input vector.
        z : float
            The z value.
        tolerance : float
            The tolerance for numerical calculations.

        Returns
        -------
        float
            The initial causal coefficient.
        """
        z1 = z
        zn = z ** (len(c) - 1)
        sum_ = c[0] + zn * c[-1]
        horizon = len(c)

        if tolerance > 0.0:
            horizon = 2 + int(np.log(tolerance) / np.log(np.abs(z)))
            horizon = min(horizon, len(c))

        zn = zn * zn
        for n in range(1, horizon - 1):
            zn = zn / z
            sum_ += (z1 + zn) * c[n]
            z1 = z1 * z

        return sum_ / (1.0 - z ** (2 * len(c) - 2))

    @staticmethod
    def symmetric_fir(h, c, s):
        """
        Perform symmetric FIR filtering.

        Parameters
        ----------
        h : list
            The filter coefficients.
        c : np.ndarray
            The input vector.
        s : np.ndarray
            The output vector after filtering.
        """
        if len(c) != len(s):
            raise IndexError("Incompatible size")

        if len(h) == 2:
            if len(c) >= 2:
                s[0] = h[0] * c[0] + 2.0 * h[1] * c[1]
                for i in range(1, len(c) - 1):
                    s[i] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1])
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
                for i in range(2, len(c) - 2):
                    s[i] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1]) + h[2] * (c[i - 2] + c[i + 2])
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
                for i in range(3, len(c) - 3):
                    s[i] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1]) + h[2] * (c[i - 2] + c[i + 2]) + h[3] * (c[i - 3] + c[i + 3])
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
