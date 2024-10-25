# utils.py
import numpy as np

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
