#@author: Alisdair Cameron, Xinqi Zhu
#@Date: 2019-04-25
#@Editor: atom

import numpy as np
import pdb


def fill_line(coords, flip=True):
    """
    fill holes between scattered lane pixels to form a connected lane line in groud truth
    Method: every 2 successive ground truth points give a line

    @params
    coords: scattered lane pixels' coordinates

    @return
    new_coords: coordinates after connecting

    @example usage:
    coordinates =[[(x, y) for (x, y) in zip (row, col) if x >= 0] for row in rows]
    connected_coords = [fill_line(coord) for coord in coordinates if coord]
    """
    # unzip coords into x and y
    c = list(zip(*coords))

    x = c[0]
    y = c[1]

    #init new coords
    x_n = np.array([])
    y_n = np.array([])

    for index in range(len(x)-1):
        x1,y1 = (x[index], y[index])
        x2,y2 = (x[index+1], y[index+1])
        line_coef = np.polyfit([x1,x2], [y1,y2], 1)
        f_line = np.poly1d(line_coef)

        x_min = min(x1, x2)
        x_max = max(x1, x2)
        xs = np.array([i for i in range(x_min, x_max+1)])
        ys = np.array([f_line(i) for i in xs])
        x_n = np.concatenate((x_n, xs))
        y_n = np.concatenate((y_n, ys))

        y_min = min(y1, y2)
        y_max = max(y1, y2)
        if flip and x_max-x_min<2 and y_max - y_min >=2:
            line_flipped_coef = np.polyfit([y1,y2], [x1, x2], 1)
            f_flipped_line = np.poly1d(line_flipped_coef)

            ys_flipped = np.array([i for i in range(y_min, y_max+1)])
            xs_flipped = np.array([f_flipped_line(i) for i in ys_flipped])

            y_n = np.concatenate((y_n, ys_flipped))
            x_n = np.concatenate((x_n, xs_flipped))

    # zip coordinates back together
    x_n = x_n.astype(int)
    y_n = y_n.astype(int)
    new_coords = list(zip(x_n, y_n))
    new_coords = list(set(new_coords))
    return new_coords
