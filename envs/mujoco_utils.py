def euclidian_dist(x_pos, y_pos, x_center, y_center):
    return ((x_pos - x_center) ** 2 + (y_pos - y_center) ** 2) ** 0.5

def in_range(x_pos, y_pos, x_center, y_center, radius):
    return euclidian_dist(x_pos, y_pos, x_center, y_center) <= radius
