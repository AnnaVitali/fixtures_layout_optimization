
def line_equation(x1, y1, x2, y2):
    """
    Compute the m and q of a line passing through two points

    :param x1: the x of the first point
    :param y1: the y of the first point
    :param x2: the x of the second point
    :param y2: the y of the second point
    :return: the angular coefficient m and the known term q
    """
    m = (y2 - y1) / (x2 - x1)
    q = y1 - m * x1
    return m, q
