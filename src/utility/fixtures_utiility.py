def square_suction_cup(lb_x, lb_y):
    square_side = 145

    square_coords = [
        (lb_x, lb_y),
        (lb_x, lb_y + square_side),
        (lb_x + square_side, lb_y + square_side),
        (lb_x + square_side, lb_y),
        (lb_x, lb_y),  # Close the square
    ]
    return zip(*square_coords)


def rectangle_suction_cup(lb_x, lb_y):
    width = 145
    height = 55

    rectangle_coords = [
        (lb_x, lb_y),
        (lb_x, lb_y + height),
        (lb_x + width, lb_y + height),
        (lb_x + width, lb_y),
        (lb_x, lb_y),  # Close the square
    ]
    return zip(*rectangle_coords)


def create_fixture(cup_type, lb_x, lb_y):
    if cup_type == 1:
        return list(square_suction_cup(lb_x, lb_y))
    elif cup_type == 2:
        return list(rectangle_suction_cup(lb_x, lb_y))
    else:
        raise ValueError("Invalid cup type: must be 1 (square) or 2 (rectangle)")