SQUARE_CUP_DIM = 145
RECTANGULAR_CUP_DIM_X = 145
RECTANGULAR_CUP_DIM_Y = 55

def square_suction_cup(lb_x, lb_y):
    square_coords = [
        (lb_x, lb_y),
        (lb_x, lb_y + SQUARE_CUP_DIM),
        (lb_x + SQUARE_CUP_DIM, lb_y + SQUARE_CUP_DIM),
        (lb_x + SQUARE_CUP_DIM, lb_y),
        (lb_x, lb_y),
    ]
    return zip(*square_coords)


def rectangle_suction_cup(lb_x, lb_y):
    rectangle_coords = [
        (lb_x, lb_y),
        (lb_x, lb_y + RECTANGULAR_CUP_DIM_Y),
        (lb_x + RECTANGULAR_CUP_DIM_X, lb_y + RECTANGULAR_CUP_DIM_Y),
        (lb_x + RECTANGULAR_CUP_DIM_X, lb_y),
        (lb_x, lb_y),
    ]
    return zip(*rectangle_coords)


def create_fixture(cup_type, lb_x, lb_y):
    if cup_type == 1:
        return list(square_suction_cup(lb_x, lb_y))
    elif cup_type == 2:
        return list(rectangle_suction_cup(lb_x, lb_y))
    else:
        raise ValueError("Invalid cup type: must be 1 (square) or 2 (rectangle)")