""" Custom warnings for proba package"""


class NegativeKL(Warning):
    pass


class MissingShape(Warning):
    pass


class NegativeCov(Warning):
    pass


class ShapeWarning(Warning):
    pass


class ShapeMismatch(Exception):
    def __init__(self, shape, expected_shape):
        message = f"Expected shape {expected_shape}, got shape {shape}"
        super().__init__(message)


def check_shape(shape, target_shape):
    if shape != target_shape:
        raise ShapeMismatch(shape, target_shape)
