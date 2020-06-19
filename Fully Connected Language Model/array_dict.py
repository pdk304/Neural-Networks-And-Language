class ArrayDict(dict):
    """
    A container that is like a dict, but can be multiplied by a scalar
    and added to other ArrayDicts. Each ArrayDict contains NumPy arrays,
    and two ArrayDicts can be added if their keys and value shapes are
    compatible.
    """

    def __mul__(self, a: float):
        """
        Scalar multiplication.

        :param a: A scalar
        :return: This object with values scaled by a
        """
        return ArrayDict({k: a * self[k] for k in self})

    def __rmul__(self, a: float):
        return self * a

    def __add__(self, other):
        """
        Addition.

        :param other: Another ArrayDict
        :return: The entry-wise sum of this and the other object
        """
        return ArrayDict({k: self[k] + other[k] for k in self})

    def __sub__(self, other):
        """
        Subtraction.

        :param other: Another ArrayDict
        :return: The entry-wise difference between this and the other
            object
        """
        return ArrayDict({k: self[k] - other[k] for k in self})
