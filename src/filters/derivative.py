import numpy as np

def derivative(img:np.ndarray, order:int) -> np.ndarray:
    """ Compute the discrete derivative of an image along the horizontal direction.

    Parameters
    ----------
    img : np.ndarray
        The input image, of shape (H, W)
    order : int
        The order of the derivative
    
    Returns
    -------
    np.ndarray
        The derivative image of shape (H, W-order)
    """

    assert len(img.shape) == 2, "Input image must be 2D, got shape {}".format(img.shape)
    assert order >= 0, "Order of derivative must be non-negative, got {}".format(order)

    if order == 0:
        return img
    else:
        # Recursive computation: Deriv(img, n) = Deriv(Deriv(img, n-1))
        img_n_1 = derivative(img, order-1)
        img_n = np.zeros((img_n_1.shape[0], img_n_1.shape[1]-1))
        img_n = img_n_1[:, 1:] - img_n_1[:, :-1]
        return img_n


def test_derivative():
    img = np.array([[1, 2, 4, 7],
                    [0, 1, 3, 6],
                    [5, 6, 8, 11]], dtype=np.float32)
    
    first_order = derivative(img, 1)
    expected_first_order = np.array([[1, 2, 3],
                                     [1, 2, 3],
                                     [1, 2, 3]], dtype=np.float32)
    assert np.array_equal(first_order, expected_first_order), "First order derivative test failed."

    second_order = derivative(img, 2)
    expected_second_order = np.array([[1, 1],
                                      [1, 1],
                                      [1, 1]], dtype=np.float32)
    assert np.array_equal(second_order, expected_second_order), "Second order derivative test failed."

    print("All derivative tests passed.")


if __name__ == "__main__":
    test_derivative()
