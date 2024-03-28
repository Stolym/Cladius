class ILoader:
    def __init__(self) -> None:
        assert self.load_data != None, "load_data method must be implemented"
        data_x, data_y, data_x_shape, data_y_shape, dx_dtype, dy_dtype = self.load_data()

        assert data_x is not None, "data_x must be implemented"
        assert data_y is not None, "data_y must be implemented"

        self.data = {
            "data_x": data_x,
            "data_x_shape": data_x_shape,
            "data_x_dtype": str(dx_dtype),
            "data_y": data_y,
            "data_y_shape": data_y_shape,
            "data_y_dtype": str(dy_dtype),
        }