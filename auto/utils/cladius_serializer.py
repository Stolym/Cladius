import base64
import numpy as np

def ndarray_to_base64(ndarray):
    return base64.b64encode(ndarray.tobytes()).decode('utf-8')

def base64_to_ndarray(b64_string, dtype, shape):
    bytes = base64.b64decode(b64_string)
    return np.frombuffer(bytes, dtype=dtype).reshape(shape)
