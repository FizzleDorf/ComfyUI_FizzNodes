import numpy as np
import pandas as pd
import re
import numexpr

class LerpNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"num_Images": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 1.0}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "current_frame": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999, "step": 1.0}),
                             }}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "lerp"
    
    CATEGORY = "FizzNodes"

    def lerp(self, num_Images, strength, current_frame):
        step = strength/num_Images
        output = strength - (step * current_frame)
        return (output, )