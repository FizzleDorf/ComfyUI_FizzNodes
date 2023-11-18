
class CalculateFrameOffset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_frame": ("INT", {"default": 0, "min": 0}),
                "max_frames": ("INT", {"default": 18, "min": 0}),
                "num_latent_inputs": ("INT", {"default": 4, "min": 0}),
                "index": ("INT", {"default": 4, "min": 0}),
            }
        }
    RETURN_TYPES = ("INT", )
    FUNCTION = "assignFrameNum"

    CATEGORY = "FizzNodes 📅🅕🅝/HelperNodes"

    def assignFrameNum(self, current_frame, max_frames, num_latent_inputs, index):
        if current_frame == 0:
            return (index,)
        else:
            start_frame = (current_frame - 1) * (num_latent_inputs - 1) + (num_latent_inputs-1)
            return ((start_frame + index) % max_frames,)
class ConcatStringSingle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_a": ("STRING", {"forceInput":True,"default":"","multiline": True}),
                "string_b": ("STRING", {"forceInput":True,"default":"","multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING", )
    FUNCTION = "concat"

    CATEGORY = "FizzNodes 📅🅕🅝/HelperNodes"

    def concat(self, string_a, string_b):
        c = string_a + string_b
        return (c,)

class convertKeyframeKeysToBatchKeys:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", {"forceInput": True, "default": 0}),
                "num_latents": ("INT", {"default": 16}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "concat"

    CATEGORY = "FizzNodes 📅🅕🅝/HelperNodes"

    def concat(self, input, num_latents):
        c = input * num_latents -1
        return (c,)