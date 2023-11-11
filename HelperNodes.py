

class CalculateLatentInterpFrameNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_frame": ("INT", {"default": 0, "min": 0}),
                "max_frames": ("INT", {"default": 18, "min": 0}),
                "num_latent_inputs": ("INT", {"default": 4, "min": 0}),
            }
        }
    RETURN_TYPES = ("INT", "INT" ,"INT" ,"INT", )
    FUNCTION = "assignFrameNum"

    CATEGORY = "FizzNodes 📅🅕🅝/FrameNodes"

    def assignFrameNum(self, current_frame, max_frames, batch_size):
        if current_frame == 0:
            return 0, 1, 2, 3
        else:
            start_frame = (current_frame - 1) * (batch_size - 1) + 3
            print(tuple((start_frame + i) % max_frames for i in range(4)))
            return tuple((start_frame + i) % max_frames for i in range(4))