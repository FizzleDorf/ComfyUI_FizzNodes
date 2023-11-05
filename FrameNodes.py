class StringConcatenate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_a": ("STRING", {"forceInput": True}),
                "text_b": ("STRING", {"forceInput": True}),
                "frame_b": ("INT", {"default": 32, "min": 1})
            },
            "optional": {
                "text_c": ("STRING", {"forceInput": True}),
                "frame_c": ("INT", {"min": 2}),
                "text_d": ("STRING", {"forceInput": True}),
                "frame_d": ("INT", {"min": 3}),
                "text_e": ("STRING", {"forceInput": True}),
                "frame_e": ("INT", {"min": 4}),
                "text_f": ("STRING", {"forceInput": True}),
                "frame_f": ("INT", {"min": 5}),
                "text_g": ("STRING", {"forceInput": True}),
                "frame_g": ("INT", {"min": 6})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "frame_concatenate_list"

    CATEGORY = "FizzNodes/CustomNodes"

    def frame_concatenate_list(self, text_a, text_b, frame_b, text_c=None, frame_c=None, text_d=None, frame_d=None, text_e=None, frame_e=None, text_f=None, frame_f=None, text_g=None, frame_g=None):
        
        text_list = f'"0": "{text_a}",\n'
        text_list += f'"{frame_b}": "{text_b}",\n'
        if frame_c and text_c:
            text_list += f'"{frame_c}": "{text_c}",\n'
        if frame_d and text_d:
            text_list += f'"{frame_d}": "{text_d}",\n'
        if frame_e and text_e:
            text_list += f'"{frame_e}": "{text_e}",\n'
        if frame_f and text_f:
            text_list += f'"{frame_f}": "{text_f}",\n'
        if frame_g and text_g:
            text_list += f'"{frame_g}": "{text_g}",\n'
    
        text_list = text_list[:-2]

        print(text_list)

        return (text_list,)
    
class NodeFrame:

    def __init__(self):
        self.frames = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_text": ("STRING", {"multiline": True}),
                "frame": ("INT", {"default": 0, "min": 0})
            },
            "optional": {
                "negative_text": ("STRING", {"multiline": True}),
                "general_negative": ("STRING", {"multiline": True}),
                "general_positive": ("STRING", {"multiline": True}),
                "previous_frame": ("FIZZFRAME", {"forceInput": True}),
            }
        }
    RETURN_TYPES = ("FIZZFRAME","STRING", "STRING",)
    FUNCTION = "create_frame"

    CATEGORY = "FizzNodes/CustomNodes"

    def create_frame(self, positive_text, frame, negative_text=None, general_negative=None, general_positive=None, previous_frame=None):
        new_frame = {
            "positive_text": positive_text,
            "negative_text": negative_text
        }

        if previous_frame:
            new_frame.general_positive = previous_frame.general_positive
            new_frame.general_negative = previous_frame.general_negative

        if general_positive:
            new_frame.general_positive = general_positive
        
        if general_negative:
            new_frame.general_negative = general_negative

        self.frames[frame] = new_frame
        new_positive_text = f"{positive_text}, {general_positive}"
        new_negative_text = f"{negative_text}, {general_negative}"

        return (self, new_positive_text, new_negative_text,)

class FrameConcatenate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": ("FIZZFRAME", {"forceInput": True})
            },
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "frame_concatenate"

    CATEGORY = "FizzNodes/CustomNodes"

    def frame_concatenate(self, frame):
        text_list = ""
        for frame_digit in frame.frames:
            new_frame = frame.frames[frame_digit]
            text_list += f'"{frame_digit}": "{new_frame["positive_text"]}'
            if new_frame.get("general_positive"):
                text_list += f', {new_frame["general_positive"]}'
            if new_frame.get("negative_text") or new_frame.get("general_negative"):
                text_list += f', --neg '
                if new_frame.get("negative_text"):
                    text_list += f', {new_frame["negative_text"]}'
                if new_frame.get("general_negative"):
                    text_list += f', {new_frame["general_negative"]}'
            text_list += f'",\n'
        text_list = text_list[:-2]

        print(text_list)

        return (text_list,)