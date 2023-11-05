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
    
class InitNodeFrame:
    def __init__(self):
        self.frames = {}
        self.thisFrame = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": ("INT", {"default": 0, "min": 0}),
                "positive_text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "negative_text": ("STRING", {"multiline": True}),
                "general_positive": ("STRING", {"multiline": True}),
                "general_negative": ("STRING", {"multiline": True}),
                "previous_frame": ("FIZZFRAME", {"forceInput": True}),
                "clip": ("CLIP",),
            }
        }
    RETURN_TYPES = ("FIZZFRAME","CONDITIONING","CONDITIONING",)
    FUNCTION = "create_frame"

    CATEGORY = "FizzNodes/CustomNodes"

    def create_frame(self, frame, positive_text, negative_text=None, general_positive=None, general_negative=None, previous_frame=None, clip=None):
        new_frame = {
            "positive_text": positive_text,
            "negative_text": negative_text,
        }

        if previous_frame:
            prev_frame = previous_frame.thisFrame
            print(f"Previous Frame Found {prev_frame['general_positive']}")
            new_frame["general_positive"] = prev_frame["general_positive"]
            new_frame["general_negative"] = prev_frame["general_negative"]
            new_frame["clip"] = prev_frame["clip"]
            self.frames = previous_frame.frames

        if general_positive:
            print(f"General positive {general_positive}")
            new_frame["general_positive"] = general_positive
        
        if general_negative:
            new_frame["general_negative"] = general_negative

        new_positive_text = f"{positive_text}, {new_frame['general_positive']}"
        new_negative_text = f"{negative_text}, {new_frame['general_negative']}"

        if clip:
            new_frame["clip"] = clip 

        print(f"Positive Text: {new_positive_text}") 
        print(f"Negative Text: {new_negative_text}") 

        pos_tokens = new_frame["clip"].tokenize(new_positive_text)        
        pos_cond, pos_pooled = new_frame["clip"].encode_from_tokens(pos_tokens, return_pooled=True)
        new_frame["pos_conditioning"] = {"cond": pos_cond, "pooled": pos_pooled}

        neg_tokens = new_frame["clip"].tokenize(new_negative_text)
        neg_cond, neg_pooled = new_frame["clip"].encode_from_tokens(neg_tokens, return_pooled=True)
        new_frame["neg_conditioning"] = {"cond": neg_cond, "pooled": neg_pooled}

        self.frames[frame] = new_frame
        self.thisFrame = new_frame

        return (self, [[pos_cond, {"pooled_output": pos_pooled}]], [[neg_cond, {"pooled_output": neg_pooled}]])

class NodeFrame:

    def __init__(self):
        self.frames = {}
        self.thisFrame = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": ("INT", {"default": 0, "min": 0}),
                "previous_frame": ("FIZZFRAME", {"forceInput": True}),
                "positive_text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "negative_text": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("FIZZFRAME","CONDITIONING","CONDITIONING",)
    FUNCTION = "create_frame"

    CATEGORY = "FizzNodes/CustomNodes"

    def create_frame(self, frame, previous_frame, positive_text, negative_text=None):
        self.frames = previous_frame.frames
        prev_frame = previous_frame.thisFrame
        
        new_positive_text = f"{positive_text}, {prev_frame['general_positive']}"
        new_negative_text = f"{negative_text}, {prev_frame['general_negative']}"

        pos_tokens = prev_frame["clip"].tokenize(new_positive_text)        
        pos_cond, pos_pooled = prev_frame["clip"].encode_from_tokens(pos_tokens, return_pooled=True)

        neg_tokens = prev_frame["clip"].tokenize(new_negative_text)
        neg_cond, neg_pooled = prev_frame["clip"].encode_from_tokens(neg_tokens, return_pooled=True)

        new_frame = {
            "positive_text": positive_text,
            "negative_text": negative_text,
            "general_positive": prev_frame["general_positive"],
            "general_negative": prev_frame["general_negative"],
            "clip": prev_frame["clip"],
            "pos_conditioning": {"cond": pos_cond, "pooled": pos_pooled},
            "neg_conditioning": {"cond": neg_cond, "pooled": neg_pooled},
        }
        self.thisFrame = new_frame
        self.frames[frame] = new_frame

        return (self, [[pos_cond, {"pooled_output": pos_pooled}]], [[neg_cond, {"pooled_output": neg_pooled}]])

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