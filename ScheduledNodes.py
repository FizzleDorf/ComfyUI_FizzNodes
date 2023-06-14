#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum
import numexpr
#import torch
import numpy as np
import pandas as pd
import re
import simplejson as json

# used by both nodes
def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

#This node parses the user's formatted prompt,
#sequences the current prompt,next prompt, and 
#conditioning strength, evalates expressions in 
#the prompts, and then returns either current, 
#next or averaged conditioning.
#class PromptSchedule:
#    @classmethod
#    def INPUT_TYPES(s):
#        return {"required": {"text": ("STRING", {"multiline": True}), 
#                            "clip": ("CLIP", ),
#                            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
#                            "current_frame": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0}),}}
#                            
#    RETURN_TYPES = ("CONDITIONING",)
#    FUNCTION = "animate"
#
#    CATEGORY = "FizzNodes/ScheduleNodes"
#
#    def animate(self, text, max_frames, current_frame, clip,):
#        #format the input so it's valid json
#        inputText = self.prepare_prompt(str("{"+text+"}"), max_frames, current_frame)
#        animation_prompts = json.loads(inputText.strip())
#        #inputText = str("{"+text+"}")
#        #interpolate values in-between keyframes
#        #evaluate expressions (if any) in the prompt
#        print("animation_prompts", animation_prompts)
#        #return a conditioning value
#        return self.interpolate_prompts(animation_prompts, max_frames, int(current_frame), clip)    
#    
#
#    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength):
#        out = []
#        print(conditioning_to)
#        print(conditioning_from)
#        if len(conditioning_from) > 1:
#            print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")
#    
#        cond_from = np.array(conditioning_from[0][0])  # Convert to NumPy array
#        
#        for i in range(len(conditioning_to)):
#            t1 = np.array(conditioning_to[i][0])  # Convert to NumPy array
#            t0 = cond_from[:, :t1.shape[1]]
#            if t0.shape[1] < t1.shape[1]:
#                t0 = np.concatenate([t0] + [np.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], axis=1)
#    
#            tw = torch.mul(torch.tensor(t1), conditioning_to_strength) + torch.mul(torch.tensor(t0), (1.0 - conditioning_to_strength))
#            n = [tw, conditioning_to[i][1].copy()]
#            out.append(n)
#        return (out,)
#
#    #parse the conditioning strength and determine in-betweens.
#    def parse_weight(self, match, frame=0, max_frames=0) -> float:
#        w_raw = match.group("weight")
#        max_f = max_frames  # this line has to be left intact as it's in use by numexpr even though it looks like it doesn't
#        if w_raw is None:
#            return 1
#        if check_is_number(w_raw):
#            return float(w_raw)
#        else:
#            t = frame
#            if len(w_raw) < 3:
#                print('the value inside `-characters cannot represent a math function')
#                return 1
#            print(float(numexpr.evaluate(w_raw[1:-1])))
#            return float(numexpr.evaluate(w_raw[1:-1]))
#    
#        
#    def prepare_prompt(self, prompt_series, max_frames, frame_idx):
#        max_f = max_frames - 1
#        pattern = r'`.*?`'
#        regex = re.compile(pattern)
#        prompt_parsed = str(prompt_series)
#        print("prompt_parsed:",prompt_parsed)
#        for match in regex.finditer(prompt_parsed):
#            matched_string = match.group(0)
#            print("matched_string : ", matched_string)
#            parsed_string = matched_string.replace('t', f'{frame_idx}').replace("max_f", f"{max_f}").replace('`', '')
#            parsed_value = numexpr.evaluate(parsed_string)
#            print("value : ", parsed_value)
#            prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
#        
#        return prompt_parsed.strip()
#    
#    def interpolate_prompts(self, animation_prompts, max_frames, current_frame, clip):
#        # Get prompts sorted by keyframe
#        max_f = max_frames
#        parsed_animation_prompts = {}
#        for key, value in animation_prompts.items():
#            if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
#                parsed_animation_prompts[key] = value
#            else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
#                parsed_animation_prompts[int(numexpr.evaluate(key))] = value
#    
#        sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))
#    
#        # Setup container for interpolated prompts
#        curr_prompt_series = [np.nan] * max_frames
#        nxt_prompt_series = [np.nan] * max_frames
#        weight_series = [np.nan] * max_frames
#        
#        
#        # For every keyframe prompt except the last
#        for i in range(0, len(sorted_prompts) - 1):
#            # Get current and next keyframe
#            current_key = int(sorted_prompts[i][0])
#            print("curKey:"+str(current_key))
#            next_key = int(sorted_prompts[i + 1][0])
#            print("nxtKey:"+str(next_key))
#            # Ensure there's no weird ordering issues or duplication in the animation prompts
#            # (unlikely because we sort above, and the json parser will strip dupes)
#            if current_key >= next_key:
#                print(f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_frame} are not monotonously increasing; skipping interpolation.")
#                continue
#    
#            # Get current and next keyframes' positive and negative prompts (if any)
#            current_prompt = sorted_prompts[i][1]
#            next_prompt = sorted_prompts[i + 1][1]
#            
#            # Calculate how much to shift the weight from current to next prompt at each frame
#            weight_step = 1 / (next_key - current_key)
#    
#            for f in range(current_key, next_key):
#                next_weight = weight_step * (f - current_key)
#                current_weight = 1 - next_weight
#                print(current_key)
#                # We will build the prompt incrementally depending on which prompts are present
#                curr_prompt_series[f] = ''
#                nxt_prompt_series[f] = ''
#                weight_series[f] = 0.0
#                
#                curr_prompt_series[f] += self.prepare_prompt(current_prompt, max_frames, f)
#                nxt_prompt_series[f] += self.prepare_prompt(next_prompt, max_frames, f)
#                weight_series[f] += current_weight
#                print(curr_prompt_series)                
#                print(nxt_prompt_series)                
#                print(weight_series)                
#                #([[clip.encode(text), {}]], )                
#                print(curr_prompt_series[f])
#                print(nxt_prompt_series[f])
#                print(weight_series[f])
#                #([[clip.encode(text), {}]], )
#            # Return the filled series, in case max_frames is greater than the last keyframe or any ranges were skipped.
#        if str(curr_prompt_series[current_frame]) == str(nxt_prompt_series[current_frame]):
#            print("only curr")
#            return ([[clip.encode(str(curr_prompt_series[current_frame])), {}]], )
#        if weight_series[current_frame] == 0:
#            print("only curr")
#            return ([[clip.encode(str(curr_prompt_series[current_frame])), {}]], )
#        if weight_series[current_frame] == 1:
#            print("only next")
#            return ([[clip.encode(str(nxt_prompt_series[current_frame])), {}]], )
#        else:
#            print("both")
#            return self.addWeighted(([[clip.encode(str(nxt_prompt_series[current_frame])), {}]], ), ([[clip.encode(str(curr_prompt_series[current_frame])), {}]], ), weight_series[current_frame])
#

#This node parses the user's test input into interpolated floats. 
#Expressions can be input and evaluated.            
class ValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
                             "current_frame": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0}),}}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"
    
    def animate(self, text, max_frames, current_frame,):
        t = self.get_inbetweens(self.parse_key_frames(text, max_frames), max_frames)
        cFrame = int(current_frame) #current_frame is currently recieved as a float hence the cast to int
        return (t[cFrame],)
    
    def sanitize_value(self, value):
        return value.replace("'","").replace('"',"").replace('(',"").replace(')',"")

    def get_inbetweens(self, key_frames, max_frames, integer=False, interp_method='Linear', is_single_string = False):
        key_frame_series = pd.Series([np.nan for a in range(max_frames)])
        # get our ui variables set for numexpr.evaluate
        max_f = max_frames -1
        value_is_number = False
        for i in range(0, max_frames):
            if i in key_frames:
                value = key_frames[i]
                value_is_number = check_is_number(self.sanitize_value(value))
                if value_is_number: # if it's only a number, leave the rest for the default interpolation
                    key_frame_series[i] = self.sanitize_value(value)
            if not value_is_number:
                t = i
                # workaround for values formatted like 0:("I am test") //used for sampler schedules
                key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else self.sanitize_value(value)
            elif is_single_string:# take previous string value and replicate it
                key_frame_series[i] = key_frame_series[i-1]
        key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series # as string

        if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
            interp_method = 'Quadratic'
        if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
            interp_method = 'Linear'

        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
        key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
        
        if integer:
            return key_frame_series.astype(int)
        return key_frame_series
    
    def parse_key_frames(self, string, max_frames):
        # because math functions (i.e. sin(t)) can utilize brackets 
        # it extracts the value in form of some stuff
        # which has previously been enclosed with brackets and
        # with a comma or end of line existing after the closing one
        frames = dict()
        for match_object in string.split(","):
            frameParam = match_object.split(":")
            max_f = max_frames -1
            frame = int(self.sanitize_value(frameParam[0])) if check_is_number(self.sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(frameParam[0].strip().replace("'","",1).replace('"',"",1)[::-1].replace("'","",1).replace('"',"",1)[::-1]))
            frames[frame] = frameParam[1].strip()
        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames