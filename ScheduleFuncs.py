#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum

import numexpr
import torch
import numpy as np
import pandas as pd
import re
import json

#functions used by PromptSchedule nodes

#Addweighted function from Comfyui
def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength):
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return out

def reverseConcatenation(final_conditioning, final_pooled_output, max_frames):
    # Split the final_conditioning and final_pooled_output tensors into their original components
    cond_out = torch.split(final_conditioning, max_frames)
    pooled_out = torch.split(final_pooled_output, max_frames)

    return cond_out, pooled_out

def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

def split_weighted_subprompts(text, frame=0, pre_text='', app_text=''):
    pre_text = str(pre_text)
    app_text = str(app_text)

    if "--neg" in pre_text:
        pre_pos, pre_neg = pre_text.split("--neg")
    else:
        pre_pos, pre_neg = pre_text, ""

    if "--neg" in app_text:
        app_pos, app_neg = app_text.split("--neg")
    else:
        app_pos, app_neg = app_text, ""

    # Check if the text is a string; if not, convert it to a string
    if not isinstance(text, str):
        text = str(text)

    math_parser = re.compile("(?P<weight>(`[\S\s]*?`))", re.VERBOSE)

    parsed_prompt = re.sub(math_parser, lambda m: str(parse_weight(m, frame)), text)

    negative_prompts = ""
    positive_prompts = ""

    # Check if the last character is '0' and remove it
    prompt_split = parsed_prompt.split("--neg")
    if len(prompt_split) > 1:
        positive_prompts, negative_prompts = prompt_split[0], prompt_split[1]
    else:
        positive_prompts = prompt_split[0]

    pos = {}
    neg = {}
    pos[frame] = (str(pre_pos) + " " + str(positive_prompts) + " " + str(app_pos))
    neg[frame] = (str(pre_neg) + " " + str(negative_prompts) + " " + str(app_neg))
    if pos[frame].endswith('0'):
        pos[frame] = pos[frame][:-1]
    if neg[frame].endswith('0'):
        neg[frame] = neg[frame][:-1]

    return pos, neg

def parse_weight(match, frame=0, max_frames=0) -> float: #calculate weight steps for in-betweens
        w_raw = match.group("weight")
        max_f = max_frames  # this line has to be left intact as it's in use by numexpr even though it looks like it doesn't
        if w_raw is None:
            return 1
        if check_is_number(w_raw):
            return float(w_raw)
        else:
            t = frame
            if len(w_raw) < 3:
                print('the value inside `-characters cannot represent a math function')
                return 1
            return float(numexpr.evaluate(w_raw[1:-1]))

def prepare_prompt(prompt_series, max_frames, frame_idx, prompt_weight_1 = 0, prompt_weight_2 = 0, prompt_weight_3 = 0, prompt_weight_4 = 0): #calculate expressions from the text input and return a string
        max_f = max_frames - 1
        pattern = r'`.*?`' #set so the expression will be read between two backticks (``)
        regex = re.compile(pattern)
        prompt_parsed = str(prompt_series)
        for match in regex.finditer(prompt_parsed):
            matched_string = match.group(0)
            parsed_string = matched_string.replace('t', f'{frame_idx}').replace("pw_a", f"prompt_weight_1").replace("pw_b", f"prompt_weight_2").replace("pw_c", f"prompt_weight_3").replace("pw_d", f"prompt_weight_4").replace("max_f", f"{max_f}").replace('`', '') #replace t, max_f and `` respectively
            parsed_value = numexpr.evaluate(parsed_string)
            prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
        return prompt_parsed.strip()

def interpolate_string(animation_prompts, max_frames, current_frame, pre_text, app_text, prompt_weight_1,
                        prompt_weight_2, prompt_weight_3,
                        prompt_weight_4):  # parse the conditioning strength and determine in-betweens.
    # Get prompts sorted by keyframe
    max_f = max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_prompts = {}
    for key, value in animation_prompts.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_prompts[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_prompts[int(numexpr.evaluate(key))] = value

    sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))

    # Setup containers for interpolated prompts
    cur_prompt_series = pd.Series([np.nan for a in range(max_frames)])

    # simple array for strength values
    weight_series = [np.nan] * max_frames

    # in case there is only one keyed promt, set all prompts to that prompt
    if len(sorted_prompts) - 1 == 0:
        for i in range(0, len(cur_prompt_series) - 1):
            current_prompt = sorted_prompts[0][1]
            cur_prompt_series[i] = str(pre_text) + " " + str(current_prompt) + " " + str(app_text)

    # Initialized outside of loop for nan check
    current_key = 0
    next_key = 0

    # For every keyframe prompt except the last
    for i in range(0, len(sorted_prompts) - 1):
        # Get current and next keyframe
        current_key = int(sorted_prompts[i][0])
        next_key = int(sorted_prompts[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_key >= next_key:
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt = sorted_prompts[i][1]

        for f in range(current_key, next_key):
            # add the appropriate prompts and weights to their respective containers.
            cur_prompt_series[f] = ''
            weight_series[f] = 0.0

            cur_prompt_series[f] += (str(pre_text) + " " + str(current_prompt) + " " + str(app_text))

        current_key = next_key
        next_key = max_frames
        # second loop to catch any nan runoff

        for f in range(current_key, next_key):
            # add the appropriate prompts and weights to their respective containers.
            cur_prompt_series[f] = ''
            cur_prompt_series[f] += (str(pre_text) + " " + str(current_prompt) + " " + str(app_text))

    # Evaluate the current and next prompt's expressions
    cur_prompt_series[current_frame] = prepare_prompt(cur_prompt_series[current_frame], max_frames, current_frame,
                                                      prompt_weight_1, prompt_weight_2, prompt_weight_3,
                                                      prompt_weight_4)

    # Show the to/from prompts with evaluated expressions for transparency.
    print("\n", "Max Frames: ", max_frames, "\n", "Current Prompt: ", cur_prompt_series[current_frame], "\n")

    # Output methods depending if the prompts are the same or if the current frame is a keyframe.
    # if it is an in-between frame and the prompts differ, composable diffusion will be performed.
    return (cur_prompt_series[current_frame])
def PoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip):  
    if str(cur_prompt) == str(nxt_prompt):
        tokens = clip.tokenize(str(cur_prompt))
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    if weight == 1:
        tokens = clip.tokenize(str(cur_prompt))
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    if weight == 0:
        tokens = clip.tokenize(str(nxt_prompt))
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    else:
        tokens = clip.tokenize(str(nxt_prompt))
        cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        tokens = clip.tokenize(str(cur_prompt))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)
        return addWeighted([[cond_to, {"pooled_output": pooled_to}]], [[cond_from, {"pooled_output": pooled_from}]], weight)

def SDXLencode(clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
    tokens = clip.tokenize(text_g)
    tokens["l"] = clip.tokenize(text_l)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]

def interpolate_prompts_SDXL(animation_promptsG, animation_promptsL, max_frames, current_frame, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, print_output): #parse the conditioning strength and determine in-betweens.
        #Get prompts sorted by keyframe
        max_f = max_frames #needed for numexpr even though it doesn't look like it's in use.
        parsed_animation_promptsG = {}
        parsed_animation_promptsL = {}
        for key, value in animation_promptsG.items():
            if check_is_number(key):  #default case 0:(1 + t %5), 30:(5-t%2)
                parsed_animation_promptsG[key] = value
            else:  #math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsed_animation_promptsG[int(numexpr.evaluate(key))] = value
        
        sorted_prompts_G = sorted(parsed_animation_promptsG.items(), key=lambda item: int(item[0]))

        for key, value in animation_promptsL.items():
            if check_is_number(key):  #default case 0:(1 + t %5), 30:(5-t%2)
                parsed_animation_promptsL[key] = value
            else:  #math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsed_animation_promptsL[int(numexpr.evaluate(key))] = value
        
        sorted_prompts_L = sorted(parsed_animation_promptsL.items(), key=lambda item: int(item[0]))
        
        #Setup containers for interpolated prompts
        cur_prompt_series_G = pd.Series([np.nan for a in range(max_frames)])
        nxt_prompt_series_G = pd.Series([np.nan for a in range(max_frames)])

        cur_prompt_series_L = pd.Series([np.nan for a in range(max_frames)])
        nxt_prompt_series_L = pd.Series([np.nan for a in range(max_frames)])

        #simple array for strength values
        weight_series = [np.nan] * max_frames

        #in case there is only one keyed promt, set all prompts to that prompt
        if len(sorted_prompts_G) - 1 == 0:
            for i in range(0, len(cur_prompt_series_G)-1):           
                current_prompt_G = sorted_prompts_G[0][1] 
                cur_prompt_series_G[i] = str(pre_text_G) + " " + str(current_prompt_G) + " " + str(app_text_G)
                nxt_prompt_series_G[i] = str(pre_text_G) + " " + str(current_prompt_G) + " " + str(app_text_G)

        if len(sorted_prompts_L) - 1 == 0:
            for i in range(0, len(cur_prompt_series_L)-1):           
                current_prompt_L = sorted_prompts_L[0][1]           
                cur_prompt_series_L[i] = str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L)
                nxt_prompt_series_L[i] = str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L)

        
                
        #Initialized outside of loop for nan check
        current_key = 0
        next_key = 0

        # For every keyframe prompt except the last
        for i in range(0, len(sorted_prompts_G) - 1):
            # Get current and next keyframe
            current_key = int(sorted_prompts_G[i][0])
            next_key = int(sorted_prompts_G[i + 1][0])

            # Ensure there's no weird ordering issues or duplication in the animation prompts
            # (unlikely because we sort above, and the json parser will strip dupes)
            if current_key >= next_key:
                print(f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
                continue

            # Get current and next keyframes' positive and negative prompts (if any)
            current_prompt_G = sorted_prompts_G[i][1]
            next_prompt_G = sorted_prompts_G[i + 1][1]
            
            # Calculate how much to shift the weight from current to next prompt at each frame.
            weight_step = 1 / (next_key - current_key)

            for f in range(current_key, next_key):
                next_weight = weight_step * (f - current_key)
                current_weight = 1 - next_weight
                
                #add the appropriate prompts and weights to their respective containers.
                cur_prompt_series_G[f] = ''
                nxt_prompt_series_G[f] = ''
                weight_series[f] = 0.0

                cur_prompt_series_G[f] += (str(pre_text_G) + " " + str(current_prompt_G) + " " + str(app_text_G))
                nxt_prompt_series_G[f] += (str(pre_text_G) + " " + str(next_prompt_G) + " " + str(app_text_G))

                weight_series[f] += current_weight
        
            current_key = next_key
            next_key = max_frames
            current_weight = 0.0
            #second loop to catch any nan runoff
            for f in range(current_key, next_key):
                 next_weight = weight_step * (f - current_key)
                 
                 #add the appropriate prompts and weights to their respective containers.
                 cur_prompt_series_G[f] = ''
                 nxt_prompt_series_G[f] = ''
                 weight_series[f] = current_weight

                 cur_prompt_series_G[f] += (str(pre_text_G) + " " + str(current_prompt_G) + " " + str(app_text_G))
                 nxt_prompt_series_G[f] += (str(pre_text_G) + " " + str(next_prompt_G) + " " + str(app_text_G))


        #Reset outside of loop for nan check
        current_key = 0
        next_key = 0

        # For every keyframe prompt except the last
        for i in range(0, len(sorted_prompts_L) - 1):
            # Get current and next keyframe
            current_key = int(sorted_prompts_L[i][0])
            next_key = int(sorted_prompts_L[i + 1][0])

            # Ensure there's no weird ordering issues or duplication in the animation prompts
            # (unlikely because we sort above, and the json parser will strip dupes)
            if current_key >= next_key:
                print(f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
                continue

            # Get current and next keyframes' positive and negative prompts (if any)
            current_prompt_L = sorted_prompts_L[i][1]
            next_prompt_L = sorted_prompts_L[i + 1][1]
            
            # Calculate how much to shift the weight from current to next prompt at each frame.
            weight_step = 1 / (next_key - current_key)

            for f in range(current_key, next_key):
                next_weight = weight_step * (f - current_key)
                current_weight = 1 - next_weight
                
                #add the appropriate prompts and weights to their respective containers.
                cur_prompt_series_L[f] = ''
                nxt_prompt_series_L[f] = ''
                weight_series[f] = 0.0

                cur_prompt_series_L[f] += (str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L))
                nxt_prompt_series_L[f] += (str(pre_text_L) + " " + str(next_prompt_L) + " " + str(app_text_L))

                weight_series[f] += current_weight
        
            current_key = next_key
            next_key = max_frames
            current_weight = 0.0
            #second loop to catch any nan runoff
            for f in range(current_key, next_key):
                 next_weight = weight_step * (f - current_key)
                 
                 #add the appropriate prompts and weights to their respective containers.
                 cur_prompt_series_L[f] = ''
                 nxt_prompt_series_L[f] = ''
                 weight_series[f] = current_weight

                 cur_prompt_series_L[f] += (str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L))
                 nxt_prompt_series_L[f] += (str(pre_text_L) + " " + str(next_prompt_L) + " " + str(app_text_L))

        #Evaluate the current and next prompt's expressions
        cur_prompt_series_G[current_frame] = prepare_prompt(cur_prompt_series_G[current_frame], max_frames, current_frame, pw_a, pw_b, pw_c, pw_d)
        nxt_prompt_series_G[current_frame] = prepare_prompt(nxt_prompt_series_G[current_frame], max_frames, current_frame, pw_a, pw_b, pw_c, pw_d) 
        cur_prompt_series_L[current_frame] = prepare_prompt(cur_prompt_series_L[current_frame], max_frames, current_frame, pw_a, pw_b, pw_c, pw_d)
        nxt_prompt_series_L[current_frame] = prepare_prompt(nxt_prompt_series_L[current_frame], max_frames, current_frame, pw_a, pw_b, pw_c, pw_d)       
        if print_output == True:
            #Show the to/from prompts with evaluated expressions for transparency.
            print("\n", "G_Clip:", "\n", "Max Frames: ", max_frames, "\n", "Current Prompt: ", cur_prompt_series_G[current_frame], "\n", "Next Prompt: ", nxt_prompt_series_G[current_frame], "\n", "Strength : ", weight_series[current_frame], "\n")

            print("\n", "L_Clip:", "\n", "Max Frames: ", max_frames, "\n", "Current Prompt: ", cur_prompt_series_L[current_frame], "\n", "Next Prompt: ", nxt_prompt_series_L[current_frame], "\n", "Strength : ", weight_series[current_frame], "\n")

        #Output methods depending if the prompts are the same or if the current frame is a keyframe.
        #if it is an in-between frame and the prompts differ, composable diffusion will be performed.
        current_cond = SDXLencode(clip, width, height, crop_w, crop_h, target_width, target_height, cur_prompt_series_G[current_frame], cur_prompt_series_L[current_frame])

        if str(cur_prompt_series_G[current_frame]) == str(nxt_prompt_series_G[current_frame]) and str(cur_prompt_series_L[current_frame]) == str(nxt_prompt_series_L[current_frame]):           
            return current_cond
        
        if weight_series[current_frame] == 1:
            return current_cond
        
        if weight_series[current_frame] == 0:
            next_cond = SDXLencode(clip, width, height, crop_w, crop_h, target_width, target_height, cur_prompt_series_G[current_frame], cur_prompt_series_L[current_frame])
            return next_cond
        
        else:
            next_cond = SDXLencode(clip, width, height, crop_w, crop_h, target_width, target_height, cur_prompt_series_G[current_frame], cur_prompt_series_L[current_frame])
            return addWeighted(current_cond, next_cond, weight_series[current_frame])