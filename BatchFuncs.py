# These nodes were made using code from the Deforum extension for A1111 webui
# You can find the project here: https://github.com/deforum-art/sd-webui-deforum

import numexpr
import torch
import numpy as np
import pandas as pd
import re

from .ScheduleFuncs import addWeighted, check_is_number, parse_weight, prepare_prompt, SDXLencode, reverseConcatenation

def prepare_batch_prompt(prompt_series, max_frames, frame_idx, prompt_weight_1=0, prompt_weight_2=0, prompt_weight_3=0,
                         prompt_weight_4=0):  # calculate expressions from the text input and return a string
    max_f = max_frames - 1
    pattern = r'`.*?`'  # set so the expression will be read between two backticks (``)
    regex = re.compile(pattern)
    prompt_parsed = str(prompt_series)

    for match in regex.finditer(prompt_parsed):
        matched_string = match.group(0)
        parsed_string = matched_string.replace('t', f'{frame_idx}').replace("pw_a", f"{prompt_weight_1}").replace("pw_b",
                                                                                                                 f"{prompt_weight_2}").replace(
            "pw_c", f"{prompt_weight_3}").replace("pw_d", f"{prompt_weight_4}").replace("max_f", f"{max_f}").replace('`',
                                                                                                                     '')  # replace t, max_f and `` respectively
        parsed_value = numexpr.evaluate(parsed_string)
        prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
    return prompt_parsed.strip()

def batch_split_weighted_subprompts(text, pre_text, app_text):
    pos = {}
    neg = {}
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

    for frame, prompt in text.items():
        negative_prompts = ""
        positive_prompts = ""
        prompt_split = prompt.split("--neg")

        if len(prompt_split) > 1:
            positive_prompts, negative_prompts = prompt_split[0], prompt_split[1]
        else:
            positive_prompts = prompt_split[0]

        pos[frame] = ""
        neg[frame] = ""
        pos[frame] += (str(pre_pos) + " " + positive_prompts + " " + str(app_pos))
        neg[frame] += (str(pre_neg) + " " + negative_prompts + " " + str(app_neg))
        if pos[frame].endswith('0'):
            pos[frame] = pos[frame][:-1]
        if neg[frame].endswith('0'):
            neg[frame] = neg[frame][:-1]
    return pos, neg

def interpolate_prompt_series(animation_prompts, max_frames, start_frame, pre_text, app_text, prompt_weight_1=[],
                              prompt_weight_2=[], prompt_weight_3=[], prompt_weight_4=[], Is_print = False):

    max_f = max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_prompts = {}
    for key, value in animation_prompts.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_prompts[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_prompts[int(numexpr.evaluate(key))] = value

    sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))

    # Automatically set the first keyframe to 0 if it's missing
    if sorted_prompts[0][0] != "0":
        sorted_prompts.insert(0, ("0", sorted_prompts[0][1]))

    # Automatically set the last keyframe to the maximum number of frames
    if sorted_prompts[-1][0] != str(max_frames):
        sorted_prompts.append((str(max_frames), sorted_prompts[-1][1]))
    # Setup containers for interpolated prompts
    cur_prompt_series = pd.Series([np.nan for a in range(max_frames)])
    nxt_prompt_series = pd.Series([np.nan for a in range(max_frames)])

    # simple array for strength values
    weight_series = [np.nan] * max_frames

    # in case there is only one keyed promt, set all prompts to that prompt
    if len(sorted_prompts) == 1:
        for i in range(0, len(cur_prompt_series) - 1):
            current_prompt = sorted_prompts[0][1]
            cur_prompt_series[i] = str(current_prompt)
            nxt_prompt_series[i] = str(current_prompt)

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
        next_prompt = sorted_prompts[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(max(current_key, 0), min(next_key, len(cur_prompt_series))):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            cur_prompt_series[f] = ''
            nxt_prompt_series[f] = ''
            weight_series[f] = 0.0
            cur_prompt_series[f] = str(current_prompt)
            nxt_prompt_series[f] = str(next_prompt)

            weight_series[f] += current_weight

        current_key = next_key
        next_key = max_frames
        current_weight = 0.0


    if type(prompt_weight_1) in {int, float}:
        prompt_weight_1 = tuple([prompt_weight_1] * max_frames)

    if type(prompt_weight_2) in {int, float}:
        prompt_weight_2 = tuple([prompt_weight_2] * max_frames)

    if type(prompt_weight_3) in {int, float}:
        prompt_weight_3 = tuple([prompt_weight_3] * max_frames)

    if type(prompt_weight_4) in {int, float}:
        prompt_weight_4 = tuple([prompt_weight_4] * max_frames)

    index_offset = 0
    # Evaluate the current and next prompt's expressions

    for i in range(start_frame,len(cur_prompt_series)):
        cur_prompt_series[i] = prepare_batch_prompt(cur_prompt_series[i], max_frames, i, prompt_weight_1[i],
                                                    prompt_weight_2[i], prompt_weight_3[i], prompt_weight_4[i])
        nxt_prompt_series[i] = prepare_batch_prompt(nxt_prompt_series[i], max_frames, i, prompt_weight_1[i],
                                                    prompt_weight_2[i], prompt_weight_3[i], prompt_weight_4[i])
        if Is_print == True:
            # Show the to/from prompts with evaluated expressions for transparency.
            print("\n", "Max Frames: ", max_frames, "\n", "frame index: ", (start_frame+i), "\n", "Current Prompt: ",
                  cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                  weight_series[i], "\n")
        index_offset = index_offset+1



    # Output methods depending if the prompts are the same or if the current frame is a keyframe.
    # if it is an in-between frame and the prompts differ, composable diffusion will be performed.
    return (cur_prompt_series, nxt_prompt_series, weight_series)

def BatchPoolAnimConditioning(cur_prompt_series, nxt_prompt_series, weight_series, clip):
    pooled_out = []
    cond_out = []


    def pad_with_clip_tokens(tensor, target_length):
        pad_token = clip.cond_stage_model.clip_l.special_tokens['pad']
        tokens_to_pad = clip.tokenize(pad_token)
        tokens_to_pad = tokens_to_pad.unsqueeze(0).expand(1, target_length, tokens_to_pad.shape[2])
        tokens_to_pad = tokens_to_pad.to(tensor.device)
        return torch.cat([tensor, tokens_to_pad], dim=1)

    for i in range(len(cur_prompt_series)):
        tokens = clip.tokenize(str(cur_prompt_series[i]))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)

        if i < len(nxt_prompt_series):
            tokens = clip.tokenize(str(nxt_prompt_series[i]))
            cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        else:
            cond_from, pooled_from = torch.zeros_like(cond_to), torch.zeros_like(pooled_to)
            pooled_from = pad_with_clip_tokens(pooled_from, cond_to.shape[1])

        interpolated_conditioning = addWeighted([[cond_to, {"pooled_output": pooled_to}]],
                                                [[cond_from, {"pooled_output": pooled_from}]],
                                                weight_series[i])

        interpolated_cond = interpolated_conditioning[0][0]
        interpolated_pooled = interpolated_conditioning[0][1].get("pooled_output", pooled_from)

        pooled_out.append(interpolated_pooled)
        cond_out.append(interpolated_cond)

    final_pooled_output = torch.cat(pooled_out, dim=0)
    final_conditioning = torch.cat(cond_out, dim=0)

    return [[final_conditioning, {"pooled_output": final_pooled_output}]]



def BatchGLIGENConditioning(cur_prompt_series, nxt_prompt_series, weight_series, clip):
    pooled_out = []
    cond_out = []

    for i in range(len(cur_prompt_series)):
        tokens = clip.tokenize(str(cur_prompt_series[i]))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)

        tokens = clip.tokenize(str(nxt_prompt_series[i]))
        cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)

        interpolated_conditioning = addWeighted([[cond_to, {"pooled_output": pooled_to}]],
                                                [[cond_from, {"pooled_output": pooled_from}]],
                                                weight_series[i])

        interpolated_cond = interpolated_conditioning[0][0]
        interpolated_pooled = interpolated_conditioning[0][1].get("pooled_output", pooled_from)

        pooled_out.append(interpolated_pooled)
        cond_out.append(interpolated_cond)

    final_pooled_output = torch.cat(pooled_out, dim=0)
    final_conditioning = torch.cat(cond_out, dim=0)

    return cond_out, pooled_out

def BatchPoolAnimConditioningSDXL(cur_prompt_series, nxt_prompt_series, weight_series, clip):
    pooled_out = []
    cond_out = []

    for i in range(len(cur_prompt_series)):
        interpolated_conditioning = addWeighted(cur_prompt_series[i],
                                                nxt_prompt_series[i],
                                                weight_series[i])

        interpolated_cond = interpolated_conditioning[0][0]
        interpolated_pooled = interpolated_conditioning[0][1].get("pooled_output")

        pooled_out.append(interpolated_pooled)
        cond_out.append(interpolated_cond)

    final_pooled_output = torch.cat(pooled_out, dim=0)
    final_conditioning = torch.cat(cond_out, dim=0)

    return [[final_conditioning, {"pooled_output": final_pooled_output}]]


def BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, max_frames, clip, app_text_G,
                             app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w,
                             crop_h, target_width, target_height, Is_print = False):

    # parse the conditioning strength and determine in-betweens.
    # Get prompts sorted by keyframe
    max_f = max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_promptsG = {}
    parsed_animation_promptsL = {}
    for key, value in animation_promptsG.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_promptsG[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_promptsG[int(numexpr.evaluate(key))] = value

    sorted_prompts_G = sorted(parsed_animation_promptsG.items(), key=lambda item: int(item[0]))

    for key, value in animation_promptsL.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_promptsL[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_promptsL[int(numexpr.evaluate(key))] = value

    sorted_prompts_L = sorted(parsed_animation_promptsL.items(), key=lambda item: int(item[0]))

    # Setup containers for interpolated prompts
    cur_prompt_series_G = pd.Series([np.nan for a in range(max_frames)])
    nxt_prompt_series_G = pd.Series([np.nan for a in range(max_frames)])

    cur_prompt_series_L = pd.Series([np.nan for a in range(max_frames)])
    nxt_prompt_series_L = pd.Series([np.nan for a in range(max_frames)])

    # simple array for strength values
    weight_series = [np.nan] * max_frames

    # in case there is only one keyed promt, set all prompts to that prompt
    if len(sorted_prompts_G) - 1 == 0:
        for i in range(0, len(cur_prompt_series_G) - 1):
            current_prompt_G = sorted_prompts_G[0][1]
            cur_prompt_series_G[i] = str(pre_text_G) + " " + str(current_prompt_G) + " " + str(app_text_G)
            nxt_prompt_series_G[i] = str(pre_text_G) + " " + str(current_prompt_G) + " " + str(app_text_G)

    if len(sorted_prompts_L) - 1 == 0:
        for i in range(0, len(cur_prompt_series_L) - 1):
            current_prompt_L = sorted_prompts_L[0][1]
            cur_prompt_series_L[i] = str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L)
            nxt_prompt_series_L[i] = str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L)

    # Initialized outside of loop for nan check
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
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt_G = sorted_prompts_G[i][1]
        next_prompt_G = sorted_prompts_G[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(current_key, next_key):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            if f < max_frames:
                cur_prompt_series_G[f] = ''
                nxt_prompt_series_G[f] = ''
                weight_series[f] = 0.0

                cur_prompt_series_G[f] += (str(pre_text_G) + " " + str(current_prompt_G) + " " + str(app_text_G))
                nxt_prompt_series_G[f] += (str(pre_text_G) + " " + str(next_prompt_G) + " " + str(app_text_G))

                weight_series[f] += current_weight

        current_key = next_key
        next_key = max_frames
        current_weight = 0.0
        # second loop to catch any nan runoff
        for f in range(current_key, next_key):
            next_weight = weight_step * (f - current_key)

            # add the appropriate prompts and weights to their respective containers.
            cur_prompt_series_G[f] = ''
            nxt_prompt_series_G[f] = ''
            weight_series[f] = current_weight

            cur_prompt_series_G[f] += (str(pre_text_G) + " " + str(current_prompt_G) + " " + str(app_text_G))
            nxt_prompt_series_G[f] += (str(pre_text_G) + " " + str(next_prompt_G) + " " + str(app_text_G))

    # Reset outside of loop for nan check
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
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt_L = sorted_prompts_L[i][1]
        next_prompt_L = sorted_prompts_L[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(current_key, next_key):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            if f < max_frames:
                cur_prompt_series_L[f] = ''
                nxt_prompt_series_L[f] = ''
                weight_series[f] = 0.0

                cur_prompt_series_L[f] += (str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L))
                nxt_prompt_series_L[f] += (str(pre_text_L) + " " + str(next_prompt_L) + " " + str(app_text_L))

                weight_series[f] += current_weight

        current_key = next_key
        next_key = max_frames
        current_weight = 0.0
        # second loop to catch any nan runoff
        for f in range(current_key, next_key):
            next_weight = weight_step * (f - current_key)

            # add the appropriate prompts and weights to their respective containers.
            cur_prompt_series_L[f] = ''
            nxt_prompt_series_L[f] = ''
            weight_series[f] = current_weight

            cur_prompt_series_L[f] += (str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L))
            nxt_prompt_series_L[f] += (str(pre_text_L) + " " + str(next_prompt_L) + " " + str(app_text_L))

    # Evaluate the current and next prompt's expressions
    for i in range(0, max_frames):
        cur_prompt_series_G[i] = prepare_batch_prompt(cur_prompt_series_G[i], max_frames, i,
                                                             pw_a, pw_b, pw_c, pw_d)
        nxt_prompt_series_G[i] = prepare_batch_prompt(nxt_prompt_series_G[i], max_frames, i,
                                                             pw_a, pw_b, pw_c, pw_d)
        cur_prompt_series_L[i] = prepare_batch_prompt(cur_prompt_series_L[i], max_frames, i,
                                                             pw_a, pw_b, pw_c, pw_d)
        nxt_prompt_series_L[i] = prepare_batch_prompt(nxt_prompt_series_L[i], max_frames, i,
                                                             pw_a, pw_b, pw_c, pw_d)

    current_conds = []
    next_conds = []
    for i in range(0, max_frames):
        current_conds.append(SDXLencode(clip, width, height, crop_w, crop_h, target_width, target_height,
                                cur_prompt_series_G[i], cur_prompt_series_L[i]))
        next_conds.append(SDXLencode(clip, width, height, crop_w, crop_h, target_width, target_height,
                                nxt_prompt_series_G[i], nxt_prompt_series_L[i]))

    if Is_print == True:
        # Show the to/from prompts with evaluated expressions for transparency.
        for i in range(0, max_frames):
            print("\n", "Max Frames: ", max_frames, "\n", "Current Prompt G: ", cur_prompt_series_G[i],
                  "\n", "Current Prompt L: ", cur_prompt_series_L[i], "\n", "Next Prompt G: ", nxt_prompt_series_G[i],
                  "\n", "Next Prompt L : ", nxt_prompt_series_L[i],  "\n"), "\n", "Current weight: ", weight_series[i]

    return BatchPoolAnimConditioningSDXL(current_conds, next_conds, weight_series, clip)
