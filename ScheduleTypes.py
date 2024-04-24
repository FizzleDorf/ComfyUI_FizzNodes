import comfy
import numexpr
import torch
import numpy as np
import pandas as pd
import re
import json

from .ScheduleFuncs import *
from .BatchFuncs import *

def prompt_schedule(settings:ScheduleSettings,clip):
    settings.start_frame = 0
    # modulus rollover when current frame exceeds max frames
    settings.current_frame = settings.current_frame % settings.max_frames

    # clear whitespace and newlines from json
    animation_prompts = process_input_text(settings.text_g)

    # add pre_text and app_text then split the combined prompt into positive and negative prompts
    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    # Interpolate the positive prompt weights over frames
    pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_seriesA(pos, settings)
    neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_seriesA(neg, settings)

    # Apply composable diffusion across the batch
    p = PoolAnimConditioning(pos_cur_prompt[settings.current_frame], pos_nxt_prompt[settings.current_frame],
                             weight[settings.current_frame], clip)

    n = PoolAnimConditioning(neg_cur_prompt[settings.current_frame], neg_nxt_prompt[settings.current_frame],
                             weight[settings.current_frame], clip)

    # return the positive and negative conditioning at the current frame
    return (p, n,)

def batch_prompt_schedule(settings:ScheduleSettings,clip):
    # Clear whitespace and newlines from json
    animation_prompts = process_input_text(settings.text_g)

    # Add pre_text and app_text then split the combined prompt into positive and negative prompts
    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    # Interpolate the positive prompt weights over frames
    pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_seriesA(pos, settings)
    neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_seriesA(neg, settings)

    # Apply composable diffusion across the batch
    p = BatchPoolAnimConditioning(pos_cur_prompt, pos_nxt_prompt, weight, clip, )
    n = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip, )

    # return positive and negative conditioning as well as the current and next prompts for each
    return (p, n,)

def batch_prompt_schedule_latentInput(settings:ScheduleSettings,clip, latents):
    # Clear whitespace and newlines from json
    animation_prompts = process_input_text(settings.text_g)

    # Add pre_text and app_text then split the combined prompt into positive and negative prompts
    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    # Interpolate the positive prompt weights over frames
    pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_seriesA(pos, settings)

    # Apply composable diffusion across the batch
    p = BatchPoolAnimConditioning(pos_cur_prompt, pos_nxt_prompt, weight, clip)

    # Interpolate the negative prompt weights over frames
    neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_seriesA(neg, settings)

    # Apply composable diffusion across the batch
    n = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip)

    return (p, n, latents,)

def string_schedule(settings:ScheduleSettings):
    settings.start_frame = 0

    # modulus rollover when current frame exceeds max frames
    settings.current_frame = settings.current_frame % settings.max_frames

    # Clear whitespace and newlines from json
    animation_prompts = process_input_text(settings.text_g)

    # add pre_text and app_text then split the combined prompt into positive and negative prompts
    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    # Interpolate the positive prompt weights over frames
    pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_seriesA(pos, settings)

    # Interpolate the negative prompt weights over frames
    neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_seriesA(neg, settings)

    return(pos_cur_prompt[settings.current_frame], neg_cur_prompt[settings.current_frame], )

def batch_string_schedule(settings:ScheduleSettings):
    settings.start_frame = 0

    # Clear whitespace and newlines from json
    animation_prompts = process_input_text(settings.text_g)

    # add pre_text and app_text then split the combined prompt into positive and negative prompts
    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    # Interpolate the positive prompt weights over frames
    pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_seriesA(pos, settings)

    # Interpolate the negative prompt weights over frames
    neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_seriesA(neg, settings)

    return (pos_cur_prompt, neg_cur_prompt,)

def prompt_schedule_SDXL(settings:ScheduleSettings,clip):
    # modulus rollover when current frame exceeds max frames
    settings.current_frame = settings.current_frame % settings.max_frames

    # Clear whitespace and newlines from json
    animation_prompts_G = process_input_text(settings.text_g)
    animation_prompts_L = process_input_text(settings.text_l)

    # add pre_text and app_text then split the combined prompt into positive and negative prompts
    posG, negG = batch_split_weighted_subprompts(animation_prompts_G, settings.pre_text_G, settings.app_text_G)
    posL, negL = batch_split_weighted_subprompts(animation_prompts_L, settings.pre_text_L, settings.app_text_L)

    pc, pn, pw = BatchInterpolatePromptsSDXL(posG, posL, clip, settings, )
    nc, nn, nw = BatchInterpolatePromptsSDXL(negG, negL, clip, settings, )

    #apply composable diffusion to the current frame
    p = addWeighted(pc[settings.current_frame], pn[settings.current_frame], pw[settings.current_frame])
    n = addWeighted(nc[settings.current_frame], nn[settings.current_frame], nw[settings.current_frame])

    return (p, n,)

def batch_prompt_schedule_SDXL(settings:ScheduleSettings,clip):

    # Clear whitespace and newlines from json
    animation_prompts_G = process_input_text(settings.text_g)
    animation_prompts_L = process_input_text(settings.text_l)

    # add pre_text and app_text then split the combined prompt into positive and negative prompts
    posG, negG = batch_split_weighted_subprompts(animation_prompts_G, settings.pre_text_G, settings.app_text_G)
    posL, negL = batch_split_weighted_subprompts(animation_prompts_L, settings.pre_text_L, settings.app_text_L)

    pc, pn, pw = BatchInterpolatePromptsSDXL(posG, posL, clip, settings,)
    nc, nn, nw = BatchInterpolatePromptsSDXL(negG, negL, clip, settings,)

    p = BatchPoolAnimConditioningSDXL(pc, pn, pw)
    n = BatchPoolAnimConditioningSDXL(nc, nn, nw)

    return (p, n,)

def batch_prompt_schedule_SDXL_latentInput(settings:ScheduleSettings,clip, latents):
    settings.start_frame = 0

    # Clear whitespace and newlines from json
    animation_prompts_G = process_input_text(settings.text_g)
    animation_prompts_L = process_input_text(settings.text_l)

    # add pre_text and app_text then split the combined prompt into positive and negative prompts
    posG, negG = batch_split_weighted_subprompts(animation_prompts_G, settings.pre_text_G, settings.app_text_G)
    posL, negL = batch_split_weighted_subprompts(animation_prompts_L, settings.pre_text_L, settings.app_text_L)

    pc, pn, pw = BatchInterpolatePromptsSDXL(posG, posL, clip, settings)
    nc, nn, nw = BatchInterpolatePromptsSDXL(negG, negL, clip, settings)

    p = BatchPoolAnimConditioningSDXL(pc, pn, pw)
    n = BatchPoolAnimConditioningSDXL(nc, nn, nw)

    return (p, n, latents,)



















