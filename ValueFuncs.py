import numexpr
import torch
import numpy as np
import pandas as pd
import re
import json

from .ScheduleFuncs import check_is_number


def sanitize_value(value):
    # Remove single quotes, double quotes, and parentheses
    value = value.replace("'", "").replace('"', "").replace('(', "").replace(')', "")
    return value


def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear', is_single_string=False):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
    value_is_number = False
    for i in range(0, max_frames):
        if i in key_frames:
            value = key_frames[i]
            value_is_number = check_is_number(sanitize_value(value))
            if value_is_number:  # if it's only a number, leave the rest for the default interpolation
                key_frame_series[i] = sanitize_value(value)
        if not value_is_number:
            t = i
            # workaround for values formatted like 0:("I am test") //used for sampler schedules
            key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else sanitize_value(value)
        elif is_single_string:  # take previous string value and replicate it
            key_frame_series[i] = key_frame_series[i - 1]
    key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series  # as string

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')

    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def parse_key_frames(string, max_frames):
    # because math functions (i.e. sin(t)) can utilize brackets
    # it extracts the value in form of some stuff
    # which has previously been enclosed with brackets and
    # with a comma or end of line existing after the closing one
    frames = dict()
    for match_object in string.split(","):
        frameParam = match_object.split(":")
        max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
        frame = int(sanitize_value(frameParam[0])) if check_is_number(
            sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(
            frameParam[0].strip().replace("'", "", 1).replace('"', "", 1)[::-1].replace("'", "", 1).replace('"', "", 1)[::-1]))
        frames[frame] = frameParam[1].strip()
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames

def batch_get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear', is_single_string=False):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
    value_is_number = False
    for i in range(0, max_frames):
        if i in key_frames:
            value = str(key_frames[i])  # Convert to string to ensure it's treated as an expression
            value_is_number = check_is_number(sanitize_value(value))
            if value_is_number:
                key_frame_series[i] = sanitize_value(value)
        if not value_is_number:
            t = i
            # workaround for values formatted like 0:("I am test") //used for sampler schedules
            key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else sanitize_value(value)
        elif is_single_string:  # take previous string value and replicate it
            key_frame_series[i] = key_frame_series[i - 1]
    key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series  # as string

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')

    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def batch_parse_key_frames(string, max_frames):
    # because math functions (i.e. sin(t)) can utilize brackets
    # it extracts the value in form of some stuff
    # which has previously been enclosed with brackets and
    # with a comma or end of line existing after the closing one
    string = re.sub(r',\s*$', '', string)
    frames = dict()
    for match_object in string.split(","):
        frameParam = match_object.split(":")
        max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
        frame = int(sanitize_value(frameParam[0])) if check_is_number(
            sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(
            frameParam[0].strip().replace("'", "", 1).replace('"', "", 1)[::-1].replace("'", "", 1).replace('"', "",1)[::-1]))
        frames[frame] = frameParam[1].strip()
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames