import ezc3d
import pdb
import math
import numpy as np


def get_point_frequency_ezc3d(acq):
    fq = acq['parameters']['POINT']['RATE']['value'][0]
    return fq


def get_first_frame_ezc3d(acq):
    first_frame = acq['header']['points']['first_frame']
    return first_frame


def get_event_ezc3d(acq):
    event_time = acq['parameters']['EVENT']['TIMES']['value'][1]
    event_name = acq['parameters']['EVENT']['LABELS']['value']
    context_exist = 'CONTEXTS' in acq['parameters']['EVENT'].keys()

    all_event = list()
    if context_exist:
        event_context = acq['parameters']['EVENT']['CONTEXTS']['value']
        for label, context in zip(event_name, event_context):
            all_event.append(context+' '+label)
    else:
        for label in event_name:
            all_event.append(label)

    all_event = set(all_event)

    event = dict()

    fq = get_point_frequency_ezc3d(acq)
    first_frame = get_first_frame_ezc3d(acq)
    for name_event in all_event:
        event[name_event] = []

    if context_exist:
        for time, label, context in zip(event_time, event_name, event_context):
            event[context+' '+label].append(math.ceil(time*fq-first_frame-1))
    else:
        for time, label in zip(event_time, event_name):
            event[label].append(math.ceil(time*fq-first_frame-1))

    # Change the value in to numpy array to work more easily on it
    for event_name in all_event:
        event[event_name] = np.array(event[event_name])

    return event


def get_points_ezc3d(acq):
    """Points extraction with a dictionnary allowing to find 
    the point position in the numpy array using text without
    using a dictionnary"""

    points_name = acq['parameters']['POINT']['LABELS']['value']

    points_temp = acq['data']['points'][0:3, :, :]
    points_ind = dict()
    for index_point, name_point in enumerate(points_name):
        points_ind[name_point] = index_point

    return points_temp, points_name, points_ind


if __name__ == "__main__":
    filename = 'Sit_to_stand_croped.c3d'
    acq = ezc3d.c3d(filename)
    event = get_event_ezc3d(acq)
    pdb.set_trace()
