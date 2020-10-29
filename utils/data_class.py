class_names = [
    # prohibitory
    'speed_30',
    'speed_120',
    'no_overtaking',
    'prohibition',

    # danger
    'construction',
    'signal_ahead',
    'pedestrian',
    'deer',

    # mandatory
    'right',
    'straight',
    'left_straight',
    'roundabout',

    # others
    'priority_road',
    'give_way',
    'stop',
    'cancel_no_overtaking',
]

class_indices = [
    # prohibitory
    1,  # speed_30
    8,  # speed_120
    9,  # no_overtaking
    15,  # prohibition

    # danger
    25,  # construction
    26,  # signal_ahead
    27,  # pedestrian
    31,  # deer

    # mandatory
    33,  # right
    35,  # straight
    37,  # left_straight
    40,  # roundabout

    # others
    12,  # priority_road
    13,  # give_way
    14,  # stop
    41,  # cancel_no_overtaking
]
