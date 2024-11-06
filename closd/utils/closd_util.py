import enum

class STATES(enum.IntEnum):
    """
    For the state machine
    """
    NO_STATE = enum.auto()
    REACH = enum.auto()
    STRIKE_KICK = enum.auto()
    STRIKE_PUNCH = enum.auto()
    HALT = enum.auto()
    SIT = enum.auto()
    GET_UP = enum.auto()
    TEXT2MOTION = enum.auto()
