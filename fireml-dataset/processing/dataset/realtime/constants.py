from enum import Enum


class DiscretizationMethods(Enum):
    EXACT = 1
    SQUARE = 2


class LandCoverLayerType(Enum):
    EVT = 1
    FILTER = 2
    OTHER = 3


class MeteorologyLayerType(Enum):
    OTHER = 1


class FilterReason(Enum):
    DETECTIONS = 1
    LAND_COVER = 2
