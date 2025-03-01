from space_type import SpaceType


class BoardSpace:
    def __init__(self, name: str, position: int, space_type: SpaceType):
        self.name = name
        self.position = position
        self.space_type = space_type

