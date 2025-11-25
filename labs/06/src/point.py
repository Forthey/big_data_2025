import dataclasses

@dataclasses.dataclass
class Point[Type]:
    x: Type
    y: Type
