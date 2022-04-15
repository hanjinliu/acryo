from typing_extensions import TypeAlias

RangeLike = tuple[float, float]
Ranges = tuple[RangeLike, RangeLike, RangeLike]

# type alias
nm: TypeAlias = float  # nm is just an example. Can be angstrom or others.
degree: TypeAlias = float
pixel: TypeAlias = int
