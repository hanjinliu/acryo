from typing import Union, TYPE_CHECKING
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation

RangeLike = tuple[float, float]
Ranges = tuple[RangeLike, RangeLike, RangeLike]
RotationType = Union[Ranges, "Rotation"]

# type alias
nm: TypeAlias = float
degree: TypeAlias = float
pixel: TypeAlias = int
subpixel: TypeAlias = float
