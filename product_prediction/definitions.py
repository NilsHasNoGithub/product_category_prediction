from typing import Any, Dict, List, Optional

import attrs

RANDOM_STATE = 42

OptKwargs = Optional[Dict[str, Any]]


@attrs.define
class ProductInfo:
    url: str
    main_image_url: str
    other_image_urls: List[str]
    category: str
    gender: str
    rating: Optional[float]
    rating_count: float
    title: str
    description: Optional[str]
