from collections.abc import Iterator

import pytest
from stub_router import Stub, serve


@pytest.fixture
def router() -> Iterator[Stub]:
    with serve() as stub:
        yield stub
