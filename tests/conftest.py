import pytest
def pytest_addoption(parser):
    parser.addoption("--backend", default="numpy")

@pytest.fixture(autouse=True)
def setup_backend(request):
    from acryo.backend import set_backend
    backend = request.config.getoption('--backend')
    set_backend(backend)
    
