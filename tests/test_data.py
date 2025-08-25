from xtc.data import get_datasets

def test_load():
    ds, labels = get_datasets()
    assert "train" in ds and "test" in ds
    assert len(labels) == 4
