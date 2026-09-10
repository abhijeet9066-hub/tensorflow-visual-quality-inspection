from pathlib import Path

def test_demo_dataset_has_all_classes():
    root=Path(__file__).resolve().parents[1]/"data"/"images"
    assert {p.name for p in root.iterdir() if p.is_dir()} == {"normal","crack","scratch","dent"}
