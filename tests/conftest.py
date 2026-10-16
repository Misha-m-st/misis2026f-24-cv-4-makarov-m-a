import numpy as np
import pytest
from PIL import Image

from document_vos.data import Frame, Sequence, save_manifest


@pytest.fixture
def sequences(tmp_path):
    result = []
    for v in range(6):
        frames = []
        for i in range(4):
            path = tmp_path / f"image-{v}-{i}.png"
            image = np.random.default_rng(v * 100 + i).integers(0, 255, (32, 48, 3), dtype=np.uint8)
            Image.fromarray(image).save(path)
            frames.append(Frame(str(i), str(path), i * 0.1, [[8, 6], [36, 6], [36, 25], [8, 25]]))
        result.append(
            Sequence(
                "midv2020",
                f"v{v}",
                f"d{v}",
                f"t{v}",
                f"g{v}",
                frames,
                time_source="synthetic timestamps",
            )
        )
    return result


@pytest.fixture
def manifest(tmp_path, sequences):
    path = tmp_path / "manifest.jsonl"
    save_manifest(path, sequences)
    return path
