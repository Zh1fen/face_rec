import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.gallery import Gallery, safe_person_name


def _unit(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1
    return vector


class GalleryTest(unittest.TestCase):
    def test_replace_match_remove_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            gallery = Gallery(directory=directory)
            gallery.replace("甲", [_unit(0), _unit(0)])
            gallery.replace("乙", [_unit(1)])
            gallery.save()
            loaded = Gallery.load(directory)
            self.assertEqual(loaded.match(_unit(0), 0.5).name, "甲")
            self.assertEqual(loaded.match(_unit(1), 0.5).name, "乙")
            mixed = (_unit(0) + _unit(1)) / np.sqrt(2)
            missed = loaded.match(mixed, 0.9)
            self.assertIsNone(missed.name)
            self.assertGreater(missed.score, 0.7)
            self.assertEqual(loaded.remove("甲"), 2)
            loaded.save()
            again = Gallery.load(directory)
            self.assertIsNone(again.match(_unit(0), 0.5).name)
            self.assertEqual(again.counts(), {"乙": 1})

    def test_rejects_path_names(self):
        with self.assertRaises(ValueError):
            safe_person_name("../甲")
        with self.assertRaises(ValueError):
            safe_person_name("")


if __name__ == "__main__":
    unittest.main()
