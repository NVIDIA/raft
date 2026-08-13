import re
import unittest
from pathlib import Path


# PEP 621 stores the license in an inline table; the recipe must use .text.
BARE_LICENSE_RE = re.compile(
    r'load_from_file\("python/[^"]+/pyproject.toml"\)\.project\.license(?!\.)'
)


class TestRaftDaskRecipe(unittest.TestCase):
    def test_license_reference_uses_text_field(self):
        path = Path(__file__).resolve().parents[1] / "conda/recipes/raft-dask/recipe.yaml"
        text = path.read_text(encoding="utf-8")
        matches = BARE_LICENSE_RE.findall(text)
        self.assertEqual(
            matches,
            [],
            "recipe uses bare '.project.license'; use '.project.license.text'",
        )
