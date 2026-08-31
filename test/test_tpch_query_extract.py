#!/usr/bin/env python3
import importlib.util
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_tpch_sf001", REPO_ROOT / "scripts" / "run_tpch_sf001.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TpchQueryExtractTest(unittest.TestCase):
    def test_extracts_all_queries_and_q15_lifecycle(self):
        source = REPO_ROOT / "bench/MonetDB/sql/benchmarks/tpch/Tests/one.test.in"
        with tempfile.TemporaryDirectory() as tempdir:
            output = pathlib.Path(tempdir)
            MODULE.extract_queries(source, output)

            self.assertEqual(len(list(output.glob("q[0-9][0-9].sql"))), 22)
            self.assertIn("l_returnflag", (output / "q01.sql").read_text())
            self.assertTrue((output / "q15-setup.sql").is_file())
            self.assertTrue((output / "q15-cleanup.sql").is_file())
            self.assertIn("create view revenue0", (output / "q15-setup.sql").read_text())
            self.assertIn("drop view revenue0", (output / "q15-cleanup.sql").read_text())

    def test_top_level_order_by_detection_ignores_subqueries(self):
        self.assertTrue(MODULE.has_top_level_order_by("select x from t order by x"))
        self.assertFalse(
            MODULE.has_top_level_order_by(
                "select * from t where x = (select max(x) from u order by x)"
            )
        )

    def test_checked_in_sf001_counts_are_exact(self):
        data = REPO_ROOT / "bench/MonetDB/sql/benchmarks/tpch/SF-0.01"
        self.assertEqual(MODULE.validate_data_counts(data), MODULE.EXPECTED_COUNTS)


if __name__ == "__main__":
    unittest.main()
