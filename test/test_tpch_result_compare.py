#!/usr/bin/env python3
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPARATOR = REPO_ROOT / "scripts" / "compare_tpch_results.py"


class TpchResultCompareTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tempdir.name)
        self.baseline = root / "baseline"
        self.candidate = root / "candidate"
        self.baseline.mkdir()
        self.candidate.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def write_result(self, directory, rows, *, ordered, types=None):
        types = types or ["varchar", "decimal(15,2)", "date"]
        metadata = {
            "schema_version": 1,
            "query_id": 1,
            "ordered": ordered,
            "row_count": len(rows),
            "columns": [
                {"name": f"c{index}", "type": type_name}
                for index, type_name in enumerate(types)
            ],
        }
        (directory / "q01.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        with (directory / "q01.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerows(rows)

    def compare(self):
        self.assertTrue(COMPARATOR.is_file(), f"missing comparator: {COMPARATOR}")
        return subprocess.run(
            [sys.executable, str(COMPARATOR), str(self.baseline), str(self.candidate)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_exact_typed_values_preserve_null_date_decimal_and_separators(self):
        rows = [["alpha\tbeta|gamma", "123.40", "1998-12-01"], ["\\N", "-0.00", "\\N"]]
        self.write_result(self.baseline, rows, ordered=True)
        self.write_result(self.candidate, rows, ordered=True)

        result = self.compare()

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Q01 PASS", result.stdout)

    def test_unordered_query_sorts_complete_rows(self):
        rows = [["a", "1.00", "1998-01-01"], ["b", "2.00", "1998-01-02"]]
        self.write_result(self.baseline, rows, ordered=False)
        self.write_result(self.candidate, list(reversed(rows)), ordered=False)

        result = self.compare()

        self.assertEqual(result.returncode, 0, result.stdout)

    def test_ordered_query_rejects_reordered_rows(self):
        rows = [["a", "1.00", "1998-01-01"], ["b", "2.00", "1998-01-02"]]
        self.write_result(self.baseline, rows, ordered=True)
        self.write_result(self.candidate, list(reversed(rows)), ordered=True)

        result = self.compare()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("Q01 FAIL", result.stdout)

    def test_one_ulp_float_difference_is_not_tolerated(self):
        types = ["double"]
        self.write_result(self.baseline, [["1.0000000000000000"]], ordered=True, types=types)
        self.write_result(self.candidate, [["1.0000000000000002"]], ordered=True, types=types)

        result = self.compare()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("value mismatch", result.stdout)

    def test_column_type_mismatch_is_rejected(self):
        self.write_result(self.baseline, [["1"]], ordered=True, types=["integer"])
        self.write_result(self.candidate, [["1"]], ordered=True, types=["bigint"])

        result = self.compare()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("metadata mismatch", result.stdout)

    def test_missing_ordering_contract_is_reported_without_traceback(self):
        self.write_result(self.baseline, [["1"]], ordered=True, types=["integer"])
        self.write_result(self.candidate, [["1"]], ordered=True, types=["integer"])
        metadata_path = self.candidate / "q01.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        del metadata["ordered"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        result = self.compare()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("missing metadata fields", result.stdout)
        self.assertNotIn("Traceback", result.stdout)


if __name__ == "__main__":
    unittest.main()
