import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("joblib_first", [False, True])
def test_canonicalization_executor_is_independent_of_joblib(joblib_first: bool) -> None:
    # A fresh process prevents earlier Joblib calls from hiding the singleton collision.
    script = textwrap.dedent(
        """
        import sys

        from joblib import Parallel, delayed

        from retrochimera.inference.smiles_transformer import get_reusable_executor
        from retrochimera.utils.root_aligned_score import canonicalize_smiles_clear_map

        def run_joblib():
            assert Parallel(n_jobs=2, backend="loky")(
                delayed(abs)(value) for value in [-1, -2]
            ) == [1, 2]

        if sys.argv[1] == "True":
            run_joblib()
        lines = [("OCC", 0.8), ("CC", 0.2)]
        expected = [canonicalize_smiles_clear_map(line) for line in lines]
        executor = get_reusable_executor(max_workers=2, timeout=300)
        try:
            assert list(executor.map(canonicalize_smiles_clear_map, lines)) == expected
            run_joblib()
            assert get_reusable_executor(max_workers=2, timeout=300) is executor
            assert list(executor.map(canonicalize_smiles_clear_map, lines)) == expected
        finally:
            executor.shutdown(wait=True)

        replacement = get_reusable_executor(max_workers=2, timeout=300)
        try:
            assert replacement is not executor
            assert list(replacement.map(canonicalize_smiles_clear_map, lines)) == expected
        finally:
            replacement.shutdown(wait=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(joblib_first)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
