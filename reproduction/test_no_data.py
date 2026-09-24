"""Explicit no-CIFAR test entry: never selects test_sampler_real."""
import json,sys,unittest
from pathlib import Path
from test_semantics import Semantics
from test_preflight import Preflight
from test_results import Results
if __name__=='__main__':
    suite=unittest.TestSuite()
    for cls in (Semantics,Preflight,Results):
        for name in unittest.defaultTestLoader.getTestCaseNames(cls):
            if name!='test_sampler_real':suite.addTest(cls(name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path(__file__).resolve().parent.parent/'artifacts/preflight/test-result.json'
    out.write_text(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),successful=result.wasSuccessful(),scope='Synthetic data only; official teacher checkpoint is loaded. CIFAR-100 is not read; no benchmark training.'),indent=2))
    sys.exit(not result.wasSuccessful())
