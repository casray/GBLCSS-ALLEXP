# Engineering record

- Before implementation, 12 behavioral tests failed because core.py did not exist. After implementation and reference checks, 14 tests passed.
- Initial prepare stopped before freezing or model training: CoraFull H.pt is a plain tensor, while Cora/CiteSeer artifacts are dictionaries containing H. protocol.py now supports both known existing formats. failure_prepare.json retains the original traceback. No source data, method parameters, split, target rule or test-based decision was changed.
- Preparation is rerun after this explicit format fix. Partial generated inputs may be overwritten only before frozen_inputs.json exists. Completed model fits are never overwritten.
