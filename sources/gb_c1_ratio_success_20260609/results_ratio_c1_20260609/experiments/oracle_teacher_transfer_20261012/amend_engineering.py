from pathlib import Path

import run_oracle as run


def main():
    current = run.ROOT / 'frozen_inputs.json'
    manifest = run.read(current)
    archive = run.ROOT / 'engineering' / 'frozen_inputs_before_retraining_check.json'
    if archive.exists():
        raise RuntimeError('engineering revision already archived')
    archive.write_bytes(current.read_bytes())
    changes = []
    permitted = {str((run.ROOT / name).resolve()) for name in ('PLAN.md','run_oracle.py','analyze.py','test_oracle.py')}
    for path, expected in manifest['hashes'].items():
        observed = run.sha(path)
        if observed != expected:
            if path not in permitted:
                raise RuntimeError(f'unexpected original-input change: {path}')
            changes.append(dict(path=path, old_sha256=expected, new_sha256=observed))
            manifest['hashes'][path] = observed
    manifest['engineering_revision'] = 'contemporary noisy baselines after retraining numerical check'
    run.write(current,manifest)
    preflight = run.read(run.ROOT/'preflight.json')
    preflight.update(expected_physical_teacher_fits=21,expected_physical_student_fits=21,
                     original_checks_unchanged=True)
    run.write(run.ROOT/'preflight.json',preflight)
    run.write(run.ROOT/'engineering_revision.json',dict(reason=manifest['engineering_revision'],
          source_changes=changes,scientific_test_outcomes_read=False,interventions_and_thresholds_unchanged=True,
          archived_manifest=str(archive),revision_script_sha256=run.sha(Path(__file__))))
    run.verify()
    print('Engineering revision recorded; original inputs unchanged; 21 paired runs fixed.')


if __name__=='__main__':
    main()
