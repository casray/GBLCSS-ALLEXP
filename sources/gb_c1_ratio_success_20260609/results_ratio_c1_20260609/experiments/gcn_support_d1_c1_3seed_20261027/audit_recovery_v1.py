"""Bind one missing audit helper while retaining every frozen scientific file."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

CODE = Path(__file__).resolve().parent


def bind_helper(module):
    module.relative_input = module.analysis.relative_input
    return module


def main():
    spec = importlib.util.spec_from_file_location('gcn_frozen_audit_recovery', CODE/'audit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    a = module.analysis
    root = Path(a.read(CODE/'config.json')['runtime'])
    owned = root/'audit_recovery_v1'
    owned.mkdir(exist_ok=True)
    for filename in ('audit_failure.json','launcher_status.json'):
        dest = owned/filename
        if (root/filename).exists() and not dest.exists():
            shutil.copy2(root/filename, dest)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = dict(started=a.now(), wrapper_sha256=fingerprint,
                   frozen_audit_sha256=a.sha(CODE/'audit.py'),
                   fix='relative_input = analysis.relative_input',
                   training_changed=False, selection_changed=False)
    a.save(owned/'receipt.json', receipt)
    try:
        result = bind_helper(module).audit(root=root)
        if a.sha(__file__) != fingerprint:
            raise RuntimeError('Audit wrapper changed during execution')
        receipt.update(status='passed', finished=a.now(), complete_sha256=a.sha(root/'complete.json'))
        a.save(owned/'receipt.json', receipt)
        a.save(CODE/'audit_recovery_receipt.json', receipt)
        a.save(root/'launcher_status.json', dict(stage='COMPLETE', time=a.now(),
               note='CPU audit recovered with explicit helper binding; original failure archived',
               recovery_receipt=str(owned/'receipt.json'), training_changed=False))
        a.save(CODE/'launcher_status.json', a.read(root/'launcher_status.json'))
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        receipt.update(status='failed', finished=a.now(), traceback=traceback.format_exc())
        a.save(owned/'receipt.json', receipt)
        raise


if __name__ == '__main__':
    main()
