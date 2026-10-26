"""Exact read-only reuse of the pilot's pure float64 numerical kernels."""
from protocol import import_file, PILOT

_kernel = import_file('confirmation_pilot_targets', PILOT/'targets.py')
evidence = _kernel.evidence
shrink = _kernel.shrink
groups = _kernel.groups
global_mass = _kernel.global_mass
match_mass = _kernel.match_mass
