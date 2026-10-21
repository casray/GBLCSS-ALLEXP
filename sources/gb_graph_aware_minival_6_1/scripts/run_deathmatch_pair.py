from pathlib import Path

from run_deathmatch_common import ROOT, main


if __name__ == "__main__":
    main(
        default_dataset="Cora",
        default_noise_type="pair",
        default_noise_rate=0.4,
        default_out_dir=ROOT / "reports/deathmatch_pair_20260607",
    )
