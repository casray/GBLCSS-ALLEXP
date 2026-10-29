"""Cross-device roundoff audit without changing any saved prediction."""
import torch


def assess_replay(saved, cpu, gpu, cpu_tolerance=2e-5):
    assert saved.shape==cpu.shape==gpu.shape
    assert all(torch.isfinite(x).all() for x in [saved,cpu,gpu]), 'Predictions must be finite'
    error=(cpu.double()-saved.double()).abs()
    maximum=float(error.max())
    assert maximum<=cpu_tolerance, ('CPU probability error',maximum)
    assert torch.equal(gpu,saved), ('GPU reload must reproduce the saved probabilities exactly',float((gpu-saved).abs().max()))
    predicted=saved.argmax(1);alternate=cpu.argmax(1)
    ids=(predicted!=alternate).nonzero().flatten()
    margin=saved.double()[ids,predicted[ids]]-saved.double()[ids,alternate[ids]]
    local_error=error.amax(1)[ids]
    assert (margin<=2*local_error).all(), 'CPU rank reversal exceeds measured probability roundoff'
    return dict(cpu_max_error=maximum,gpu_max_error=0.,gpu_bitwise_equal=True,
        cpu_disagreement_nodes=ids.tolist(),saved_classes=predicted[ids].tolist(),cpu_classes=alternate[ids].tolist(),
        saved_margin_to_cpu_winner=margin.tolist(),cpu_row_error=local_error.tolist())
