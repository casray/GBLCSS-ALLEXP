def run(data, config, outdir) -> dict:
    """Fit on noisy train labels; choose by val_y only; write best CPU probabilities."""
    validate_data(data)
    cfg = resolve_config(data.name, config)
    out = Path(outdir).resolve()
    artifacts = [out / filename for filename in ('epochs.jsonl', 'checkpoint.pt', 'probabilities.pt')]
    if any((path.exists() for path in artifacts)):
        raise FileExistsError('GSSC artifacts already exist; use a fresh outdir')
    out.mkdir(parents=True, exist_ok=True)
    device, dtype = (data.x.device, data.x.dtype)
    model = NeighContrast(data.x.shape[1], cfg['hidden'], data.num_classes, cfg).to(device=device, dtype=dtype)
    mixup = MixupScale(data.x.shape[1], cfg['hidden'], cfg).to(device=device, dtype=dtype)
    sampler = EdgeSampler(data.x.shape[1], cfg['hidden'], cfg).to(device=device, dtype=dtype)
    modules = (model, mixup, sampler)
    parameters_by_module = {name: sum((p.numel() for p in module.parameters())) for name, module in zip(('model', 'mixup', 'sampler'), modules)}
    src_all, dst_all = data.edge_index
    best_val, best_epoch, best_phase, best_phase_epoch = (-math.inf, 0, '', 0)
    best_state = None
    global_epoch, classifier_epoch = (0, 0)
    diagnostics = {'official_commit': OFFICIAL_COMMIT, 'test_labels_read': False, 'validation_labels_in_optimization': False, 'cpu_training': False, 'num_nodes': data.x.shape[0], 'num_raw_edges': src_all.numel(), 'parameters_by_module': parameters_by_module, 'parameter_fallback': cfg['official_parameter_dataset'] == 'cora' and data.name.lower() != 'cora', 'sampler_updates': 0, 'homophily_updates': 0, 'homophily_skipped': 0, 'empty_sampled_epochs': 0, 'singleton_edges_skipped': 0, 'uncovered_train_classes': [], 'phase_budgets': cfg['phase_budgets'], 'source_sha256': {str(path.relative_to(VENDOR)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (VENDOR / 'code' / 'main.py', VENDOR / 'code' / 'model.py', VENDOR / 'param' / 'best_parameters.json')}}
    main_optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    main_scale_optimizer = torch.optim.Adam(mixup.parameters(), lr=cfg['mixup_lr'], weight_decay=cfg['weight_decay'])
    main_edge_optimizer = torch.optim.Adam(sampler.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    with artifacts[0].open('w', encoding='utf-8') as log:

        def record(phase, phase_epoch, loss, details, eligible):
            nonlocal global_epoch, classifier_epoch, best_val, best_epoch, best_state, best_phase, best_phase_epoch
            global_epoch += 1
            if eligible:
                classifier_epoch += 1
            val = validation_accuracy(model, data)
            if not math.isfinite(val):
                raise FloatingPointError('Non-finite validation accuracy')
            selected = eligible and val > best_val
            if selected:
                best_val, best_epoch = (val, classifier_epoch)
                best_phase, best_phase_epoch = (phase, phase_epoch)
                best_state = _cpu_state(model, mixup, sampler)
            row = dict(global_epoch=global_epoch, epoch=classifier_epoch if eligible else None, phase=phase, phase_epoch=phase_epoch, loss=loss, val=val, checkpoint_eligible=eligible, selected=selected, **details)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
        edge_optimizer = torch.optim.Adam(sampler.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        for epoch in range(1, cfg['ep_epochs'] + 1):
            sampler.train()
            sampled = sampler(data.x, src_all, dst_all)
            loss = F.binary_cross_entropy_with_logits(sampled, torch.ones_like(sampled))
            _finite(loss, 'edge_pretrain')
            edge_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            edge_optimizer.step()
            diagnostics['sampler_updates'] += 1
            record('edge_pretrain', epoch, float(loss.detach()), {}, False)
            del sampled, loss
        warm_optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        warm_scale_optimizer = torch.optim.Adam(mixup.parameters(), lr=cfg['mixup_lr'], weight_decay=cfg['weight_decay'])
        for phase, budget, optimizer, scale_optimizer in (('classifier_pretrain', cfg['cl_epochs'], warm_optimizer, warm_scale_optimizer), ('main', cfg['epochs'], main_optimizer, main_scale_optimizer)):
            for epoch in range(1, budget + 1):
                sampled, edge_weights = (None, None)
                if phase == 'main':
                    sampler.train()
                    sampled = sampler(data.x, src_all, dst_all)
                    selected = sampled.detach() == 1
                    src, dst = (src_all[selected], dst_all[selected])
                    edge_weights = sampled[selected]
                else:
                    src, dst = (src_all, dst_all)
                details = _train_classifier_epoch(model, mixup, data, src, dst, optimizer, scale_optimizer, cfg)
                details['sampled_edges'] = src.numel()
                diagnostics['singleton_edges_skipped'] += details['singleton_edges_skipped']
                diagnostics['uncovered_train_classes'] = sorted(set(diagnostics['uncovered_train_classes'] + details['uncovered_train_classes']))
                if phase == 'main':
                    diagnostics['empty_sampled_epochs'] += int(src.numel() == 0)
                    model.train()
                    with torch.no_grad():
                        pred = model.update(data.x)
                    hom_mean, hom_sum = homophily_loss(pred, src, dst, edge_weights)
                    if hom_mean is not None:
                        _finite(hom_mean, 'homophily')
                        main_edge_optimizer.zero_grad(set_to_none=True)
                        hom_mean.backward()
                        main_edge_optimizer.step()
                        diagnostics['homophily_updates'] += 1
                        diagnostics['sampler_updates'] += 1
                    else:
                        diagnostics['homophily_skipped'] += 1
                    details.update(homophily_mean=None if hom_mean is None else float(hom_mean.detach()), homophily_sum=None if hom_sum is None else float(hom_sum.detach()))
                    del pred, hom_mean, hom_sum
                record(phase, epoch, details['edge_total'] + details['center'], details, True)
                del sampled, edge_weights
    if best_state is None:
        raise RuntimeError('No eligible classifier checkpoint')
    for name, module in zip(('model', 'mixup', 'sampler'), modules):
        module.load_state_dict(best_state[name])
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x)
        probabilities = logits.softmax(-1).detach().cpu()
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError('Non-finite saved probabilities')
    diagnostics.update(completed_global_epochs=global_epoch, completed_classifier_epochs=classifier_epoch, best_phase=best_phase, best_phase_epoch=best_phase_epoch, inference_parameters=model.inference_parameters(), memory_order='edge sampler: O(N*hidden + E + edge_chunk_size*hidden), excluding raw features/parameters; no N^2 adjacency')
    torch.save(dict(state=best_state, config=cfg, best_epoch=best_epoch, best_val=best_val, diagnostics=diagnostics), artifacts[1])
    torch.save(probabilities, artifacts[2])
    return dict(best_epoch=best_epoch, best_val=best_val, parameters=model.inference_parameters(), config=cfg, probabilities_file=str(artifacts[2]), diagnostics=diagnostics)
