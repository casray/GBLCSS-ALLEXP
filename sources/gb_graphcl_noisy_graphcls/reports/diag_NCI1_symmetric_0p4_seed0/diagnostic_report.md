# Selection Diagnostic: NCI1 symmetric 0.4 seed0

## ce
- test_acc_at_best_val: 0.6302
- stopped_epoch: 65
- val_acc at epoch 0: 0.4623
- val_acc at warmup end (epoch 20): 0.6277
- val_acc at ramp end (epoch 40): 0.6399
- val_acc at stop (epoch 65): 0.6448
- keep_ratio at stop: 1.0000
- selected_clean_rate at stop: 0.6010
- selected_count at stop: 3288
- best val_acc: 0.6496 at epoch 35
- early_stop_before_ramp_end: False

## sel_topk
- test_acc_at_best_val: 0.6010
- stopped_epoch: 43
- val_acc at epoch 0: 0.4623
- val_acc at warmup end (epoch 20): 0.6277
- val_acc at ramp end (epoch 40): 0.5766
- val_acc at stop (epoch 43): 0.5742
- keep_ratio at stop: 0.6001
- selected_clean_rate at stop: 0.6842
- selected_count at stop: 1973
- best val_acc: 0.6326 at epoch 13
- early_stop_before_ramp_end: False

## sel_gb
- test_acc_at_best_val: 0.6034
- stopped_epoch: 56
- val_acc at epoch 0: 0.4623
- val_acc at warmup end (epoch 20): 0.6277
- val_acc at ramp end (epoch 40): 0.5718
- val_acc at stop (epoch 56): 0.6083
- keep_ratio at stop: 0.6001
- selected_clean_rate at stop: 0.6756
- selected_count at stop: 1973
- best val_acc: 0.6350 at epoch 26
- early_stop_before_ramp_end: False

## sel_noball
- test_acc_at_best_val: 0.6010
- stopped_epoch: 43
- val_acc at epoch 0: 0.4623
- val_acc at warmup end (epoch 20): 0.6277
- val_acc at ramp end (epoch 40): 0.5012
- val_acc at stop (epoch 43): 0.5036
- keep_ratio at stop: 0.6001
- selected_clean_rate at stop: 0.6356
- selected_count at stop: 1973
- best val_acc: 0.6326 at epoch 13
- early_stop_before_ramp_end: False

## Key diagnostic questions

1. CE stopped at epoch 65, sel_topk at 43, sel_gb at 56
2. sel_topk - ce = -0.0292 (selection hurts)
2. sel_gb - ce = -0.0268 (selection hurts)
2. sel_noball - ce = -0.0292 (selection hurts)