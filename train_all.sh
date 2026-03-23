# python run_experiments.py --config configs/daformer/city2dense_uda_openset.py
# python run_experiments.py --config configs/daformer/syn2dense_uda_openset.py
# python run_experiments.py --config configs/daformer/gta2syn_uda_openset.py
# python run_experiments.py --config configs/daformer/syn2acdc_uda_openset.py

#!/bin/bash

# Cityscapes → Dense
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --config configs/daformer/city2dense_uda_openset_graph.py &

# Synth → Dense
CUDA_VISIBLE_DEVICES=1 python run_experiments.py --config configs/daformer/syn2dense_uda_openset_graph.py &

# GTA → Synth
CUDA_VISIBLE_DEVICES=2 python run_experiments.py --config configs/daformer/gta2syn_uda_openset_graph.py  &

# Synth → ACDC
CUDA_VISIBLE_DEVICES=3 python run_experiments.py --config configs/daformer/syn2acdc_uda_openset_graph.py &

wait
echo "All experiments finished."
