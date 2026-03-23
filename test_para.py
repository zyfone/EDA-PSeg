import torch
import time
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.cnn import get_model_complexity_info
import types
import warnings

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def add_forward_dummy(model):
    """
    Adds a 'forward_dummy' method to the MMSeg model for FLOPs/MACs calculation.
    """
    def forward_dummy(self, img):
        # Extract features using the backbone
        x = self.extract_feat(img)
        # Check if decode_head has its own forward_dummy; otherwise, return features
        if hasattr(self.decode_head, 'forward_dummy'):
            return self.decode_head.forward_dummy(x)
        else:
            return x
    model.forward_dummy = types.MethodType(forward_dummy, model)

def patch_forward_for_flops(model):
    """
    Patches the model's forward method so get_model_complexity_info only 
    passes 'img', avoiding 'img_metas' missing errors.
    """
    def forward_only_img(self, img, *args, **kwargs):
        return self.forward_dummy(img)
    model.forward = types.MethodType(forward_only_img, model)

def test_model_stats(config_file, input_shape=(3, 512, 512), device='cuda'):
    """
    Calculates model parameters, FLOPs, MACs, and inference time per image.
    Compatible with mmcv 1.3.7.
    """
    # 1. Load Configuration
    cfg = Config.fromfile(config_file)
    if 'pretrained' in cfg.model:
        cfg.model.pop('pretrained', None)

    # 2. Build the Model
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model.eval()
    model.to(device)

    # 3. Add forward_dummy for compatibility
    add_forward_dummy(model)
    
    # 4. Patch forward for FLOPs calculation tool
    patch_forward_for_flops(model)

    # 5. Calculate Parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {num_params / 1e6:.2f} M')

    # 6. Calculate FLOPs & MACs
    flops, _ = get_model_complexity_info(
        model,
        input_shape,
        as_strings=False,
        print_per_layer_stat=False
    )
    flops_g = flops / 1e9
    # Note: Traditionally, 1 GFLOPs ≈ 2 GMACs in many frameworks
    macs_g = flops_g / 2 
    print(f'FLOPs: {flops_g:.2f} G')
    print(f'MACs: {macs_g:.2f} G')

    # 7. Measure Inference Time (ms / image)
    dummy = torch.randn(1, *input_shape).to(device)
    
    # Warm-up phase
    for _ in range(10):
        with torch.no_grad():
            _ = model.forward_dummy(dummy)

    n_runs = 100
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model.forward_dummy(dummy)
            
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    time_per_image = (end - start) / n_runs * 1000
    print(f'Inference time per image: {time_per_image:.2f} ms')

    return {
        'params_m': num_params / 1e6,
        'flops_g': flops_g,
        'macs_g': macs_g,
        'time_ms_per_image': time_per_image
    }

if __name__ == '__main__':
    # Replace with your actual MMSeg configuration path
    config_file = '/home/zyfone/hard-disk/zyf/code-seg/ALL-DAFormer-SAM-Graph-Euler/configs/daformer/city2dense_uda_openset_graph.py'
    stats = test_model_stats(config_file)
