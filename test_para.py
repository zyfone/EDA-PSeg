import torch
import time
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.cnn import get_model_complexity_info
import types
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def add_forward_dummy(model):
    """
    给 MMSeg 模型添加 forward_dummy 方法，用于 FLOPs/MACs 统计
    """
    def forward_dummy(self, img):
        # backbone 提取特征
        x = self.extract_feat(img)
        # decode_head 可能有 forward_dummy
        if hasattr(self.decode_head, 'forward_dummy'):
            return self.decode_head.forward_dummy(x)
        else:
            return x
    model.forward_dummy = types.MethodType(forward_dummy, model)

def patch_forward_for_flops(model):
    """
    替换模型 forward，使 get_model_complexity_info 只传 img，不再报 img_metas 缺失
    """
    def forward_only_img(self, img, *args, **kwargs):
        return self.forward_dummy(img)
    model.forward = types.MethodType(forward_only_img, model)

def test_model_stats(config_file, input_shape=(3, 512, 512), device='cuda'):
    """
    统计 MMSeg 模型参数量、FLOPs、MACs、推理时间（每张图片）
    兼容 mmcv 1.3.7
    """
    # 1. 加载配置
    cfg = Config.fromfile(config_file)
    if 'pretrained' in cfg.model:
        cfg.model.pop('pretrained', None)

    # 2. 构建模型
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model.eval()
    model.to(device)

    # 3. 添加 forward_dummy
    add_forward_dummy(model)
    # 4. 替换 forward 用于 FLOPs
    patch_forward_for_flops(model)

    # 5. 参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {num_params / 1e6:.2f} M')

    # 6. FLOPs & MACs
    flops, _ = get_model_complexity_info(
        model,
        input_shape,
        as_strings=False,
        print_per_layer_stat=False
    )
    flops_g = flops / 1e9
    macs_g = flops_g / 2
    print(f'FLOPs: {flops_g:.2f} G')
    print(f'MACs: {macs_g:.2f} G')

    # 7. 推理时间 (ms / image)
    dummy = torch.randn(1, *input_shape).to(device)
    dummy_metas = [{'img_shape': input_shape, 'ori_shape': input_shape, 
                    'pad_shape': input_shape, 'scale_factor': 1.0}]

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model.forward_dummy(dummy)  # 推理时间测量也可直接用 forward_dummy

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
    # 替换为你的 MMSeg 配置文件路径
    config_file = '/home/zyfone/hard-disk/zyf/code-seg/ALL-DAFormer-SAM-Graph-Euler/configs/daformer/city2dense_uda_openset_graph.py'
    stats = test_model_stats(config_file)
