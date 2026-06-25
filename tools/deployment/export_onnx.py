"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn

from src.core import YAMLConfig


def get_output_file(resume: str | None, batch_size: int = 1) -> str:
    batch_suffix = "" if batch_size == 1 else f".b{batch_size}"
    if resume:
        path = Path(resume)
        return str(path.with_name(f"{path.stem}{batch_suffix}.onnx"))
    return f"model{batch_suffix}.onnx"


def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    eval_size = cfg.yaml_cfg.get("eval_spatial_size", None)
    if args.input_size is not None:
        input_h = input_w = args.input_size
    elif eval_size is not None:
        input_h, input_w = eval_size
    else:
        input_h = input_w = 640

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().eval()

    # data = torch.rand(args.batch_size, 3, input_h, input_w)

    for batch_size in args.batch_size:
        data = torch.randint(
            low=0,
            high=256,
            size=(batch_size, 3, input_h, input_w),
            dtype=torch.uint8,
        )

        size = torch.tensor([[input_h, input_w]]).repeat(batch_size, 1)
        with torch.no_grad():
            _ = model(data, size)

        output_file = get_output_file(args.resume, batch_size)

        torch.onnx.export(
            model,
            (data, size),
            output_file,
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes=None,
            opset_version=18,
            verbose=False,
            do_constant_folding=True,
            external_data=False
        )

        if args.check:
            import onnx

            onnx_model = onnx.load(output_file)
            onnx.checker.check_model(onnx_model)
            print("Check export onnx model done...")

        if args.simplify:
            import onnx
            import onnxsim

            input_shapes = {"images": data.shape, "orig_target_sizes": size.shape}
            onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
            onnx.save(onnx_model_simplify, output_file)
            print(f"Simplify onnx model {check}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/dome/dfine_hgnetv2_l_coco.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Square input size. Defaults to eval_spatial_size from the config when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1],
        help="Fixed export batch size(s). Example: --batch-size 4 8 16.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    main(args)
