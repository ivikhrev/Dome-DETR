"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn

from src.core import YAMLConfig


@contextmanager
def export_nms_mode(model: nn.Module, mode: str):
    modules = [module for module in model.modules() if hasattr(module, "export_nms_mode")]
    previous_modes = [module.export_nms_mode for module in modules]
    for module in modules:
        module.export_nms_mode = mode

    try:
        yield
    finally:
        for module, previous_mode in zip(modules, previous_modes):
            module.export_nms_mode = previous_mode


def get_output_file(resume: str | None, suffix: str, batch_size: int = 1) -> str:
    batch_suffix = "" if batch_size == 1 else f"-b{batch_size}"
    if resume:
        path = Path(resume)
        stem_parts = path.stem.split("-")
        if len(stem_parts) != 3 or stem_parts[0] != "aitod" or stem_parts[2] != "best":
            raise ValueError(
                f"Cannot infer model size from checkpoint name '{path.stem}'. "
                "Expected format: aitod-{s,m,l}-best"
            )
        model_size = stem_parts[1]
        if model_size not in {"s", "m", "l"}:
            raise ValueError(
                f"Unsupported model size '{model_size}' in checkpoint name '{path.stem}'. "
                "Expected one of: s, m, l"
            )
        return str(path.with_name(f"dome-detr-aitod-{model_size}{batch_suffix}{suffix}"))
    return f"dome-detr-aitod{batch_suffix}{suffix}"


def export_onnx(
    model: nn.Module,
    data: torch.Tensor,
    size: torch.Tensor,
    output_file: str,
    check: bool,
    simplify: bool,
    nms_mode: str,
) -> None:
    with export_nms_mode(model, nms_mode):
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
            external_data=False,
        )

    print(f"Export ONNX model done: {output_file}")

    if check:
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export ONNX model done...")

    if simplify:
        import onnx
        import onnxsim

        input_shapes = {
            "images": data.shape,
            "orig_target_sizes": size.shape,
        }
        onnx_model_simplify, check = onnxsim.simplify(
            output_file,
            test_input_shapes=input_shapes,
        )
        onnx.save(onnx_model_simplify, output_file)
        print(f"Simplify ONNX model {check}...")


def export_torchscript(
    model: nn.Module,
    data: torch.Tensor,
    size: torch.Tensor,
    output_file: str,
    nms_mode: str,
) -> None:
    with export_nms_mode(model, nms_mode):
        with torch.no_grad():
            scripted_model = torch.jit.trace(
                model,
                (data, size),
                strict=False,
                check_trace=False,
            )

    scripted_model.save(output_file)
    print(f"Export TorchScript model done: {output_file}")


def export_exported_program(
    model: nn.Module,
    data: torch.Tensor,
    size: torch.Tensor,
    output_file: str,
    nms_mode: str,
) -> None:
    with export_nms_mode(model, nms_mode):
        with torch.no_grad():
            exported_program = torch.export.export(
                model,
                args=(data, size),
                strict=False,
            )

    # torch.export specializes _assert_tensor_metadata nodes to the device used
    # during export. A model exported on CPU therefore fails before inference
    # when an adapter moves the ExportedProgram module and inputs to CUDA.
    # These nodes only validate metadata; removing them does not change model
    # computation or the remaining shape/range constraints.
    graph_module = exported_program.graph_module
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        for node in list(module.graph.nodes):
            if (
                node.target == torch.ops.aten._assert_tensor_metadata.default
                and not node.users
            ):
                module.graph.erase_node(node)

        module.graph.eliminate_dead_code()
        module.recompile()

    torch.export.save(exported_program, output_file)
    print(f"Export ExportedProgram model done: {output_file}")


def main(args):
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
        print("not load model.state_dict, use default init state dict...")

    eval_size = cfg.yaml_cfg.get("eval_spatial_size", None)
    if args.input_size is not None:
        input_h = input_w = args.input_size
    elif eval_size is not None:
        input_h, input_w = eval_size
    else:
        input_h = input_w = 640

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().eval()

    for batch_size in args.batch_size:
        data = torch.rand(batch_size, 3, input_h, input_w)
        size = torch.tensor([[input_h, input_w]]).repeat(batch_size, 1)

        with torch.no_grad():
            _ = model(data, size)

        if "onnx" in args.format:
            output_file = get_output_file(args.resume, ".onnx", batch_size)
            print(f"Exporting ONNX (batch={batch_size}, NMS={args.export_nms_mode}): {output_file}")
            export_onnx(
                model=model,
                data=data,
                size=size,
                output_file=output_file,
                check=args.check,
                simplify=args.simplify,
                nms_mode=args.export_nms_mode,
            )

        if "torchscript" in args.format:
            output_file = get_output_file(args.resume, ".torchscript", batch_size)
            print(
                f"Exporting TorchScript (batch={batch_size}, "
                f"NMS={args.export_nms_mode}): {output_file}"
            )
            export_torchscript(
                model=model,
                data=data,
                size=size,
                output_file=output_file,
                nms_mode=args.export_nms_mode,
            )

        if "pt2" in args.format:
            output_file = get_output_file(args.resume, ".pt2", batch_size)
            print(f"Exporting PT2 (batch={batch_size}, NMS={args.export_nms_mode}): {output_file}")
            export_exported_program(
                model=model,
                data=data,
                size=size,
                output_file=output_file,
                nms_mode=args.export_nms_mode,
            )


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
        "--format",
        nargs="+",
        choices=["onnx", "torchscript", "pt2"],
        default=["onnx"],
        help="Export format. Can be provided multiple times: --format onnx torchscript pt2.",
    )
    parser.add_argument(
        "--export-nms-mode",
        "--pt2-nms-mode",
        dest="export_nms_mode",
        choices=["parallel", "exact"],
        default="parallel",
        help=(
            "NMS implementation used by ONNX, TorchScript, and PT2 exports. "
            "'parallel' exports much faster and runs much faster on GPU but "
            "approximates greedy NMS; 'exact' preserves greedy NMS and produces "
            "a very large graph."
        ),
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
