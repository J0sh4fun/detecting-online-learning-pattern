from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
FRONTEND_MODELS_DIR = ROOT_DIR.parent / "web_app" / "frontend" / "public" / "models"


def resolve_weights_path(weights_arg: Path) -> Path:
    if weights_arg.is_absolute():
        candidates = [weights_arg]
    else:
        candidates = [
            Path.cwd() / weights_arg,
            ROOT_DIR / weights_arg,
            ROOT_DIR / weights_arg.name,
            MODELS_DIR / weights_arg.name,
            ROOT_DIR.parent / weights_arg.name,
        ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    checked = "\n".join(f"- {p.resolve()}" for p in candidates)
    raise FileNotFoundError(
        f"YOLO weights not found from '{weights_arg}'. Checked:\n{checked}\n"
        "Provide a valid file with --yolo-weights."
    )


def export_yolo(weights_path: Path, output_dir: Path) -> Path:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    export_result = model.export(format="onnx", dynamic=False, opset=12)
    exported_path = Path(export_result)
    if not exported_path.is_absolute():
        exported_path = (Path.cwd() / exported_path).resolve()
    if not exported_path.exists():
        raise FileNotFoundError(f"YOLO export returned path not found: {exported_path}")

    final_path = output_dir / f"{weights_path.stem}.onnx"
    if exported_path != final_path:
        if final_path.exists():
            final_path.unlink()
        shutil.move(str(exported_path), str(final_path))
    return final_path


def export_posture(model_pkl: Path, output_onnx: Path) -> tuple[int, str, str]:
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        if not model_pkl.exists():
            return 1, "", f"Missing model file: {model_pkl}"

        model = joblib.load(model_pkl)
        options = {type(model): {"zipmap": False}}
        onx = convert_sklearn(
            model,
            initial_types=[("float_input", FloatTensorType([None, 8]))],
            options=options,
        )
        output_onnx.write_bytes(onx.SerializeToString())
        return 0, str(output_onnx), ""
    except Exception as e:
        import traceback

        return 1, "", traceback.format_exc()


def copy_to_frontend(src: Path) -> Path:
    FRONTEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = FRONTEND_MODELS_DIR / src.name
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Export posture/yolo models to ONNX.")
    parser.add_argument(
        "--target",
        choices=["all", "posture", "yolo"],
        default="all",
        help="Choose which model to export.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("yolo26s.pt"),
        help="YOLO .pt path (default: yolo26s.pt with auto-search).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for ONNX files.",
    )
    parser.add_argument(
        "--copy-frontend",
        action="store_true",
        help="Also copy exported ONNX files to web_app/frontend/public/models.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []
    errors: list[str] = []

    if args.target in ("all", "yolo"):
        try:
            weights = resolve_weights_path(args.yolo_weights)
            yolo_onnx = export_yolo(weights, output_dir)
            exported.append(yolo_onnx)
            print(f"[OK] YOLO exported: {yolo_onnx}")
        except Exception as exc:
            errors.append(f"[FAIL] YOLO export: {exc}")

    if args.target in ("all", "posture"):
        model_pkl = MODELS_DIR / "best_posture_model.pkl"
        posture_onnx = output_dir / "best_posture_model.onnx"
        return_code, stdout_text, stderr_text = export_posture(model_pkl, posture_onnx)
        if return_code == 0 and posture_onnx.exists():
            exported.append(posture_onnx)
            print(f"[OK] Posture exported: {posture_onnx}")
        else:
            details = stderr_text or stdout_text or f"Process exited with code {return_code}"
            errors.append(
                "[FAIL] Posture export crashed. "
                "This is usually an environment issue with skl2onnx/onnx binaries.\n"
                f"Details: {details}"
            )

    if args.copy_frontend:
        for file_path in exported:
            copied = copy_to_frontend(file_path)
            print(f"[OK] Copied to frontend: {copied}")

    print("\n=== Export Summary ===")
    if exported:
        for p in exported:
            print(f"- {p}")
    else:
        print("- No files exported.")
    if errors:
        for e in errors:
            print(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
