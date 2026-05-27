from pathlib import Path

import argparse
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent

# Hyperparameters: edit these values before running this script.
# Use None for X_MAX/Y_MAX to crop to the image's right/bottom edge.
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "cropped"
X_MIN = 670
X_MAX = 1300
Y_MIN = 280
Y_MAX = 860

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop all images in one folder and save them with unchanged file names."
    )
    parser.add_argument("--x-min", type=int, default=X_MIN, help="Left crop coordinate in pixels.")
    parser.add_argument("--x-max", type=int, default=X_MAX, help="Right crop coordinate in pixels.")
    parser.add_argument("--y-min", type=int, default=Y_MIN, help="Top crop coordinate in pixels.")
    parser.add_argument("--y-max", type=int, default=Y_MAX, help="Bottom crop coordinate in pixels.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Folder containing images. Defaults to this script's folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder to save cropped images.",
    )
    return parser.parse_args()


def validate_crop_box(x_min, x_max, y_min, y_max):
    if x_min < 0 or y_min < 0:
        raise ValueError("x_min and y_min must be >= 0.")
    if x_max is not None and x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")
    if y_max is not None and y_min >= y_max:
        raise ValueError("y_min must be smaller than y_max.")


def iter_images(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def save_without_quality_loss(cropped, source_image, output_path):
    suffix = output_path.suffix.lower()
    save_kwargs = {}

    if "exif" in source_image.info:
        save_kwargs["exif"] = source_image.info["exif"]
    if "icc_profile" in source_image.info:
        save_kwargs["icc_profile"] = source_image.info["icc_profile"]

    if suffix in JPEG_EXTENSIONS:
        if cropped.mode in ("RGBA", "LA", "P"):
            cropped = cropped.convert("RGB")
        save_kwargs.update(quality=100, subsampling=0)
    elif suffix == ".png":
        save_kwargs["compress_level"] = 6
    elif suffix == ".webp":
        save_kwargs.update(lossless=True, quality=100)

    cropped.save(output_path, **save_kwargs)


def crop_images(input_dir, output_dir, x_min, x_max, y_min, y_max):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cropped_count = 0

    for image_path in iter_images(input_dir):
        with Image.open(image_path) as image:
            width, height = image.size
            actual_x_max = width if x_max is None else x_max
            actual_y_max = height if y_max is None else y_max
            crop_box = (x_min, y_min, actual_x_max, actual_y_max)

            if actual_x_max > width or actual_y_max > height:
                raise ValueError(
                    f"Crop box {crop_box} exceeds image size {width}x{height}: {image_path.name}"
                )

            cropped = image.crop(crop_box)
            save_without_quality_loss(cropped, image, output_dir / image_path.name)
            cropped_count += 1

    return cropped_count


def main():
    args = parse_args()

    validate_crop_box(args.x_min, args.x_max, args.y_min, args.y_max)
    cropped_count = crop_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    print(f"Cropped {cropped_count} image(s) into: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
