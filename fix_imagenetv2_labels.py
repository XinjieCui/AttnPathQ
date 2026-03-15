from pathlib import Path
import shutil


def main() -> None:
    root = Path("/home/cxj/DyadicFold/data/imagenetv2")
    src = root / "imagenetv2-matched-frequency-format-val"
    dst = root / "val"

    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)
    for idx in range(1000):
        target = src / str(idx)
        if not target.exists():
            raise FileNotFoundError(f"Missing class directory: {target}")
        (dst / f"{idx:03d}").symlink_to(target)

    print(f"Created {len(list(dst.iterdir()))} class links under {dst}")


if __name__ == "__main__":
    main()
