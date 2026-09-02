from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from grounded_vqa.paths import ProjectPaths


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    minimum_free_gib: float
    extracted: str


ARTIFACTS = {
    "train_questions": Artifact(
        "train_questions",
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        1.0,
        "v2_OpenEnded_mscoco_train2014_questions.json",
    ),
    "val_questions": Artifact(
        "val_questions",
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        1.0,
        "v2_OpenEnded_mscoco_val2014_questions.json",
    ),
    "train_annotations": Artifact(
        "train_annotations",
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        1.0,
        "v2_mscoco_train2014_annotations.json",
    ),
    "val_annotations": Artifact(
        "val_annotations",
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        1.0,
        "v2_mscoco_val2014_annotations.json",
    ),
    "train_images": Artifact(
        "train_images",
        "http://images.cocodataset.org/zips/train2014.zip",
        35.0,
        "train2014",
    ),
    "val_images": Artifact(
        "val_images",
        "http://images.cocodataset.org/zips/val2014.zip",
        15.0,
        "val2014",
    ),
}


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def download(artifact: Artifact, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / Path(artifact.url).name
    partial = target.with_suffix(target.suffix + ".part")
    if target.is_file():
        return target
    if free_gib(destination) < artifact.minimum_free_gib:
        raise RuntimeError(
            f"Insufficient free space for {artifact.name}: "
            f"need {artifact.minimum_free_gib:.1f} GiB guard, "
            f"have {free_gib(destination):.1f} GiB"
        )

    resume_at = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
    with requests.get(artifact.url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        append = resume_at > 0 and response.status_code == 206
        if resume_at and not append:
            resume_at = 0
        total = int(response.headers.get("content-length", 0)) + resume_at
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle, tqdm(
            total=total,
            initial=resume_at,
            unit="B",
            unit_scale=True,
            desc=artifact.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    partial.replace(target)
    return target


def extract_checked(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member in {archive}: {bad_member}")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official VQAv2 data")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_environment()
    paths.ensure()
    archives = paths.data_root / "archives"
    selections = [f"{args.split}_questions", f"{args.split}_annotations"]
    if args.include_images:
        selections.append(f"{args.split}_images")

    for key in selections:
        artifact = ARTIFACTS[key]
        if (paths.data_root / artifact.extracted).exists():
            print(f"{artifact.name}: already extracted; skipping")
            continue
        archive = download(artifact, archives)
        extract_checked(archive, paths.data_root)
        if not args.keep_archives:
            archive.unlink()


if __name__ == "__main__":
    main()
