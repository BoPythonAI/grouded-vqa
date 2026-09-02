from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    code_root: Path
    data_root: Path
    model_root: Path
    output_root: Path
    log_root: Path

    @classmethod
    def from_environment(cls) -> ProjectPaths:
        project = Path(os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/vision-language"))
        return cls(
            project_root=project,
            code_root=Path(os.environ.get("VQA_CODE_ROOT", project / "code/grounded-vqa")),
            data_root=Path(os.environ.get("VQA_DATA_ROOT", project / "data/vqav2")),
            model_root=Path(os.environ.get("VQA_MODEL_ROOT", project / "models")),
            output_root=Path(os.environ.get("VQA_OUTPUT_ROOT", project / "outputs")),
            log_root=Path(os.environ.get("VQA_LOG_ROOT", project / "logs")),
        )

    def ensure(self) -> None:
        for path in (
            self.data_root,
            self.model_root,
            self.output_root,
            self.log_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
