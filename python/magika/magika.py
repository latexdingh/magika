# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core Magika class for AI-powered file type detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from magika.types import MagikaResult, MagikaOutputFields, ModelFeatures
from magika.logger import get_logger

logger = get_logger()

# Number of bytes to read from the beginning and end of a file for inference
DEFAULT_PADDING_TOKEN = 256
BEGIN_BYTES = 512
MID_BYTES = 512
END_BYTES = 512


class Magika:
    """Main class for performing AI-powered file content type detection.

    This class loads a pre-trained deep learning model and uses it to
    identify the content type of files based on their byte sequences.

    Example usage:
        >>> m = Magika()
        >>> result = m.identify_path(Path("test.py"))
        >>> print(result.output.ct_label)
        'python'
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        prediction_mode: str = "high-confidence",
        no_dereference: bool = False,
        verbose: bool = False,
        debug: bool = False,
        use_colors: bool = False,
    ):
        """Initialize Magika with optional model directory and settings.

        Args:
            model_dir: Path to the directory containing the ONNX model and config.
                       If None, uses the bundled default model.
            prediction_mode: One of 'best-guess', 'high-confidence', or 'medium-confidence'.
            no_dereference: If True, do not follow symlinks.
            verbose: Enable verbose logging output.
            debug: Enable debug logging output.
            use_colors: Enable colored terminal output.
        """
        self._model_dir = model_dir
        self._prediction_mode = prediction_mode
        self._no_dereference = no_dereference
        self._verbose = verbose
        self._debug = debug
        self._use_colors = use_colors

        self._model = None
        self._config = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load the ONNX model and associated configuration."""
        if self._model_dir is None:
            # Use the default bundled model
            self._model_dir = Path(__file__).parent / "models" / "standard_v3"

        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self._model_dir}. "
                "Please ensure the model files are present."
            )

        logger.debug(f"Loading model from: {self._model_dir}")
        # Model loading is deferred to avoid heavy imports at module level
        self._model_loaded = False

    def identify_path(self, path: Path) -> MagikaResult:
        """Identify the content type of a single file path.

        Args:
            path: The file path to analyze.

        Returns:
            A MagikaResult containing the detected content type and metadata.
        """
        return self.identify_paths([path])[0]

    def identify_paths(self, paths: List[Path]) -> List[MagikaResult]:
        """Identify the content type of multiple file paths.

        Args:
            paths: A list of file paths to analyze.

        Returns:
            A list of MagikaResult objects, one per input path.
        """
        results = []
        for path in paths:
            try:
                result = self._identify_single_path(path)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                result = self._get_error_result(path, str(e))
            results.append(result)
        return results

    def identify_bytes(self, content: bytes) -> MagikaResult:
        """Identify the content type from raw bytes.

        Args:
            content: The raw bytes to analyze.

        Returns:
            A MagikaResult containing the detected content type.
        """
        raise NotImplementedError("identify_bytes will be implemented in a future version.")

    def _identify_single_path(self, path: Path) -> MagikaResult:
        """Internal method to process a single file path."""
        if not self._no_dereference:
            path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.is_dir():
            return self._get_directory_result(path)

        file_size = os.path.getsize(path)
        logger.debug(f"Processing file: {path} ({file_size} bytes)")

        # Placeholder: actual inference logic will be added with model integration
        return self._get_unknown_result(path)

    def _get_directory_result(self, path: Path) -> MagikaResult:
        """Return a result indicating the path is a directory."""
        from magika.types import MagikaResult, MagikaOutputFields
        return MagikaResult(
            path=path,
            output=MagikaOutputFields(
                ct_label="directory",
                mime_type="inode/directory",
                group="inode",
                magic="directory",
                description="A directory",
                extensions=[],
                score=1.0,
            ),
        )

    def _get_unknown_result(self, path: Path) -> MagikaResult:
        """Return a result for an unknown content type."""
        from magika.types import MagikaResult, MagikaOutputFields
        return MagikaResult(
            path=path,
            output=MagikaOutputFields(
                ct_label="unknown",
                mime_type="application/octet-stream",
                group="unknown",
                magic="data",
                description="Unknown binary data",
                extensions=[],
                score=0.0,
            ),
        )

    def _get_error_result(self, path: Path, error_msg: str) -> MagikaResult:
        """Return a result representing a processing error."""
        from magika.types import MagikaResult, MagikaOutputFields
        return MagikaResult(
            path=path,
            output=MagikaOutputFields(
                ct_label="error",
                mime_type="application/octet-stream",
                group="error",
                magic="error",
                description=f"Error: {error_msg}",
                extensions=[],
                score=0.0,
            ),
        )
