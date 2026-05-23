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

# Number of bytes to read from the beginning and end of a file for inference.
# Increased BEGIN_BYTES from 512 to 1024 to improve detection accuracy for
# file formats that store identifying information deeper in the header.
DEFAULT_PADDING_TOKEN = 256
BEGIN_BYTES = 1024
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
                "
