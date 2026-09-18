# wav2vec2

Unofficial implementation of wav2vec 2.0 [[PDF](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf)]

## Prerequisites

- uv [[installation](https://docs.astral.sh/uv/getting-started/installation/)]

## Installation

For CPU environments

```sh
uv sync --extra cpu
```

For GPU environments

```sh
uv sync --extra cu126 # For CUDA 12.x
uv sync --extra cu130 # For CUDA 13.x
```
