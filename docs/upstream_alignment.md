# Upstream Alignment

## Project Boundary

SPVD is an independent project. Runtime code under `src/` must not import from
the upstream reference repository or from a local compatibility layer named
after that repository.

The only direct dependency for generic CLIP machinery is OpenCLIP. The project
uses OpenCLIP for standard CLIP model construction, backbone components,
tokenization compatibility, and image transforms.

## Local Responsibilities

Project-owned code is responsible for:

- SPVD model construction and forward outputs;
- project model registry and training entrypoint wiring;
- InfoNCE and sigmoid losses;
- data loading, logging, checkpointing, scheduling, and evaluation glue;
- SPVD-specific decomposition modules inside `src/model.py`.

`src/clip_components.py` contains the small set of OpenCLIP tower-building and
precision/preprocess helpers needed by `SPVDModel`.

## Baseline Boundary

Ordinary CLIP and SigLIP baseline models are constructed through
`open_clip.create_model`. SPVD models are constructed from project-local JSON
configs through `src/factory.py`.
