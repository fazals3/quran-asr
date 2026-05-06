# Runtime data

This directory is mounted into the Docker services as `/data`.

Tracked:
- `quran.json`: source Quran text used by `tools/build_quran_db.py`.

Generated or local-only:
- `quran.db`: build with `python3 tools/build_quran_db.py --input data/quran.json --output data/quran.db`.
- `alignment/v2/...`: precomputed Rust alignment assets. Put the latest asset directory here and set `AYAH_ALIGN_V2_ASSETS_DIR=/data/alignment/v2`.
- `hf/`: Hugging Face model cache.
- `jobs/`: uploaded audio and job outputs; old entries are removed by the Rust API job cleanup loop.
- `tmp/`: temporary audio files.
