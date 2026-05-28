.PHONY: help mac setup mac-build db clean-mac

help:
	@echo "Quran-ASR — macOS (Apple Silicon / CoreML) targets:"
	@echo "  make setup      One-time: pull LFS assets, build DB, build everything"
	@echo "  make mac        Run the full native stack (transcriber + API + demo)"
	@echo "  make mac-build  Build the Swift transcriber and Rust API (release)"
	@echo "  make db         (Re)build data/quran.db from data/quran.json"
	@echo "  make clean-mac  Remove Swift/Rust build artifacts"

setup:
	./scripts/setup-mac.sh

mac:
	./scripts/run-mac.sh

mac-build:
	cd mac/transcriber-swift && swift build -c release
	cd api && cargo build --release

db:
	python3 tools/build_quran_db.py --input data/quran.json --output data/quran.db

clean-mac:
	rm -rf mac/transcriber-swift/.build
	cd api && cargo clean
