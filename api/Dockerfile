FROM rust:1.83-bullseye AS builder

WORKDIR /app

COPY api/Cargo.toml /app/Cargo.toml
COPY api/Cargo.lock /app/Cargo.lock
RUN mkdir -p /app/src && echo "fn main(){}" > /app/src/main.rs
RUN cargo build --release

COPY api/src /app/src
COPY api/assets /app/assets
# The COPY keeps source mtimes from the host; if they go "backwards" vs the dummy build,
# Cargo can incorrectly treat the crate as unchanged. Touch after COPY to force rebuild.
RUN find /app/src /app/assets -type f -exec touch {} +
RUN cargo build --release

FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/quran-asr-api /usr/local/bin/api

ENV RUST_LOG=info

EXPOSE 8001

CMD ["/usr/local/bin/api"]
