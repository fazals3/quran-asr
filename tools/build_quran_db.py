#!/usr/bin/env python3
import argparse
import datetime as _dt
import hashlib
import json
import os
import sqlite3
import sys

from quran_text import clean_token, expand_search_terms, normalize_arabic as simplify_arabic
from quran_text import strip_arabic_diacritics


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        PRAGMA user_version = 2;

        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE surah (
            surah_id INTEGER PRIMARY KEY,
            name_ar TEXT NOT NULL,
            name_ar_norm TEXT NOT NULL,
            transliteration TEXT NOT NULL,
            transliteration_norm TEXT NOT NULL,
            revelation_type TEXT NOT NULL,
            total_verses INTEGER NOT NULL
        );

        CREATE TABLE ayah (
            ayah_id INTEGER PRIMARY KEY,
            surah_id INTEGER NOT NULL REFERENCES surah(surah_id),
            ayah_num INTEGER NOT NULL,
            ayah_global INTEGER NOT NULL,
            ayah_key TEXT NOT NULL,
            text_ar TEXT NOT NULL,
            text_ar_nd TEXT NOT NULL,
            text_ar_simplified TEXT NOT NULL,
            text_ar_terms TEXT NOT NULL,
            UNIQUE(surah_id, ayah_num),
            UNIQUE(ayah_global),
            UNIQUE(ayah_key)
        );

        CREATE TABLE word (
            ayah_id INTEGER NOT NULL REFERENCES ayah(ayah_id) ON DELETE CASCADE,
            word_pos INTEGER NOT NULL,
            token_raw TEXT NOT NULL,
            token_clean TEXT NOT NULL,
            token_nd TEXT NOT NULL,
            token_simplified TEXT NOT NULL,
            PRIMARY KEY (ayah_id, word_pos)
        ) WITHOUT ROWID;

        -- Character-level substring index (3-grams) on space-stripped simplified text.
        CREATE TABLE ayah_ngram3 (
            ngram TEXT NOT NULL,
            ayah_id INTEGER NOT NULL REFERENCES ayah(ayah_id) ON DELETE CASCADE,
            pos INTEGER NOT NULL,
            PRIMARY KEY (ngram, ayah_id, pos)
        ) WITHOUT ROWID;

        CREATE VIEW v_ayah AS
        SELECT
            a.ayah_id,
            a.ayah_global,
            a.surah_id,
            s.name_ar AS surah_name_ar,
            s.transliteration AS surah_transliteration,
            s.revelation_type,
            a.ayah_num,
            a.ayah_key,
            a.text_ar,
            a.text_ar_nd,
            a.text_ar_simplified,
            a.text_ar_terms
        FROM ayah a
        JOIN surah s ON s.surah_id = a.surah_id;
        """
    )


def try_create_fts(conn: sqlite3.Connection) -> None:
    # External-content FTS keeps ayah text canonical in `ayah` while making search fast.
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE ayah_fts USING fts5(
                text_ar,
                text_ar_nd,
                text_ar_simplified,
                text_ar_terms,
                content='ayah',
                content_rowid='ayah_id',
                prefix='2 3 4 5',
                tokenize='unicode61'
            );
            """
        )
        conn.execute("INSERT INTO ayah_fts(ayah_fts) VALUES('rebuild');")
    except sqlite3.OperationalError as e:
        print(f"[warn] could not create ayah_fts: {e}", file=sys.stderr)

    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE surah_fts USING fts5(
                name_ar,
                name_ar_norm,
                transliteration,
                transliteration_norm,
                content='surah',
                content_rowid='surah_id',
                tokenize='unicode61'
            );
            """
        )
        conn.execute("INSERT INTO surah_fts(surah_fts) VALUES('rebuild');")
    except sqlite3.OperationalError as e:
        print(f"[warn] could not create surah_fts: {e}", file=sys.stderr)

    # Defensive: ensure `surah_fts` exists (some builds appear to skip creating it without error).
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='surah_fts' LIMIT 1;"
        ).fetchone()
        if not exists:
            conn.execute(
                """
                CREATE VIRTUAL TABLE surah_fts USING fts5(
                    name_ar,
                    name_ar_norm,
                    transliteration,
                    transliteration_norm,
                    content='surah',
                    content_rowid='surah_id',
                    tokenize='unicode61'
                );
                """
            )
            conn.execute("INSERT INTO surah_fts(surah_fts) VALUES('rebuild');")
    except sqlite3.OperationalError as e:
        print(f"[warn] could not ensure surah_fts: {e}", file=sys.stderr)


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_ayah_surah_num ON ayah(surah_id, ayah_num);
        CREATE INDEX idx_ayah_global ON ayah(ayah_global);

        CREATE INDEX idx_word_token_nd ON word(token_nd, ayah_id);
        CREATE INDEX idx_word_token_simplified ON word(token_simplified, ayah_id);

        CREATE INDEX idx_ngram3 ON ayah_ngram3(ngram, ayah_id);
        CREATE INDEX idx_ngram3_ayah ON ayah_ngram3(ayah_id);
        """
    )


def iter_surahs(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON list of surahs")
    return data


def main() -> int:
    p = argparse.ArgumentParser(description="Build an efficient SQLite Quran database from quran-json.")
    p.add_argument("--input", default="quran.json", help="Path to quran.json")
    p.add_argument("--output", default="quran.db", help="Output SQLite DB path")
    p.add_argument("--source-url", default="", help="Optional source URL to store in meta")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Input JSON not found: {args.input}", file=sys.stderr)
        return 2

    if os.path.exists(args.output):
        os.remove(args.output)

    created_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    source_sha256 = sha256_file(args.input)

    surahs = iter_surahs(args.input)

    conn = sqlite3.connect(args.output)
    try:
        conn.execute("PRAGMA journal_mode = OFF;")
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -200000;")  # ~200MB
        conn.execute("PRAGMA foreign_keys = ON;")

        create_schema(conn)

        conn.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?);",
            [
                ("schema_version", "2"),
                ("created_at_utc", created_at),
                ("source_file", os.path.basename(args.input)),
                ("source_sha256", source_sha256),
                ("source_url", args.source_url),
            ],
        )

        surah_rows = []
        for s in surahs:
            surah_id = int(s["id"])
            name_ar = str(s["name"])
            name_ar_norm = simplify_arabic(name_ar)
            translit = str(s["transliteration"])
            translit_norm = translit.casefold()
            revelation_type = str(s["type"])
            total_verses = int(s["total_verses"])
            surah_rows.append(
                (
                    surah_id,
                    name_ar,
                    name_ar_norm,
                    translit,
                    translit_norm,
                    revelation_type,
                    total_verses,
                )
            )
        conn.executemany(
            """
            INSERT INTO surah(
                surah_id, name_ar, name_ar_norm, transliteration, transliteration_norm, revelation_type, total_verses
            ) VALUES(?, ?, ?, ?, ?, ?, ?);
            """,
            surah_rows,
        )

        ayah_rows = []
        word_rows = []
        ngram_rows = []
        ayah_global = 0

        def flush():
            nonlocal ayah_rows, word_rows, ngram_rows
            if ayah_rows:
                conn.executemany(
                    """
                    INSERT INTO ayah(
                        ayah_id, surah_id, ayah_num, ayah_global, ayah_key, text_ar, text_ar_nd, text_ar_simplified, text_ar_terms
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    ayah_rows,
                )
                ayah_rows = []
            if word_rows:
                conn.executemany(
                    """
                    INSERT INTO word(
                        ayah_id, word_pos, token_raw, token_clean, token_nd, token_simplified
                    ) VALUES(?, ?, ?, ?, ?, ?);
                    """,
                    word_rows,
                )
                word_rows = []
            if ngram_rows:
                conn.executemany(
                    "INSERT INTO ayah_ngram3(ngram, ayah_id, pos) VALUES(?, ?, ?);",
                    ngram_rows,
                )
                ngram_rows = []

        for s in surahs:
            surah_id = int(s["id"])
            verses = s["verses"]
            for v in verses:
                ayah_num = int(v["id"])
                text_ar = str(v["text"])
                text_ar_nd = strip_arabic_diacritics(text_ar)
                text_ar_s = simplify_arabic(text_ar)
                text_ar_terms = expand_search_terms(text_ar_s)

                ayah_global += 1
                ayah_id = ayah_global
                ayah_key = f"{surah_id}:{ayah_num}"
                ayah_rows.append(
                    (
                        ayah_id,
                        surah_id,
                        ayah_num,
                        ayah_global,
                        ayah_key,
                        text_ar,
                        text_ar_nd,
                        text_ar_s,
                        text_ar_terms,
                    )
                )

                tokens = text_ar.split()
                for pos, token in enumerate(tokens, start=1):
                    tclean = clean_token(token)
                    tnd = strip_arabic_diacritics(tclean)
                    ts = simplify_arabic(tclean).replace(" ", "")
                    word_rows.append((ayah_id, pos, token, tclean, tnd, ts))

                compact = text_ar_s.replace(" ", "")
                for i in range(len(compact) - 2):
                    ngram_rows.append((compact[i : i + 3], ayah_id, i + 1))

                if len(word_rows) >= 50000 or len(ngram_rows) >= 200000 or len(ayah_rows) >= 5000:
                    flush()

        flush()
        create_indexes(conn)
        try_create_fts(conn)

        conn.execute("ANALYZE;")
        conn.execute("PRAGMA optimize;")
        conn.commit()
        conn.execute("PRAGMA journal_mode = DELETE;")
        conn.execute("PRAGMA synchronous = NORMAL;")

    finally:
        conn.close()

    print(f"Wrote {args.output} (surahs={len(surahs)}, ayahs={ayah_global}, json_sha256={source_sha256[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
