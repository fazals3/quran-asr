use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use unicode_general_category::{get_general_category, GeneralCategory};

pub fn format_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as i64;
    let hours = total_ms / 3_600_000;
    let rem = total_ms % 3_600_000;
    let minutes = rem / 60_000;
    let rem2 = rem % 60_000;
    let secs = rem2 / 1_000;
    let ms = rem2 % 1_000;
    format!("{hours:02}:{minutes:02}:{secs:02}.{ms:03}")
}

fn is_arabic_diacritic(ch: char) -> bool {
    let c = ch as u32;
    (0x0610..=0x061A).contains(&c)
        || (0x064B..=0x065F).contains(&c)
        || c == 0x0670
        || (0x06D6..=0x06DC).contains(&c)
        || (0x06DD..=0x06E4).contains(&c)
        || (0x06E7..=0x06E8).contains(&c)
        || (0x06EA..=0x06ED).contains(&c)
}

fn translate_arabic(ch: char) -> char {
    match ch as u32 {
        0x0623 | 0x0625 | 0x0622 | 0x0671 => '\u{0627}', // أ/إ/آ/ٱ -> ا
        0x0624 => '\u{0648}',                             // ؤ -> و
        0x0626 | 0x0649 => '\u{064A}',                    // ئ/ى -> ي
        0x0629 => '\u{0647}',                             // ة -> ه
        _ => ch,
    }
}

pub fn strip_arabic_diacritics(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == '\u{0640}' {
            continue; // tatweel
        }
        if is_arabic_diacritic(ch) {
            continue;
        }
        out.push(ch);
    }
    out
}

fn keep_general_category(cat: GeneralCategory) -> bool {
    matches!(
        cat,
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
            | GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
            | GeneralCategory::DecimalNumber
    )
}

pub fn normalize_arabic(text: &str) -> String {
    let stripped = strip_arabic_diacritics(text);

    let mut kept = String::with_capacity(stripped.len());
    let mut prev_space = false;

    for ch0 in stripped.chars() {
        let ch = translate_arabic(ch0);

        if ch.is_whitespace() {
            if !prev_space {
                kept.push(' ');
            }
            prev_space = true;
            continue;
        }

        let cat = get_general_category(ch);
        if keep_general_category(cat) {
            kept.push(ch);
            prev_space = false;
            continue;
        }

        // Keep Arabic-Indic digits too.
        if ('\u{0660}'..='\u{0669}').contains(&ch) {
            kept.push(ch);
            prev_space = false;
            continue;
        }

        // Drop punctuation/symbols.
    }

    kept.trim().to_string()
}

fn is_word_char(ch: char) -> bool {
    if ch == '_' {
        return true;
    }
    if ('\u{0660}'..='\u{0669}').contains(&ch) {
        return true;
    }
    keep_general_category(get_general_category(ch))
}

pub fn clean_token(token: &str) -> String {
    let t = token;
    let mut start: Option<usize> = None;
    let mut end: Option<usize> = None;
    for (idx, ch) in t.char_indices() {
        if is_word_char(ch) {
            start.get_or_insert(idx);
            end = Some(idx + ch.len_utf8());
        }
    }
    if let (Some(s), Some(e)) = (start, end) {
        let cleaned = &t[s..e];
        if !cleaned.is_empty() {
            return cleaned.to_string();
        }
    }
    t.to_string()
}

pub fn normalize_token(token: &str) -> String {
    let token = clean_token(token.trim());
    if token.is_empty() {
        return String::new();
    }
    let simplified = normalize_arabic(&token).replace(' ', "");
    // Transcription occasionally emits exaggerated letter repeats (e.g., "يسسس") for very short ayahs.
    // Collapse runs of >=3 identical chars to a single char, but preserve real double letters (e.g., "الله").
    let mut out = String::with_capacity(simplified.len());
    let mut prev: Option<char> = None;
    let mut run_len: usize = 0;

    for ch in simplified.chars() {
        if Some(ch) == prev {
            run_len = run_len.saturating_add(1);
            if run_len == 2 {
                out.push(ch);
            } else if run_len == 3 {
                // Remove the second char so the whole run becomes length 1.
                let _ = out.pop();
            } else {
                // >=4: skip
            }
        } else {
            prev = Some(ch);
            run_len = 1;
            out.push(ch);
        }
    }

    out.trim().to_string()
}

pub fn char_ngram_jaccard(a: &str, b: &str, n: usize) -> f64 {
    let a_norm = normalize_arabic(a).replace(' ', "");
    let b_norm = normalize_arabic(b).replace(' ', "");
    if a_norm.chars().count() < n || b_norm.chars().count() < n {
        return 0.0;
    }

    fn trigrams(s: &str, n: usize) -> Vec<u64> {
        let chars: Vec<u32> = s.chars().map(|c| c as u32).collect();
        if chars.len() < n {
            return vec![];
        }
        let mut keys = Vec::with_capacity(chars.len().saturating_sub(n) + 1);
        for i in 0..=chars.len() - n {
            let mut key: u64 = 0;
            for j in 0..n {
                key = (key << 21) | (chars[i + j] as u64);
            }
            keys.push(key);
        }
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    let a_set = trigrams(&a_norm, n);
    if a_set.is_empty() {
        return 0.0;
    }
    let b_set = trigrams(&b_norm, n);
    if b_set.is_empty() {
        return 0.0;
    }

    let mut i = 0usize;
    let mut j = 0usize;
    let mut inter = 0usize;
    while i < a_set.len() && j < b_set.len() {
        let av = a_set[i];
        let bv = b_set[j];
        if av == bv {
            inter += 1;
            i += 1;
            j += 1;
        } else if av < bv {
            i += 1;
        } else {
            j += 1;
        }
    }
    let union = a_set.len() + b_set.len() - inter;
    if union == 0 {
        return 0.0;
    }
    inter as f64 / union as f64
}

pub fn simhash64_from_char_ngrams(text: &str) -> u64 {
    let s = normalize_arabic(text).replace(' ', "");
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        return 0;
    }
    let mut v = [0i32; 64];
    for i in 0..=chars.len() - 3 {
        let gram: String = chars[i..i + 3].iter().collect();
        let mut hasher = Blake2bVar::new(8).expect("blake2b init");
        hasher.update(gram.as_bytes());
        let mut out = [0u8; 8];
        hasher.finalize_variable(&mut out).expect("blake2b finalize");
        let h = u64::from_le_bytes(out);
        for b in 0..64 {
            if ((h >> b) & 1) == 1 {
                v[b] += 1;
            } else {
                v[b] -= 1;
            }
        }
    }
    let mut out = 0u64;
    for (b, score) in v.iter().enumerate() {
        if *score >= 0 {
            out |= 1u64 << b;
        }
    }
    out
}

pub fn hamming64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}
