use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;

#[derive(Debug, Clone)]
pub struct NpyHeader {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

fn parse_header_dict(header: &str) -> Result<NpyHeader> {
    // Header is a Python dict literal like:
    // "{'descr': '<u4', 'fortran_order': False, 'shape': (77429,), }\\n"
    let descr = {
        let key = "'descr'";
        let start = header.find(key).ok_or_else(|| anyhow!("missing descr"))?;
        let rest = &header[start + key.len()..];
        let q1 = rest.find('\'').ok_or_else(|| anyhow!("invalid descr"))?;
        let rest2 = &rest[q1 + 1..];
        let q2 = rest2.find('\'').ok_or_else(|| anyhow!("invalid descr"))?;
        rest2[..q2].to_string()
    };

    let fortran_order = {
        let key = "'fortran_order'";
        let start = header.find(key).ok_or_else(|| anyhow!("missing fortran_order"))?;
        let rest = &header[start + key.len()..];
        let colon = rest.find(':').ok_or_else(|| anyhow!("invalid fortran_order"))?;
        let rest2 = rest[colon + 1..].trim_start();
        if rest2.starts_with("True") {
            true
        } else if rest2.starts_with("False") {
            false
        } else {
            return Err(anyhow!("invalid fortran_order value"));
        }
    };

    let shape = {
        let key = "'shape'";
        let start = header.find(key).ok_or_else(|| anyhow!("missing shape"))?;
        let rest = &header[start + key.len()..];
        let lpar = rest.find('(').ok_or_else(|| anyhow!("invalid shape"))?;
        let rest2 = &rest[lpar + 1..];
        let rpar = rest2.find(')').ok_or_else(|| anyhow!("invalid shape"))?;
        let inside = rest2[..rpar].trim();
        if inside.is_empty() {
            vec![]
        } else {
            inside
                .split(',')
                .filter_map(|p| {
                    let t = p.trim();
                    if t.is_empty() {
                        None
                    } else {
                        Some(
                            t.parse::<usize>()
                                .with_context(|| format!("invalid shape element: {t}")),
                        )
                    }
                })
                .collect::<Result<Vec<_>>>()?
        }
    };

    Ok(NpyHeader {
        descr,
        fortran_order,
        shape,
    })
}

fn product(shape: &[usize]) -> usize {
    shape.iter().copied().fold(1usize, |a, b| a.saturating_mul(b))
}

#[derive(Debug)]
pub struct NpyMemmap<T> {
    mmap: Mmap,
    data_offset: usize,
    len: usize,
    shape: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NpyMemmap<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T: Copy> NpyMemmap<T> {
    pub fn as_slice(&self) -> &[T] {
        let data = &self.mmap[self.data_offset..];
        let ptr = data.as_ptr() as *const T;
        let len = self.len;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

pub fn load_npy_memmap<T: Copy>(
    path: &Path,
    expected_descr: &str,
    expected_ndim: usize,
) -> Result<NpyMemmap<T>> {
    let f = File::open(path).with_context(|| format!("open npy: {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&f)? };

    let bytes = &mmap[..];
    if bytes.len() < 10 {
        return Err(anyhow!("npy too small"));
    }
    if &bytes[..6] != b"\x93NUMPY" {
        return Err(anyhow!("invalid npy magic"));
    }
    let major = bytes[6];
    let minor = bytes[7];

    let (header_len, header_start) = match (major, minor) {
        (1, 0) => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        (2, 0) => {
            if bytes.len() < 12 {
                return Err(anyhow!("invalid npy v2 header"));
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => return Err(anyhow!("unsupported npy version {major}.{minor}")),
    };

    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err(anyhow!("truncated npy header"));
    }

    let header_str = std::str::from_utf8(&bytes[header_start..header_end])
        .with_context(|| format!("invalid npy header utf8: {}", path.display()))?;
    let header = parse_header_dict(header_str)?;

    if header.fortran_order {
        return Err(anyhow!("fortran_order=True is not supported"));
    }
    if header.descr != expected_descr {
        return Err(anyhow!(
            "dtype mismatch for {}: expected {}, got {}",
            path.display(),
            expected_descr,
            header.descr
        ));
    }
    if header.shape.len() != expected_ndim {
        return Err(anyhow!(
            "ndim mismatch for {}: expected {}, got {}",
            path.display(),
            expected_ndim,
            header.shape.len()
        ));
    }

    let len = product(&header.shape);
    let data_offset = header_end;

    let align = std::mem::align_of::<T>();
    if data_offset % align != 0 {
        return Err(anyhow!(
            "misaligned npy data for {} (offset {} not multiple of {})",
            path.display(),
            data_offset,
            align
        ));
    }
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| anyhow!("npy size overflow"))?;
    if bytes.len() < data_offset + byte_len {
        return Err(anyhow!("npy data truncated"));
    }

    Ok(NpyMemmap {
        mmap,
        data_offset,
        len,
        shape: header.shape,
        _phantom: std::marker::PhantomData,
    })
}

