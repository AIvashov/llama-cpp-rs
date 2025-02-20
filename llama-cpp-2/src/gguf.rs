//! merge gguf files
use std::path::PathBuf;
use std::{ptr, error::Error};
use std::ffi::{CString, CStr};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::os::raw::c_char;

const GGUF_DEFAULT_ALIGNMENT: usize = 16;
const LLM_KV_SPLIT_COUNT: *const c_char = b"split.count\0".as_ptr() as *const c_char;

fn ggml_pad(n_bytes: usize, alignment: usize) -> usize {
    ((n_bytes + alignment - 1) / alignment) * alignment
}

fn write_zeros<W: Write>(writer: &mut W, count: usize) -> std::io::Result<()> {
    let zeros = vec![0u8; count];
    writer.write_all(&zeros)
}

/// merge gguf files
pub fn gguf_merge(parts: Vec<PathBuf>, output: PathBuf) -> Result<(), Box<dyn Error>> {
    println!("gguf_merge: merging {} parts into {:?}", parts.len(), output);

    if output.exists() {
        return Err(format!("Output file {:?} already exists", output).into());
    }

    let mut fout = OpenOptions::new().write(true).create_new(true).open(&output)?;

    let ctx_out = unsafe { llama_cpp_sys_2::gguf_init_empty() };
    if ctx_out.is_null() {
        return Err("Failed to initialize empty GGUF context".into());
    }

    let mut ctx_metas: Vec<*mut llama_cpp_sys_2::ggml_context> = Vec::with_capacity(parts.len());
    let mut ctx_ggufs: Vec<*mut llama_cpp_sys_2::gguf_context> = Vec::with_capacity(parts.len());
    let mut read_data: Vec<u8> = Vec::new();

    let n_split = parts.len();

    for (i, part) in parts.iter().enumerate() {
        let part_str = part.to_str().ok_or("Invalid path in parts")?;
        let c_part = CString::new(part_str)?;

        let mut ctx_meta: *mut llama_cpp_sys_2::ggml_context = ptr::null_mut();
        let params = llama_cpp_sys_2::gguf_init_params {
            no_alloc: true,
            ctx: &mut ctx_meta,
        };

        println!("Reading metadata from {} ...", part_str);

        let ctx_gguf = unsafe { llama_cpp_sys_2::gguf_init_from_file(c_part.as_ptr(), params) };
        if ctx_gguf.is_null() {
            return Err(format!("Failed to load input GGUF from {}", part_str).into());
        }
        ctx_ggufs.push(ctx_gguf);
        ctx_metas.push(ctx_meta);

        if i == 0 {
            let key_n_split = unsafe { llama_cpp_sys_2::gguf_find_key(ctx_gguf, LLM_KV_SPLIT_COUNT) };
            if key_n_split < 0 {
                unsafe {
                    llama_cpp_sys_2::gguf_free(ctx_gguf);
                    if !ctx_meta.is_null() { llama_cpp_sys_2::ggml_free(ctx_meta); }
                    llama_cpp_sys_2::gguf_free(ctx_out);
                }
                return Err(format!("Input file {} does not contain {} metadata", part_str, unsafe {
                    CStr::from_ptr(LLM_KV_SPLIT_COUNT).to_str()?
                }).into());
            }
            let meta_n_split = unsafe { llama_cpp_sys_2::gguf_get_val_u16(ctx_gguf, key_n_split) } as usize;
            if meta_n_split != n_split {
                println!("Warning: metadata split count {} does not match provided parts count {}", meta_n_split, n_split);
            }
            unsafe { llama_cpp_sys_2::gguf_set_val_u16(ctx_gguf, LLM_KV_SPLIT_COUNT, 0); }
            unsafe { llama_cpp_sys_2::gguf_set_kv(ctx_out, ctx_gguf); }
        }

        let n_tensors = unsafe { llama_cpp_sys_2::gguf_get_n_tensors(ctx_gguf) } as usize;
        for i_tensor in 0..n_tensors {
            let t_name_ptr = unsafe { llama_cpp_sys_2::gguf_get_tensor_name(ctx_gguf, i_tensor as i64) };
            if t_name_ptr.is_null() { continue; }
            let tensor = unsafe { llama_cpp_sys_2::ggml_get_tensor(ctx_meta, t_name_ptr) };
            unsafe { llama_cpp_sys_2::gguf_add_tensor(ctx_out, tensor); }
        }
    }

    let meta_size = unsafe { llama_cpp_sys_2::gguf_get_meta_size(ctx_out) };
    write_zeros(&mut fout, meta_size)?;

    for (i, part) in parts.iter().enumerate() {
        let part_str = part.to_str().ok_or("Invalid path in parts")?;
        println!("Writing tensors from {} ...", part_str);

        let mut fin = File::open(part_str)?;
        let ctx_gguf = ctx_ggufs[i];
        let ctx_meta = ctx_metas[i];
        let n_tensors = unsafe { llama_cpp_sys_2::gguf_get_n_tensors(ctx_gguf) } as usize;

        for i_tensor in 0..n_tensors {
            let t_name_ptr = unsafe { llama_cpp_sys_2::gguf_get_tensor_name(ctx_gguf, i_tensor as i64) };
            if t_name_ptr.is_null() { continue; }
            let tensor = unsafe { llama_cpp_sys_2::ggml_get_tensor(ctx_meta, t_name_ptr) };
            let n_bytes = unsafe { llama_cpp_sys_2::ggml_nbytes(tensor) };

            if read_data.len() < n_bytes {
                read_data.resize(n_bytes, 0);
            }

            let offset = unsafe { llama_cpp_sys_2::gguf_get_data_offset(ctx_gguf) }
                + unsafe { llama_cpp_sys_2::gguf_get_tensor_offset(ctx_gguf, i_tensor as i64) };
            fin.seek(SeekFrom::Start(offset as u64))?;
            fin.read_exact(&mut read_data[..n_bytes])?;
            fout.write_all(&read_data[..n_bytes])?;

            let padded = ggml_pad(n_bytes, GGUF_DEFAULT_ALIGNMENT);
            if padded > n_bytes {
                write_zeros(&mut fout, padded - n_bytes)?;
            }
        }

        unsafe {
            llama_cpp_sys_2::gguf_free(ctx_gguf);
            if !ctx_meta.is_null() { llama_cpp_sys_2::ggml_free(ctx_meta); }
        }
    }

    fout.seek(SeekFrom::Start(0))?;
    let meta_size = unsafe { llama_cpp_sys_2::gguf_get_meta_size(ctx_out) };
    let mut meta_data = vec![0u8; meta_size];
    unsafe { llama_cpp_sys_2::gguf_get_meta_data(ctx_out, meta_data.as_mut_ptr() as *mut std::os::raw::c_void); }
    fout.write_all(&meta_data)?;
    fout.flush()?;
    unsafe { llama_cpp_sys_2::gguf_free(ctx_out); }

    println!("gguf_merge: merged {} parts into output {:?}", n_split, output);
    Ok(())
}