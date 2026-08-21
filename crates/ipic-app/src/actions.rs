//! Desktop file actions: open, reveal, copy path, rename, duplicate,
//! new folder, move to trash.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn open_file(path: &Path) {
    #[cfg(target_os = "macos")]
    let program = "open";
    #[cfg(not(target_os = "macos"))]
    let program = "xdg-open";
    let _ = Command::new(program).arg(path).spawn();
}

pub fn reveal_in_file_manager(path: &Path) {
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg("-R").arg(path).spawn();
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Some(parent) = path.parent() {
            let _ = Command::new("xdg-open").arg(parent).spawn();
        }
    }
}

pub fn copy_path_to_clipboard(path: &str) {
    #[cfg(target_os = "macos")]
    let program = "pbcopy";
    #[cfg(not(target_os = "macos"))]
    let program = "xclip";
    let _ = Command::new(program)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map(|mut child| {
            use std::io::Write;
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(path.as_bytes());
            }
        });
}

pub fn rename_file(old_path: &Path, new_name: &str) -> Result<PathBuf> {
    if new_name.trim().is_empty() || new_name.contains('/') {
        return Err(anyhow!("invalid name"));
    }
    let new_path = old_path.parent().unwrap_or(old_path).join(new_name);
    if new_path == old_path {
        return Ok(new_path);
    }
    std::fs::rename(old_path, &new_path)?;
    Ok(new_path)
}

pub fn duplicate_file(path: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| anyhow!("no file name"))?
        .to_string_lossy()
        .into_owned();
    let stem = path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default();
    let extension = path
        .extension()
        .map(|extension| format!(".{}", extension.to_string_lossy()))
        .unwrap_or_default();
    for attempt in 1..100 {
        let candidate_name = if attempt == 1 {
            format!("{stem} copy{extension}")
        } else {
            format!("{stem} copy {attempt}{extension}")
        };
        let candidate = path.parent().unwrap_or(path).join(&candidate_name);
        if !candidate.exists() {
            std::fs::copy(path, &candidate)?;
            return Ok(candidate);
        }
    }
    Err(anyhow!("could not find a free copy name for {name}"))
}

pub fn create_folder(parent: &Path, name: &str) -> Result<PathBuf> {
    if name.trim().is_empty() || name.contains('/') {
        return Err(anyhow!("invalid folder name"));
    }
    let folder = parent.join(name);
    std::fs::create_dir_all(&folder)?;
    Ok(folder)
}

pub fn move_to_trash(path: &Path) -> Result<()> {
    Ok(trash::delete(path)?)
}
