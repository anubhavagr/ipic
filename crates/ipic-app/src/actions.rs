//! Desktop file actions: open, reveal, copy path, rename, move to trash.

use anyhow::Result;
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
        let _ = Command::new("open").args(["-R"]).arg(path).spawn();
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Some(parent) = path.parent() {
            let _ = Command::new("xdg-open").arg(parent).spawn();
        }
    }
}

pub fn copy_path_to_clipboard(path: &str) {
    let _ = Command::new("pbcopy")
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
    let new_path = old_path
        .parent()
        .unwrap_or(old_path)
        .join(new_name);
    if new_path == old_path {
        return Ok(new_path);
    }
    std::fs::rename(old_path, &new_path)?;
    Ok(new_path)
}

pub fn move_to_trash(path: &Path) -> Result<()> {
    Ok(trash::delete(path)?)
}
