use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-changed=build.rs");

    let python = python_bin();
    let output = Command::new(&python)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
        ])
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", python.display()));
    if !output.status.success() {
        panic!(
            "failed to read Python LIBDIR from {}: {}",
            python.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let libdir = String::from_utf8(output.stdout).expect("LIBDIR is not UTF-8");
    let libdir = libdir.trim();
    if !libdir.is_empty() {
        // Prefer this interpreter's libpython over a distro copy on the default path.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }
}

fn python_bin() -> PathBuf {
    if let Ok(python) = std::env::var("PYO3_PYTHON") {
        return PathBuf::from(python);
    }
    let venv = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    if venv.exists() {
        return venv;
    }
    PathBuf::from("python3")
}
