//! Load a Rasa model that was trained by the Python stack.
//!
//! The archive is a Python graph. This crate embeds that interpreter, trains
//! with `rasa.model_training.train`, loads the archive with `Agent.load`, and
//! serves `handle_text` over HTTP.

mod server;

pub use server::serve;

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// MongoDB conversation store. Matches Rasa's `tracker_store` type `mongod`.
#[derive(Clone, Debug)]
pub struct MongoTrackerConfig {
    pub url: String,
    pub db: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub auth_source: String,
    pub collection: String,
}

/// Tracker used by `serve`. Absent settings keep Rasa's in-memory store.
#[derive(Clone, Debug)]
pub enum TrackerStoreConfig {
    Memory,
    Mongo(MongoTrackerConfig),
    /// Rasa endpoints file. Uses `tracker_store` when its type is `mongod`.
    Endpoints(PathBuf),
}

/// A loaded Rasa `Agent`.
pub struct RasaModel {
    agent: Py<PyAny>,
    tracker_store: String,
}

impl RasaModel {
    /// Train with Python Rasa and return the path of the written archive.
    pub fn train(
        domain: &Path,
        config: &Path,
        training_files: &Path,
        output: &Path,
        fixed_model_name: Option<&str>,
    ) -> Result<PathBuf> {
        Python::with_gil(|py| {
            prepare_interpreter(py)?;
            let training = py
                .import("rasa.model_training")
                .map_err(|err| py_error(py, err))?;
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("domain", path_str(domain)?)
                .map_err(|err| py_error(py, err))?;
            kwargs
                .set_item("config", path_str(config)?)
                .map_err(|err| py_error(py, err))?;
            kwargs
                .set_item("training_files", path_str(training_files)?)
                .map_err(|err| py_error(py, err))?;
            kwargs
                .set_item("output", path_str(output)?)
                .map_err(|err| py_error(py, err))?;
            kwargs
                .set_item("fixed_model_name", fixed_model_name)
                .map_err(|err| py_error(py, err))?;
            let result = training
                .getattr("train")
                .map_err(|err| py_error(py, err))?
                .call((), Some(&kwargs))
                .map_err(|err| py_error(py, err))?;
            let model = result
                .getattr("model")
                .map_err(|err| py_error(py, err))?;
            let model_path: Option<String> = model.extract().map_err(|err| py_error(py, err))?;
            model_path
                .map(PathBuf::from)
                .ok_or_else(|| anyhow!("Python training finished without a model path"))
        })
    }

    /// Load an archive produced by Python `rasa train` / `Agent.load`.
    pub fn load(model_path: &Path) -> Result<Self> {
        Self::load_with_tracker(model_path, &TrackerStoreConfig::Memory)
    }

    /// Load an archive and attach an optional Mongo tracker store.
    pub fn load_with_tracker(model_path: &Path, tracker: &TrackerStoreConfig) -> Result<Self> {
        Python::with_gil(|py| {
            prepare_interpreter(py)?;
            let mongo = match tracker {
                TrackerStoreConfig::Memory => None,
                TrackerStoreConfig::Mongo(config) => Some(config.clone()),
                TrackerStoreConfig::Endpoints(path) => mongo_config_from_endpoints(py, path)?,
            };
            let module = py
                .import("rasa.core.agent")
                .map_err(|err| py_error(py, err))?;
            let kwargs = PyDict::new(py);
            if let Some(config) = mongo.as_ref() {
                let store = mongo_tracker_store(py, config)?;
                kwargs
                    .set_item("tracker_store", store)
                    .map_err(|err| py_error(py, err))?;
            }
            let agent = module
                .getattr("Agent")
                .map_err(|err| py_error(py, err))?
                .call_method("load", (path_str(model_path)?,), Some(&kwargs))
                .map_err(|err| py_error(py, err))?;
            let ready: bool = agent
                .call_method0("is_ready")
                .map_err(|err| py_error(py, err))?
                .extract()
                .map_err(|err| py_error(py, err))?;
            if !ready {
                return Err(anyhow!(
                    "Agent.load finished but the model at {} is not ready",
                    model_path.display()
                ));
            }
            let tracker_store = tracker_store_name(py, &agent)?;
            Ok(Self {
                agent: agent.unbind(),
                tracker_store,
            })
        })
    }

    /// Name of the tracker store attached to the agent.
    pub fn tracker_store(&self) -> &str {
        &self.tracker_store
    }

    /// Run one user message and return the bot messages as JSON.
    pub fn handle_text(&self, text: &str, sender_id: &str) -> Result<serde_json::Value> {
        Python::with_gil(|py| {
            let asyncio = py.import("asyncio").map_err(|err| py_error(py, err))?;
            let agent = self.agent.bind(py);
            let messages = agent
                .call_method1("handle_text", (text, py.None(), sender_id))
                .map_err(|err| py_error(py, err))?;
            let finished = asyncio
                .call_method1("run", (messages,))
                .map_err(|err| py_error(py, err))?;
            to_json(py, &finished)
        })
    }
}

fn prepare_interpreter(py: Python<'_>) -> Result<()> {
    let site_packages = python_site_packages(py)?;
    let locals = PyDict::new(py);
    locals
        .set_item(
            "venv_site",
            site_packages
                .to_str()
                .ok_or_else(|| anyhow!("site-packages path is not UTF-8"))?,
        )
        .map_err(|err| py_error(py, err))?;
    // The embedded interpreter starts as the base conda/system prefix, so its
    // site-packages hide the virtualenv. Keep the stdlib and use only the venv.
    py.run(
        c"
import site, sys
sys.path[:] = [
    entry
    for entry in sys.path
    if not (entry.endswith('site-packages') or entry.endswith('dist-packages'))
]
site.addsitedir(venv_site)
if venv_site in sys.path:
    sys.path.remove(venv_site)
sys.path.insert(0, venv_site)
",
        None,
        Some(&locals),
    )
    .map_err(|err| py_error(py, err))?;
    Ok(())
}

fn python_site_packages(py: Python<'_>) -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("RASA_PYTHON_SITE") {
        return Ok(PathBuf::from(explicit));
    }
    let version = py.version_info();
    let lib = format!("python{}.{}", version.major, version.minor);
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let path = PathBuf::from(venv).join("lib").join(&lib).join("site-packages");
        if path.is_dir() {
            return Ok(path);
        }
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = repo
        .join(".venv")
        .join("lib")
        .join(lib)
        .join("site-packages");
    let path = path.canonicalize().with_context(|| {
        format!(
            "Rasa's virtualenv site-packages is missing at {}. Set RASA_PYTHON_SITE or VIRTUAL_ENV.",
            path.display()
        )
    })?;
    Ok(path)
}

fn mongo_config_from_endpoints(
    py: Python<'_>,
    path: &Path,
) -> Result<Option<MongoTrackerConfig>> {
    if !path.is_file() {
        return Err(anyhow!(
            "endpoints file does not exist: {}",
            path.display()
        ));
    }
    let locals = PyDict::new(py);
    locals
        .set_item("endpoints_path", path_str(path)?)
        .map_err(|err| py_error(py, err))?;
    py.run(
        c"
from rasa.utils.endpoints import read_endpoint_config

cfg = read_endpoint_config(endpoints_path, 'tracker_store')
if cfg is None:
    parsed = None
else:
    store_type = (cfg.type or '').lower()
    if store_type != 'mongod':
        raise ValueError(
            'endpoints tracker_store type %r is not supported; use mongod' % (cfg.type,)
        )
    parsed = {
        'url': cfg.url or 'mongodb://localhost:27017',
        'db': cfg.kwargs.get('db') or 'rasa',
        'username': cfg.kwargs.get('username'),
        'password': cfg.kwargs.get('password'),
        'auth_source': cfg.kwargs.get('auth_source') or 'admin',
        'collection': cfg.kwargs.get('collection') or 'conversations',
    }
",
        None,
        Some(&locals),
    )
    .map_err(|err| py_error(py, err))?;
    let parsed = locals
        .get_item("parsed")
        .map_err(|err| py_error(py, err))?
        .ok_or_else(|| anyhow!("endpoints parser did not return a tracker config"))?;
    if parsed.is_none() {
        return Ok(None);
    }
    Ok(Some(mongo_config_from_dict(&parsed)?))
}

fn mongo_config_from_dict(parsed: &Bound<'_, PyAny>) -> Result<MongoTrackerConfig> {
    Ok(MongoTrackerConfig {
        url: dict_string(parsed, "url")?,
        db: dict_string(parsed, "db")?,
        username: dict_optional_string(parsed, "username")?,
        password: dict_optional_string(parsed, "password")?,
        auth_source: dict_string(parsed, "auth_source")?,
        collection: dict_string(parsed, "collection")?,
    })
}

fn dict_string(parsed: &Bound<'_, PyAny>, key: &str) -> Result<String> {
    parsed
        .get_item(key)
        .map_err(|err| anyhow!("{key}: {err}"))?
        .extract()
        .map_err(|err| anyhow!("{key}: {err}"))
}

fn dict_optional_string(parsed: &Bound<'_, PyAny>, key: &str) -> Result<Option<String>> {
    let value = parsed.get_item(key).map_err(|err| anyhow!(err))?;
    if value.is_none() {
        return Ok(None);
    }
    value
        .extract()
        .map(Some)
        .map_err(|err| anyhow!("{key}: {err}"))
}

fn mongo_tracker_store<'py>(
    py: Python<'py>,
    config: &MongoTrackerConfig,
) -> Result<Bound<'py, PyAny>> {
    let domain = py
        .import("rasa.shared.core.domain")
        .map_err(|err| py_error(py, err))?
        .getattr("Domain")
        .map_err(|err| py_error(py, err))?
        .call_method0("empty")
        .map_err(|err| py_error(py, err))?;
    let kwargs = PyDict::new(py);
    kwargs
        .set_item("domain", domain)
        .map_err(|err| py_error(py, err))?;
    kwargs
        .set_item("host", &config.url)
        .map_err(|err| py_error(py, err))?;
    kwargs
        .set_item("db", &config.db)
        .map_err(|err| py_error(py, err))?;
    kwargs
        .set_item("username", config.username.as_deref())
        .map_err(|err| py_error(py, err))?;
    kwargs
        .set_item("password", config.password.as_deref())
        .map_err(|err| py_error(py, err))?;
    kwargs
        .set_item("auth_source", &config.auth_source)
        .map_err(|err| py_error(py, err))?;
    kwargs
        .set_item("collection", &config.collection)
        .map_err(|err| py_error(py, err))?;
    py.import("rasa.core.tracker_store")
        .map_err(|err| py_error(py, err))?
        .getattr("MongoTrackerStore")
        .map_err(|err| py_error(py, err))?
        .call((), Some(&kwargs))
        .map_err(|err| py_error(py, err))
}

fn tracker_store_name(py: Python<'_>, agent: &Bound<'_, PyAny>) -> Result<String> {
    let store = agent
        .getattr("tracker_store")
        .map_err(|err| py_error(py, err))?;
    let inner = if store.hasattr("_tracker_store").unwrap_or(false) {
        store
            .getattr("_tracker_store")
            .map_err(|err| py_error(py, err))?
    } else {
        store
    };
    inner
        .get_type()
        .name()
        .map(|name| name.to_string())
        .map_err(|err| py_error(py, err))
}

fn path_str(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", path.display()))
}

fn to_json(py: Python<'_>, value: &Bound<'_, PyAny>) -> Result<serde_json::Value> {
    if value.is_none() {
        return Ok(serde_json::Value::Array(Vec::new()));
    }
    let json = py.import("json").map_err(|err| py_error(py, err))?;
    let dumped: String = json
        .call_method1("dumps", (value,))
        .map_err(|err| py_error(py, err))?
        .extract()
        .map_err(|err| py_error(py, err))?;
    serde_json::from_str(&dumped).context("Rasa response was not valid JSON")
}

fn py_error(py: Python<'_>, err: PyErr) -> anyhow::Error {
    let traceback = err.traceback(py).and_then(|tb| tb.format().ok());
    let message = err.to_string();
    match traceback {
        Some(tb) => anyhow!("{message}\n{tb}"),
        None => anyhow!(message),
    }
}
