use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use rasa_model::{MongoTrackerConfig, RasaModel, TrackerStoreConfig};

#[derive(Parser)]
#[command(
    name = "rasa-model",
    about = "Train, load, and serve a Python Rasa model from Rust"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a model with Python Rasa (`rasa.model_training.train`).
    Train {
        #[arg(long)]
        domain: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value = "models")]
        out: PathBuf,
        #[arg(long)]
        name: Option<String>,
    },
    /// Load a model archive with `Agent.load` and optionally handle one message.
    Load {
        model: PathBuf,
        #[arg(long)]
        text: Option<String>,
        #[arg(long, default_value = "rust")]
        sender: String,
    },
    /// Serve `handle_text` on Rasa's REST webhook.
    Serve {
        model: PathBuf,
        /// Network interface to bind. Same default as `rasa run`.
        #[arg(long, default_value = "0.0.0.0")]
        interface: String,
        #[arg(long, default_value_t = 5005)]
        port: u16,
        /// Rasa endpoints file. A `tracker_store` of type `mongod` is used when set.
        #[arg(long)]
        endpoints: Option<PathBuf>,
        /// MongoDB URL. Enables `MongoTrackerStore`. Example: mongodb://localhost:27017
        #[arg(long, env = "RASA_MONGO_URL")]
        mongo_url: Option<String>,
        #[arg(long, default_value = "rasa", env = "RASA_MONGO_DB")]
        mongo_db: String,
        #[arg(long, env = "RASA_MONGO_USERNAME")]
        mongo_username: Option<String>,
        #[arg(long, env = "RASA_MONGO_PASSWORD")]
        mongo_password: Option<String>,
        #[arg(long, default_value = "admin", env = "RASA_MONGO_AUTH_SOURCE")]
        mongo_auth_source: String,
        #[arg(long, default_value = "conversations", env = "RASA_MONGO_COLLECTION")]
        mongo_collection: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train {
            domain,
            config,
            data,
            out,
            name,
        } => {
            let model = RasaModel::train(&domain, &config, &data, &out, name.as_deref())?;
            println!("{}", model.display());
        }
        Command::Load { model, text, sender } => {
            let agent = RasaModel::load(&model)?;
            println!("loaded {}", model.display());
            if let Some(text) = text {
                let messages = agent.handle_text(&text, &sender)?;
                println!("{}", serde_json::to_string_pretty(&messages)?);
            }
        }
        Command::Serve {
            model,
            interface,
            port,
            endpoints,
            mongo_url,
            mongo_db,
            mongo_username,
            mongo_password,
            mongo_auth_source,
            mongo_collection,
        } => {
            let tracker = serve_tracker(
                endpoints,
                mongo_url,
                mongo_db,
                mongo_username,
                mongo_password,
                mongo_auth_source,
                mongo_collection,
            )?;
            let agent = RasaModel::load_with_tracker(&model, &tracker)?;
            println!("loaded {}", model.display());
            println!("tracker store: {}", agent.tracker_store());
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(rasa_model::serve(agent, &interface, port))?;
        }
    }
    Ok(())
}

fn serve_tracker(
    endpoints: Option<PathBuf>,
    mongo_url: Option<String>,
    mongo_db: String,
    mongo_username: Option<String>,
    mongo_password: Option<String>,
    mongo_auth_source: String,
    mongo_collection: String,
) -> Result<TrackerStoreConfig> {
    if endpoints.is_some() && mongo_url.is_some() {
        return Err(anyhow!(
            "Pass either --endpoints or --mongo-url, not both."
        ));
    }
    if let Some(url) = mongo_url {
        return Ok(TrackerStoreConfig::Mongo(MongoTrackerConfig {
            url,
            db: mongo_db,
            username: mongo_username,
            password: mongo_password,
            auth_source: mongo_auth_source,
            collection: mongo_collection,
        }));
    }
    if let Some(path) = endpoints {
        return Ok(TrackerStoreConfig::Endpoints(path));
    }
    Ok(TrackerStoreConfig::Memory)
}
