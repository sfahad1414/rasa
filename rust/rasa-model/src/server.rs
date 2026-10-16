//! HTTP API for `Agent.handle_text`.
//!
//! `POST /webhooks/rest/webhook` accepts `{"sender", "message"}` and returns
//! the bot messages, matching Rasa's REST channel.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::RasaModel;

/// Load is done by the caller. This binds the REST webhook and serves until stopped.
pub async fn serve(model: RasaModel, interface: &str, port: u16) -> anyhow::Result<()> {
    let state = Arc::new(model);
    let app = Router::new()
        .route("/health", get(health))
        .route("/webhooks/rest", get(health))
        .route("/webhooks/rest/", get(health))
        .route("/webhooks/rest/webhook", post(webhook))
        .with_state(state);

    let addr: SocketAddr = format!("{interface}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    println!("listening on http://{bound}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health(State(model): State<Arc<RasaModel>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "tracker_store": model.tracker_store(),
    }))
}

#[derive(Deserialize)]
struct WebhookRequest {
    sender: Option<String>,
    message: Option<String>,
}

async fn webhook(
    State(model): State<Arc<RasaModel>>,
    Json(body): Json<WebhookRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let message = body.message.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "message is required"})),
        )
    })?;
    let sender = body.sender.unwrap_or_else(|| "default".to_string());
    let model = Arc::clone(&model);

    let messages = tokio::task::spawn_blocking(move || model.handle_text(&message, &sender))
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": err.to_string()})),
            )
        })?
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": err.to_string()})),
            )
        })?;
    Ok(Json(messages))
}
