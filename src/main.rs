use base64::Engine;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};

const GROK_VIDEO_SUBMIT_URL: &str = "https://api.x.ai/v1/videos/generations";
const GROK_VIDEO_POLL_URL: &str = "https://api.x.ai/v1/videos";

const VALID_ASPECT_RATIOS: &[&str] = &["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"];

const VALID_RESOLUTIONS: &[&str] = &["480p", "720p"];

const DEFAULT_MODEL: &str = "grok-imagine-video";
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;
const DEFAULT_POLL_TIMEOUT_SECS: u64 = 600; // 10 minutes
const MAX_TRANSIENT_RETRIES: u32 = 5;
const HTTP_TIMEOUT_SECS: u64 = 300; // 5 minutes for video downloads

// --- Config ---

#[derive(Deserialize)]
struct Config {
    api_key: String,
    #[serde(default = "default_save_dir")]
    save_dir: String,
    #[serde(default = "default_poll_interval")]
    poll_interval_secs: u64,
    #[serde(default = "default_poll_timeout")]
    poll_timeout_secs: u64,
}

fn default_save_dir() -> String {
    "/tmp/grok-videos".to_string()
}

fn default_poll_interval() -> u64 {
    DEFAULT_POLL_INTERVAL_SECS
}

fn default_poll_timeout() -> u64 {
    DEFAULT_POLL_TIMEOUT_SECS
}

/// Try to read just the api_key from grok-image's config as a fallback.
fn read_grok_image_api_key() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home)
        .join(".config")
        .join("mcp-server-grok-image")
        .join("config.toml");
    let content = std::fs::read_to_string(&path).ok()?;
    let table: toml::Table = toml::from_str(&content).ok()?;
    table.get("api_key")?.as_str().map(|s| s.to_string())
}

fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let path = PathBuf::from(&home)
        .join(".config")
        .join("mcp-server-grok-video")
        .join("config.toml");

    // 1. Try own config file
    if let Ok(content) = std::fs::read_to_string(&path) {
        let config: Config = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
        info!(path = %path.display(), "Config loaded from file");
        return Ok(config);
    }

    // 2. Try XAI_API_KEY env var
    if let Ok(api_key) = std::env::var("XAI_API_KEY") {
        info!("Config loaded from XAI_API_KEY environment variable");
        return Ok(Config {
            api_key,
            save_dir: default_save_dir(),
            poll_interval_secs: default_poll_interval(),
            poll_timeout_secs: default_poll_timeout(),
        });
    }

    // 3. Fall back to grok-image's config for the API key
    if let Some(api_key) = read_grok_image_api_key() {
        info!("Config loaded: api_key from mcp-server-grok-image config");
        return Ok(Config {
            api_key,
            save_dir: default_save_dir(),
            poll_interval_secs: default_poll_interval(),
            poll_timeout_secs: default_poll_timeout(),
        });
    }

    Err(format!(
        "No config found. Either:\n\
         1. Create {} with:\n\
         \n\
         api_key = \"xai-...\"\n\
         save_dir = \"/tmp/grok-videos\"  # optional\n\
         \n\
         2. Or set the XAI_API_KEY environment variable.\n\
         3. Or configure mcp-server-grok-image (its api_key will be reused).",
        path.display()
    )
    .into())
}

// --- Grok Video API types ---

#[derive(Serialize)]
struct VideoGenerationRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    video_url: Option<String>,
}

/// Response from POST /v1/videos/generations
#[derive(Deserialize)]
struct VideoSubmitResponse {
    request_id: String,
}

/// Response from GET /v1/videos/{request_id}
#[derive(Deserialize)]
#[allow(dead_code)]
struct VideoPollResponse {
    status: String,
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    video: Option<VideoData>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct VideoData {
    url: String,
    #[serde(default)]
    duration: Option<f64>,
}

// --- MCP tool parameter types ---

#[derive(Debug, Deserialize, JsonSchema)]
struct GenerateVideoParams {
    #[schemars(description = "Text description of the desired video scene")]
    prompt: String,
    #[schemars(description = "Video duration in seconds (1-15, default chosen by API)")]
    duration: Option<u8>,
    #[schemars(
        description = "Aspect ratio. Options: 1:1, 16:9 (default), 9:16, 4:3, 3:4, 3:2, 2:3"
    )]
    aspect_ratio: Option<String>,
    #[schemars(description = "Output resolution. Options: \"480p\" (default), \"720p\"")]
    resolution: Option<String>,
    #[schemars(description = "Model to use (default: \"grok-imagine-video\")")]
    model: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AnimateImageParams {
    #[schemars(
        description = "URL or base64 data URI of the source image to animate. Also accepts a local file path."
    )]
    image_url: String,
    #[schemars(description = "Text description of the desired motion and atmosphere")]
    prompt: String,
    #[schemars(description = "Video duration in seconds (1-15, default chosen by API)")]
    duration: Option<u8>,
    #[schemars(
        description = "Aspect ratio. Options: 1:1, 16:9 (default), 9:16, 4:3, 3:4, 3:2, 2:3. Image-to-video may maintain input ratio."
    )]
    aspect_ratio: Option<String>,
    #[schemars(description = "Output resolution. Options: \"480p\" (default), \"720p\"")]
    resolution: Option<String>,
    #[schemars(description = "Model to use (default: \"grok-imagine-video\")")]
    model: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EditVideoParams {
    #[schemars(
        description = "URL or base64 data URI of the source video to edit. Also accepts a local file path."
    )]
    video_url: String,
    #[schemars(description = "Natural language edit instructions")]
    prompt: String,
    #[schemars(description = "Output resolution. Options: \"480p\" (default), \"720p\"")]
    resolution: Option<String>,
    #[schemars(description = "Model to use (default: \"grok-imagine-video\")")]
    model: Option<String>,
}

// --- Input validation ---

fn validate_duration(duration: Option<u8>) -> Result<(), String> {
    if let Some(d) = duration {
        if d < 1 || d > 15 {
            return Err(format!("duration must be between 1 and 15 seconds, got {d}"));
        }
    }
    Ok(())
}

fn validate_aspect_ratio(aspect_ratio: Option<&str>) -> Result<(), String> {
    if let Some(ar) = aspect_ratio {
        if !VALID_ASPECT_RATIOS.contains(&ar) {
            return Err(format!(
                "Invalid aspect_ratio \"{ar}\". Valid options: {}",
                VALID_ASPECT_RATIOS.join(", ")
            ));
        }
    }
    Ok(())
}

fn validate_resolution(resolution: Option<&str>) -> Result<(), String> {
    if let Some(res) = resolution {
        if !VALID_RESOLUTIONS.contains(&res) {
            return Err(format!(
                "Invalid resolution \"{res}\". Valid options: {}",
                VALID_RESOLUTIONS.join(", ")
            ));
        }
    }
    Ok(())
}

// --- Helpers ---

/// Read a local file and convert to a data: URI for the API.
async fn local_file_to_data_uri(path: &str) -> Result<String, String> {
    let data =
        tokio::fs::read(path).await.map_err(|e| format!("Failed to read file {path}: {e}"))?;

    let mime = if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        "image/jpeg"
    } else if path.ends_with(".gif") {
        "image/gif"
    } else if path.ends_with(".webp") {
        "image/webp"
    } else if path.ends_with(".mp4") {
        "video/mp4"
    } else if path.ends_with(".webm") {
        "video/webm"
    } else if path.ends_with(".mov") {
        "video/quicktime"
    } else {
        // Try magic bytes
        if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
            "image/png"
        } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            "image/jpeg"
        } else if data.starts_with(b"GIF") {
            "image/gif"
        } else if data.len() >= 12 && &data[8..12] == b"WEBP" {
            "image/webp"
        } else if data.len() >= 8 && &data[4..8] == b"ftyp" {
            "video/mp4"
        } else if data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
            "video/webm"
        } else {
            "application/octet-stream"
        }
    };

    let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
    info!(path, mime, "Encoded local file as data URI");
    Ok(format!("data:{mime};base64,{b64}"))
}

/// Resolve a URL parameter: if it looks like a local path, convert to data URI.
async fn resolve_url(url: &str) -> Result<String, String> {
    if url.starts_with("http://") || url.starts_with("https://") || url.starts_with("data:") {
        Ok(url.to_string())
    } else {
        local_file_to_data_uri(url).await
    }
}

/// Check if an HTTP status code is transient (worth retrying).
fn is_transient(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
        || status == reqwest::StatusCode::GATEWAY_TIMEOUT
        || status == reqwest::StatusCode::BAD_GATEWAY
}

// --- MCP Server ---

#[derive(Clone)]
pub struct GrokVideoServer {
    api_key: String,
    save_dir: PathBuf,
    http: reqwest::Client,
    counter: Arc<AtomicU64>,
    poll_interval: std::time::Duration,
    poll_timeout: std::time::Duration,
    tool_router: ToolRouter<Self>,
}

impl GrokVideoServer {
    /// Submit a video generation request and poll until completion.
    async fn submit_and_poll(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<VideoPollResponse, String> {
        debug!(
            model = request.model,
            prompt = request.prompt,
            "Submitting video generation request"
        );

        // Step 1: POST to submit the request
        let response = self
            .http
            .post(GROK_VIDEO_SUBMIT_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(request)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "failed to read response body".to_string());
            warn!(%status, body = body, "Grok Video API error");
            return Err(format!("Grok Video API error ({status}): {body}"));
        }

        let submit: VideoSubmitResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse submit response: {e}"))?;

        info!(request_id = submit.request_id, "Video generation submitted, polling for completion");

        // Step 2: Poll GET /v1/videos/{request_id} until done
        let poll_url = format!("{}/{}", GROK_VIDEO_POLL_URL, submit.request_id);
        let start = tokio::time::Instant::now();
        let deadline = start + self.poll_timeout;
        let mut transient_failures: u32 = 0;

        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(format!(
                    "Video generation timed out after {}s (request_id: {})",
                    self.poll_timeout.as_secs(),
                    submit.request_id
                ));
            }

            tokio::time::sleep(self.poll_interval).await;

            let elapsed = start.elapsed().as_secs();

            let poll_resp = match self
                .http
                .get(&poll_url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    transient_failures += 1;
                    if transient_failures >= MAX_TRANSIENT_RETRIES {
                        return Err(format!(
                            "Poll failed after {MAX_TRANSIENT_RETRIES} consecutive network errors: {e}"
                        ));
                    }
                    warn!(
                        error = %e,
                        attempt = transient_failures,
                        "Transient poll network error, retrying..."
                    );
                    continue;
                }
            };

            let poll_status = poll_resp.status();
            if !poll_status.is_success() {
                if is_transient(poll_status) {
                    transient_failures += 1;
                    if transient_failures >= MAX_TRANSIENT_RETRIES {
                        return Err(format!(
                            "Poll failed after {MAX_TRANSIENT_RETRIES} consecutive transient errors ({poll_status})"
                        ));
                    }
                    warn!(
                        %poll_status,
                        attempt = transient_failures,
                        "Transient poll HTTP error, retrying..."
                    );
                    continue;
                }
                let body = poll_resp
                    .text()
                    .await
                    .unwrap_or_else(|_| "failed to read response body".to_string());
                warn!(%poll_status, body = body, "Poll error");
                return Err(format!("Poll error ({poll_status}): {body}"));
            }

            // Reset transient counter on success
            transient_failures = 0;

            let result: VideoPollResponse = poll_resp
                .json()
                .await
                .map_err(|e| format!("Failed to parse poll response: {e}"))?;

            match result.status.as_str() {
                "done" => {
                    info!(
                        request_id = submit.request_id,
                        elapsed_secs = elapsed,
                        "Video generation complete"
                    );
                    return Ok(result);
                }
                "expired" => {
                    return Err(format!(
                        "Video generation expired (request_id: {})",
                        submit.request_id
                    ));
                }
                _ => {
                    info!(
                        request_id = submit.request_id,
                        status = result.status,
                        elapsed_secs = elapsed,
                        "Still generating..."
                    );
                }
            }
        }
    }

    /// Download a video from a temporary URL and save it to disk.
    ///
    /// Streams to a temp file first, then renames with the correct extension
    /// detected from the first bytes. This avoids buffering the entire video in memory.
    async fn download_video(&self, url: &str) -> Result<PathBuf, String> {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| format!("Failed to download video: {e}"))?;

        if !response.status().is_success() {
            return Err(format!("Video download failed ({})", response.status()));
        }

        tokio::fs::create_dir_all(&self.save_dir)
            .await
            .map_err(|e| {
                format!(
                    "Failed to create save directory {}: {e}",
                    self.save_dir.display()
                )
            })?;

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let seq = self.counter.fetch_add(1, Ordering::Relaxed);

        // Stream to a temp file
        let tmp_path = self.save_dir.join(format!("{ts}_{seq}.tmp"));
        let mut file = tokio::fs::File::create(&tmp_path)
            .await
            .map_err(|e| format!("Failed to create temp file: {e}"))?;

        let mut stream = response.bytes_stream();
        let mut total_bytes: u64 = 0;
        let mut header_bytes: Vec<u8> = Vec::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Stream read error: {e}"))?;
            if header_bytes.len() < 12 {
                header_bytes.extend_from_slice(&chunk[..chunk.len().min(12 - header_bytes.len())]);
            }
            total_bytes += chunk.len() as u64;
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("Failed to write chunk: {e}"))?;
        }

        file.flush()
            .await
            .map_err(|e| format!("Failed to flush file: {e}"))?;
        drop(file);

        // Detect format from the first bytes we saved
        let ext = if header_bytes.len() >= 8 && &header_bytes[4..8] == b"ftyp" {
            "mp4"
        } else if header_bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
            "webm"
        } else {
            "mp4" // default
        };

        let final_path = self.save_dir.join(format!("{ts}_{seq}.{ext}"));
        tokio::fs::rename(&tmp_path, &final_path)
            .await
            .map_err(|e| format!("Failed to rename temp file: {e}"))?;

        info!(path = %final_path.display(), size = total_bytes, "Video saved to disk");
        Ok(final_path)
    }

    /// Format a successful video response as MCP content.
    async fn format_video_response(&self, resp: &VideoPollResponse) -> Vec<Content> {
        let mut contents = Vec::new();

        if let Some(video) = &resp.video {
            // Download and save locally
            match self.download_video(&video.url).await {
                Ok(path) => {
                    let mut parts = vec![format!("Saved: {}", path.display())];
                    if let Some(dur) = video.duration {
                        parts.push(format!("Duration: {dur}s"));
                    }
                    parts.push(format!("URL: {}", video.url));
                    contents.push(Content::text(parts.join("\n")));
                }
                Err(e) => {
                    warn!("Failed to download video: {e}");
                    let mut parts = vec![format!("URL: {}", video.url)];
                    if let Some(dur) = video.duration {
                        parts.push(format!("Duration: {dur}s"));
                    }
                    parts.push(format!("(Download failed: {e})"));
                    contents.push(Content::text(parts.join("\n")));
                }
            }
        } else {
            contents.push(Content::text("No video data returned by API."));
        }

        contents
    }
}

#[tool_router]
impl GrokVideoServer {
    fn new(
        api_key: String,
        save_dir: PathBuf,
        poll_interval: std::time::Duration,
        poll_timeout: std::time::Duration,
    ) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(HTTP_TIMEOUT_SECS))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key,
            save_dir,
            http,
            counter: Arc::new(AtomicU64::new(0)),
            poll_interval,
            poll_timeout,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Generate a video from a text prompt using Grok's video generation API. Returns a locally saved video file path. Generation may take up to several minutes."
    )]
    async fn generate_video(
        &self,
        Parameters(params): Parameters<GenerateVideoParams>,
    ) -> Result<CallToolResult, McpError> {
        if let Err(e) = validate_duration(params.duration) {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }
        if let Err(e) = validate_aspect_ratio(params.aspect_ratio.as_deref()) {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }
        if let Err(e) = validate_resolution(params.resolution.as_deref()) {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }

        let model = params.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
        info!(model, prompt = params.prompt, "generate_video called");

        let request = VideoGenerationRequest {
            model,
            prompt: params.prompt,
            duration: params.duration,
            aspect_ratio: params.aspect_ratio,
            resolution: params.resolution,
            image_url: None,
            video_url: None,
        };

        match self.submit_and_poll(&request).await {
            Ok(resp) => {
                let contents = self.format_video_response(&resp).await;
                Ok(CallToolResult::success(contents))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    #[tool(
        description = "Animate a still image into a video clip using Grok's video generation API. Provide an image URL, local file path, or base64 data URI along with a prompt describing the desired motion and atmosphere."
    )]
    async fn animate_image(
        &self,
        Parameters(params): Parameters<AnimateImageParams>,
    ) -> Result<CallToolResult, McpError> {
        if let Err(e) = validate_duration(params.duration) {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }
        if let Err(e) = validate_aspect_ratio(params.aspect_ratio.as_deref()) {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }
        if let Err(e) = validate_resolution(params.resolution.as_deref()) {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }

        let image_url = match resolve_url(&params.image_url).await {
            Ok(url) => url,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let model = params.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
        info!(model, prompt = params.prompt, "animate_image called");

        let request = VideoGenerationRequest {
            model,
            prompt: params.prompt,
            duration: params.duration,
            aspect_ratio: params.aspect_ratio,
            resolution: params.resolution,
            image_url: Some(image_url),
            video_url: None,
        };

        match self.submit_and_poll(&request).await {
            Ok(resp) => {
                let contents = self.format_video_response(&resp).await;
                Ok(CallToolResult::success(contents))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    #[tool(
        description = "Edit an existing video using natural language instructions via Grok's video API. Input video capped at 8.7 seconds; output matches input resolution and aspect ratio."
    )]
    async fn edit_video(
        &self,
        Parameters(params): Parameters<EditVideoParams>,
    ) -> Result<CallToolResult, McpError> {
        if let Err(e) = validate_resolution(params.resolution.as_deref()) {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }

        let video_url = match resolve_url(&params.video_url).await {
            Ok(url) => url,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let model = params.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
        info!(model, prompt = params.prompt, "edit_video called");

        let request = VideoGenerationRequest {
            model,
            prompt: params.prompt,
            duration: None,     // editing ignores custom duration
            aspect_ratio: None, // editing maintains input ratio
            resolution: params.resolution,
            image_url: None,
            video_url: Some(video_url),
        };

        match self.submit_and_poll(&request).await {
            Ok(resp) => {
                let contents = self.format_video_response(&resp).await;
                Ok(CallToolResult::success(contents))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }
}

#[tool_handler]
impl ServerHandler for GrokVideoServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("mcp-server-grok-video", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Grok video generation server. Use generate_video to create videos from text prompts, \
                 animate_image to turn still images into video clips, \
                 or edit_video to modify existing videos with natural language instructions."
            )
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cfg = load_config()?;
    let save_dir = PathBuf::from(&cfg.save_dir);
    let poll_interval = std::time::Duration::from_secs(cfg.poll_interval_secs);
    let poll_timeout = std::time::Duration::from_secs(cfg.poll_timeout_secs);

    info!(
        save_dir = %save_dir.display(),
        poll_interval_secs = cfg.poll_interval_secs,
        poll_timeout_secs = cfg.poll_timeout_secs,
        "Starting mcp-server-grok-video"
    );

    let server = GrokVideoServer::new(cfg.api_key, save_dir, poll_interval, poll_timeout);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
