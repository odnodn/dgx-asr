# Production deployment checklist

Use this when running the Parakeet MLX server in production (e.g. behind nginx, reachable from LibreChat or other clients).

## 1. Environment

Set before starting the server (or in the LaunchAgent plist when using `install_server_service.sh`):

| Variable | Required in production | Example |
|----------|------------------------|---------|
| `ENV` | Yes | `production` |
| `API_KEY` | Yes | Strong secret (e.g. 32+ random chars) |
| `CORS_ORIGINS` | Recommended | `https://your-librechat.example.com,http://192.168.178.20` |
| `BIND` | Optional | `127.0.0.1` if only nginx talks to the app; `0.0.0.0` if clients hit the host directly |
| `LOG_LEVEL` | Optional | `INFO` or `WARNING` |
| `ALLOW_LAN` | Optional | `0` in production (use explicit `CORS_ORIGINS` instead of `*`) |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | Optional | `2` (limit simultaneous jobs; 1–2 is stable for MLX) |
| `TRANSCRIPTION_TIMEOUT` | Optional | `600` (max seconds per transcription; avoids stuck requests) |
| `UVICORN_TIMEOUT_KEEP_ALIVE` | Optional | `30` (seconds) |
| `UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN` | Optional | `15.0` (seconds) |

Production behaviour when `ENV=production`:

- Server **refuses to start** without `API_KEY`.
- CORS `*` is not allowed; use explicit `CORS_ORIGINS`.
- `/docs`, `/redoc`, and `/openapi.json` are disabled.

## 2. Start the server

**Interactive (for testing):**

```bash
export ENV=production
export API_KEY="your-strong-secret-here"
export CORS_ORIGINS="https://your-librechat.example.com"
./start_server.sh
```

**As a service (LaunchAgent):**

```bash
export ENV=production
export API_KEY="your-strong-secret-here"
export CORS_ORIGINS="https://your-librechat.example.com"
./install_server_service.sh
```

The install script writes `API_KEY`, `ENV`, `CORS_ORIGINS`, and `LOG_LEVEL` (if set) into the LaunchAgent plist so the server runs with production settings after reboot.

## 3. Reverse proxy (nginx)

- Run the app on `127.0.0.1:8002` when only nginx should reach it (`BIND=127.0.0.1` or leave unset in production).
- Put nginx in front for TLS (HTTPS), and optionally rate limiting and logging.
- Example location: `/stt/` → `http://127.0.0.1:8002/` with `client_max_body_size 100M`.

## 4. LibreChat (or other clients)

- **URL:** `https://your-domain/stt/v1/audio/transcriptions` (or the same path you use in nginx).
- **API key:** Same value as `API_KEY` on the Parakeet server (e.g. set `STT_API_KEY` in LibreChat’s `.env` and reference it in `librechat.yaml`).

## 5. Quick checklist

- [ ] `ENV=production` set
- [ ] Strong `API_KEY` set (and same key configured in LibreChat)
- [ ] `CORS_ORIGINS` set to your front-end origins (no `*`)
- [ ] Server behind HTTPS (nginx or other)
- [ ] If using LaunchAgent: install was run with `ENV` and `API_KEY` (and optionally `CORS_ORIGINS`) set so they are in the plist
- [ ] Logs monitored (`logs/parakeet-server.log` and `parakeet-server.err` when using the service)

---

## 6. What else for production?

| Area | Recommendation |
|------|----------------|
| **HTTPS** | Terminate TLS at nginx (e.g. Let’s Encrypt). Do not expose the app directly on the internet without TLS. |
| **Rate limiting** | In nginx, use `limit_req_zone` / `limit_req` on the `/stt/` location to cap requests per IP (e.g. 10 req/s). Reduces abuse and load. |
| **Log rotation** | LaunchAgent writes to `logs/parakeet-server.log` and `parakeet-server.err`. Use macOS `newsyslog` or a cron job to rotate/compress so disks don’t fill. |
| **Health checks** | **Liveness:** `GET /live` always returns 200 if the process is up (e.g. k8s livenessProbe). **Readiness:** `GET /health` returns 200 only when the model is loaded; use for readiness or “can accept work” checks. |
| **Dependencies** | Run `pip list --outdated` or `pip-audit` occasionally; update `requirements.txt` and redeploy after testing. |
| **Firewall** | If the app binds to `0.0.0.0`, restrict port 8002 to trusted IPs (e.g. nginx on the same host). Prefer `BIND=127.0.0.1` and only nginx on a public interface. |
| **Secrets** | Do not commit `.env` or plist files containing `API_KEY`. The LaunchAgent plist lives in `~/Library/LaunchAgents/` and is not in the repo. |
| **Resource limits** | Optional: use `launchctl limit` or a wrapper to cap memory if you want to avoid one process using all RAM. |

**Optional (nice to have):**

- **Metrics**: Expose something like `GET /metrics` (Prometheus format) for request count/latency; add later if you use a monitoring stack.
- **Backups**: No DB; the only “data” is the HuggingFace model cache. Backing up the cache is optional and only needed to avoid re-downloading.
