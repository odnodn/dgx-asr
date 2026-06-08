# Start nginx at boot (macOS LaunchDaemon)

This makes nginx start automatically at boot and keeps it running. All traffic to the Parakeet STT app should go through nginx (ports 80/443).

## One-time setup

Run these commands (they require sudo):

```bash
# 1. Ensure log directory exists
sudo mkdir -p /opt/homebrew/var/log/nginx /opt/homebrew/var/run

# 2. Copy the LaunchDaemon plist to the system folder (from repo root)
sudo cp "$(pwd)/nginx/org.nginx.nginx.plist" /Library/LaunchDaemons/

# 3. Set correct ownership (root)
sudo chown root:wheel /Library/LaunchDaemons/org.nginx.nginx.plist

# 4. Load and start nginx now (and at every boot)
sudo launchctl load /Library/LaunchDaemons/org.nginx.nginx.plist
```

## Useful commands

| Action | Command |
|--------|---------|
| Start nginx | `sudo launchctl load /Library/LaunchDaemons/org.nginx.nginx.plist` |
| Stop nginx | `sudo launchctl unload /Library/LaunchDaemons/org.nginx.nginx.plist` |
| Reload config | `sudo nginx -s reload` |
| Check status | `sudo launchctl list | grep nginx` |

## Uninstall (stop at boot)

```bash
sudo launchctl unload /Library/LaunchDaemons/org.nginx.nginx.plist
sudo rm /Library/LaunchDaemons/org.nginx.nginx.plist
```

## Note

- Parakeet server binds to **127.0.0.1:8002** by default, so it is only reachable via nginx (or localhost). Set `BIND=0.0.0.0` when starting the server if you need direct access from the network.
