# Deploying Grimalkin on the Network

Grimalkin binds to `127.0.0.1` by default — only reachable from the local
machine. If you need to expose it on the LAN or beyond, follow the steps
below.

## 1. Enable non-loopback binding

```bash
python grimalkin.py --host 0.0.0.0
```

You will see a warning on stderr reminding you to set up TLS.

## 2. Set an auth token

Without a token, anyone who can reach the port can use the UI.

```bash
export GRIM_AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python grimalkin.py --host 0.0.0.0
```

Gradio will show a login prompt. Enter anything as the username and the
token as the password.

## 3. Terminate TLS with a reverse proxy

Gradio does not support TLS natively. Place a reverse proxy in front of it
so credentials are never sent in cleartext.

### Caddy (automatic HTTPS)

```
# /etc/caddy/Caddyfile
grimalkin.example.com {
    reverse_proxy localhost:7860
}
```

Caddy obtains and renews certificates automatically via Let's Encrypt.

```bash
sudo systemctl reload caddy
```

### nginx (manual certificate)

```nginx
# /etc/nginx/sites-available/grimalkin
server {
    listen 443 ssl;
    server_name grimalkin.example.com;

    ssl_certificate     /etc/letsencrypt/live/grimalkin.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/grimalkin.example.com/privkey.pem;

    location / {
        proxy_pass         http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host       $host;
        proxy_set_header   X-Real-IP  $remote_addr;
    }
}

server {
    listen 80;
    server_name grimalkin.example.com;
    return 301 https://$host$request_uri;
}
```

The `Upgrade` / `Connection` headers are required — Gradio uses WebSockets.

```bash
sudo certbot certonly --nginx -d grimalkin.example.com
sudo nginx -t && sudo systemctl reload nginx
```

## 4. Firewall

Only expose port 443 (HTTPS). Never expose port 7860 directly.

```bash
sudo ufw allow 443/tcp
sudo ufw deny 7860/tcp
```

## Checklist

- [ ] `GRIM_AUTH_TOKEN` is set and stored securely (not in shell history)
- [ ] Reverse proxy is running with a valid TLS certificate
- [ ] Port 7860 is firewalled from external access
- [ ] Only port 443 is exposed
