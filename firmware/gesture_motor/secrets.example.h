// Copy to secrets.h (git-ignored) and fill in. Never commit real credentials.
#pragma once

#define WIFI_SSID "your-network"
#define WIFI_PASS "your-password"

// Shared secret; must match the GESTURECTL_TOKEN env var on the PC.
// Generate one with:  python -c "import secrets; print(secrets.token_hex(8))"
#define CONTROL_TOKEN "change-me"
