#!/bin/bash
# Run a command with no credentials in its environment: only system paths, the proxy settings
# the API calls go through (the proxy injects the Meta key itself) and the oneAPI toolchain.
exec env -i HOME="$HOME" USER="${USER:-root}" LANG=C.UTF-8 TERM=dumb \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin" \
  HTTPS_PROXY="$HTTPS_PROXY" HTTP_PROXY="$HTTP_PROXY" https_proxy="$https_proxy" http_proxy="$http_proxy" \
  NO_PROXY="$NO_PROXY" no_proxy="$no_proxy" \
  SSL_CERT_FILE="$SSL_CERT_FILE" REQUESTS_CA_BUNDLE="$REQUESTS_CA_BUNDLE" CURL_CA_BUNDLE="$CURL_CA_BUNDLE" \
  NODE_EXTRA_CA_CERTS="$NODE_EXTRA_CA_CERTS" \
  ONEAPI_AGENT_AUTH_FROM_NETWORK=1 \
  bash -c 'source /home/user/oneapi-agent-toolkit/scripts/oneapi-env.sh >/dev/null 2>&1; exec "$@"' _ "$@"
