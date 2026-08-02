#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# set_github_secrets.sh
#
# Sets all required GitHub Actions secrets for the TRIAGE CI/CD pipeline.
# Uses the GitHub CLI (gh). Run once after cloning or rotating credentials.
#
# Prerequisites:
#   1. Install gh CLI: https://cli.github.com/
#   2. Authenticate:  gh auth login
#   3. Fill in the values below (retrieve with: az acr credential show --name triageregistry2026)
#   4. Run: bash scripts/set_github_secrets.sh
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo 'YOUR_GITHUB_USER/YOUR_REPO')"
echo "🔐 Setting GitHub Actions secrets for: $REPO"
echo ""

gh_secret() {
  local name="$1"
  local value="$2"
  if [ -z "$value" ] || [ "$value" = "REPLACE_ME" ]; then
    echo "  ⚠️  Skipping $name — value not set"
    return
  fi
  echo "  ↳ Setting $name ..."
  printf '%s' "$value" | gh secret set "$name" --repo "$REPO"
}

# ── Azure Identity ────────────────────────────────────────────────────────────
# Retrieve with: az acr credential show --name triageregistry2026 -o json
# Service principal: triage-github-actions-sp (AcrPush + Contributor on triage-rg)
ACR_LOGIN_SERVER="triageregistry2026.azurecr.io"
ACR_USERNAME="REPLACE_ME"          # az ad sp create-for-rbac → appId
ACR_PASSWORD="REPLACE_ME"          # az ad sp create-for-rbac → password
AZURE_TENANT_ID="d98a15bd-46fc-4f0e-80ac-ba7e8f4b6618"
AZURE_SUBSCRIPTION_ID="b968f6c0-8fbf-4203-bd27-959a2be155c9"
AZURE_RESOURCE_GROUP="triage-rg"

gh_secret "ACR_LOGIN_SERVER"      "$ACR_LOGIN_SERVER"
gh_secret "ACR_USERNAME"          "$ACR_USERNAME"
gh_secret "ACR_PASSWORD"          "$ACR_PASSWORD"
gh_secret "AZURE_TENANT_ID"       "$AZURE_TENANT_ID"
gh_secret "AZURE_SUBSCRIPTION_ID" "$AZURE_SUBSCRIPTION_ID"
gh_secret "AZURE_RESOURCE_GROUP"  "$AZURE_RESOURCE_GROUP"

# ── Azure Container Apps ──────────────────────────────────────────────────────
AZURE_CONTAINERAPP_STAGING="triage-api-app"
AZURE_CONTAINERAPP_PROD="triage-api-app-prod"

gh_secret "AZURE_CONTAINERAPP_STAGING" "$AZURE_CONTAINERAPP_STAGING"
gh_secret "AZURE_CONTAINERAPP_PROD"    "$AZURE_CONTAINERAPP_PROD"

# ── Application Secrets ───────────────────────────────────────────────────────
# Fill these from your .env file (never commit the actual values)
GOOGLE_API_KEY="REPLACE_ME"
QDRANT_URL="REPLACE_ME"
QDRANT_API_KEY="REPLACE_ME"
ALLOWED_API_KEYS="REPLACE_ME"
LANGSMITH_API_KEY="REPLACE_ME"

gh_secret "GOOGLE_API_KEY"    "$GOOGLE_API_KEY"
gh_secret "QDRANT_URL"        "$QDRANT_URL"
gh_secret "QDRANT_API_KEY"    "$QDRANT_API_KEY"
gh_secret "ALLOWED_API_KEYS"  "$ALLOWED_API_KEYS"
gh_secret "LANGSMITH_API_KEY" "$LANGSMITH_API_KEY"

echo ""
echo "✅ Done. Verify at:"
echo "   https://github.com/$REPO/settings/secrets/actions"
