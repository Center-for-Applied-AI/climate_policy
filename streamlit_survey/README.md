# Climate Policy Engagement Survey

Streamlit app for collecting human ratings of climate policy comments. Presents 13 comments + 2 attention checks per participant, stores responses in Google Sheets.

## Setup

From the project root:

```bash
pip install -e ".[survey]"
```

### Google Sheets Integration

1. Create Google Cloud service account with Sheets API enabled
2. Download JSON key file
3. Share your Google Sheet with the service account email (Editor access)
4. Create `.streamlit/secrets.toml`:

```toml
[connections.gsheets]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour-Key-Here\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

### Environment Variables

```bash
export GOOGLE_SHEETS_URL="https://docs.google.com/spreadsheets/d/YOUR_ID/edit"
```

Completion codes have defaults in `app.py`. To override, set environment variables or add to `.streamlit/secrets.toml`.

## Run

```bash
streamlit run app.py
```

## Google Sheets Structure

**results** worksheet: `prolific_id`, `file_id`, `score`, `timestamp`

**order** worksheet: `order_id`, `prolific_id`, `comment_order`, `timestamp`, `batch_index`, `status`

