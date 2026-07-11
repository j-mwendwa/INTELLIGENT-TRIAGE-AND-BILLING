# Technical Support Knowledge Base

## Common API Error Codes

### 4xx Client Errors

| Code | Meaning | Resolution |
|---|---|---|
| 400 | Bad Request | Check request body format and required fields |
| 401 | Unauthorized | Verify API key is valid and not expired |
| 403 | Forbidden | Account lacks permission for this operation |
| 404 | Not Found | Check endpoint URL and resource ID |
| 422 | Validation Error | Review field types and constraints in API docs |
| 429 | Rate Limited | Implement exponential backoff; check rate limits |

### 5xx Server Errors

| Code | Meaning | Resolution |
|---|---|---|
| 500 | Internal Server Error | Retry after 30s; report if persistent |
| 502 | Bad Gateway | Upstream service issue; retry with backoff |
| 503 | Service Unavailable | System overload or maintenance; see status page |
| 504 | Gateway Timeout | Request took too long; reduce payload or retry |

---

## Service Level Agreement (SLA)

### Uptime Commitment
- **Professional & Enterprise**: 99.9% monthly uptime (≤ 43.8 min downtime/month)
- **Starter**: 99.5% monthly uptime
- Enterprise+ customers have a 99.95% SLA with financial penalties for breach

### Service Credits for Downtime
| Uptime in Month | Credit |
|---|---|
| 99.0% – 99.9% | 10% of monthly fee |
| 95.0% – 99.0% | 25% of monthly fee |
| < 95.0% | 50% of monthly fee |

Credits are applied automatically. They do not apply to scheduled maintenance windows.

---

## API Integration Guide

### Authentication
All API requests must include the `X-API-Key` header:
```
X-API-Key: your-api-key-here
```

### Rate Limits
| Plan | Requests/minute | Requests/day |
|---|---|---|
| Starter | 60 | 10,000 |
| Professional | 300 | 50,000 |
| Enterprise | 1,000 | 250,000 |

### Retry Strategy (Recommended)
```python
import time

def api_call_with_retry(url, headers, data, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 429:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            time.sleep(wait)
            continue
        return response
    raise Exception("Max retries exceeded")
```

---

## Troubleshooting Common Issues

### Issue: 503 Errors Since Deployment

**Symptoms:** Intermittent or persistent 503 responses

**Steps to diagnose:**
1. Check the [Status Page](https://status.example.com) for active incidents
2. Verify your API key is valid: `GET /health` with your key
3. Check if the issue is regional (try from a different location/VPN)
4. Review your request rate — are you hitting limits?
5. Check if the issue occurs on all endpoints or just one

**If the issue persists:**
- Collect 5 consecutive request/response logs with timestamps
- Note your account ID and the affected endpoint
- Submit a Priority support ticket with these details

### Issue: Webhooks Stopped Delivering

**Symptoms:** Events not arriving at your endpoint

**Checklist:**
1. Verify your webhook URL returns HTTP 200 within 5 seconds
2. Check your server's SSL certificate is valid (must be HTTPS)
3. Ensure your firewall allows inbound connections from our IP ranges: `203.0.113.0/24`
4. Check the Webhook Delivery Log in your dashboard for failures
5. Verify the event types you subscribed to are still enabled

**Common causes:**
- Endpoint returning 4xx or 5xx → We stop retrying after 5 failures
- SSL certificate expired → Update your certificate
- Timeout → Process webhook asynchronously, return 200 immediately

### Issue: High API Latency

**Expected latencies:**
- P50: < 200ms
- P99: < 2,000ms

**If consistently above:**
1. Check whether your region is experiencing issues (status page)
2. Review payload size — large payloads increase latency
3. Use our CDN endpoint for static data: `cdn.api.example.com`
4. Consider caching frequently-requested resources

---

## Scheduled Maintenance Windows

Maintenance is conducted:
- **Weekly**: Tuesdays 02:00–04:00 UTC (rolling, not all regions simultaneously)
- **Monthly**: First Saturday 01:00–05:00 UTC (major updates)

Customers are notified via email and status page 48 hours in advance.

---

## Feature Requests & Bug Reports

- **Feature requests**: Submit via `feedback.example.com` — voted on quarterly
- **Bug reports**: Use the in-app bug reporter or email `bugs@example.com`
- **Security vulnerabilities**: `security@example.com` (PGP key available on request)
