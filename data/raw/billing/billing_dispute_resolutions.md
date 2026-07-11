# Billing Dispute Resolution Records — Real Cases

---

## Case DSP-2025-0007 — Double-Charge Dispute

**Customer:** TechFlow Inc (Professional Plan — $149/mo)
**Filed:** 2025-01-12
**Disputed Amount:** $149.00
**Subject:** Invoice INV-2025-00098 — charger twice in same billing cycle

### Customer Statement
"Your system charged my credit card twice on Jan 10th. Two separate transactions for $149.00 each appeared on my bank statement. Your dashboard only shows one invoice."

### Investigation Notes
- Invoice system shows one invoice generated
- Payment gateway logs show two Authorization + Capture events within 47 seconds of each other
- Root cause: Stripe webhook delivery timeout caused retry; both webhooks were processed as separate payments
- Duplicate payment identified in Stripe reconciliation report

### Resolution
- **Date:** 2025-01-14
- **Decision:** Fully refunded — Billing Error per policy
- **Actions:**
  1. Refund of $149.00 issued via Stripe (immediate reversal)
  2. Credit note CN-2025-00007 applied
  3. Engineering flagged webhook deduplication as a backlog item
- **Communication:** Customer notified within 2 hours of discovery

---

## Case DSP-2025-0015 — Overage Charge Challenge

**Date:** 2025-02-18
**Customer:** MediCal Systems (Enterprise, $499/mo)
**Disputed Amount:** $312.45
**Status:** RESOLVED — Partial Credit
**Disputed Item:** INV-2025-00234 — API overage of 62,490 calls

### Customer Statement
"We were under a maintenance window where webhook delivery was paused. After enabling webhooks, 65,000 queued events delivered simultaneously — this artificially inflated our API call volume for Feb 8-9. We believe these should not count towards our monthly limit."

### Investigation
- Usage logs confirmed webhook queue drain of 58,642 events on Feb 8-10
- API call count for Feb 8-9: 81,234 vs. normal daily average of 4,500
- Policy review: All API calls are counted regardless of source (per pricing terms: "API calls above plan limit")
- However, the billing team determined the spike was induced by platform-side delivery queuing

### Resolution
- **Date:** 2025-02-22
- **Decision (Partial):** 50% of dispute credited as goodwill adjustment
- **Breakdown:**
  - Total overage charges disputed: $499.45 (62,490 × $0.005 plus ~$186.95 of that for queue flush)
  - Goodwill credit: $93.48 (50% of queue-related overage)
- **Action:** Applied AJ-2025-00002 as one-time goodwill adjustment
- **Prevention:** Documentation updated for maintenance window behaviour

---

## Case DSP-2025-0029 — Wire Transfer Fee Dispute

**Date:** 2025-03-25
**Customer:** GlobalPay Financial (Enterprise+, $12,000/mo)
**Disputed Amount:** $25.00
**Status:** RESOLVED — Fee Upheld

### Customer Statement
"We processed our wire transfer through your recommended banking partner. A $25 processing fee was added. We do not believe international customers should pay this fee simply because we cannot use ACH (US-only)."

### Investigation
- Policy review: "Wire Transfer — International — $25 processing fee applies" clearly documented under Accepted Payment Methods
- ACH is explicitly limited to US accounts per terms of service
- Customer is based in Singapore — no ACH available
- Enterprise+ agreement does not mention waived wire fees

### Resolution
- **Date:** 2025-03-27
- **Decision:** Fee upheld — Policy clearly states the fee
- **Action:**
  1. Explained policy to customer with direct quote from Payment Methods section
  2. Offered to switch to PayPal (available for plans under $500/mo — not applicable for $12K/mo plan)
  3. Flagged feedback to product team for Enterprise+ consideration
- **Note:** No alternative payment method that avoids the fee exists per current policy

---

## Case DSP-2025-0045 — Account Credit Expiration Concern

**Date:** 2025-04-20
**Customer:** Beacon Marketing (Starter Plan → Professional Plan upgrade)
**Disputed Amount:** N/A
**Status:** RESOLVED — Information Provided

### Customer Statement
"I received a $50 referral credit from a promotion in November 2024. I'm now upgrading to Professional. Someone on chat told me credits are 'non-transferable' and I'm worried I'll lose it."

### Investigation and Resolution
- **Findings:**
  1. Credit's promotional code FALL24 ran Nov 2024 — but credit was earned Dec 2024 (referred customer's 3-month mark was Feb 2025 — credit applied Mar 2025)
  2. Per Billing Policy "Credits do not expire and are non-transferable"
  3. "Non-transferable" applies to selling/transferring credits between accounts, not to plan changes
- **Decision:** Credit survives plan upgrade
- **Action:** Applied $50.00 credit to first Professional invoice INV-2025-00987 on upgrade date
- **Advice given to customer:** Referral credits from the Referral Program policy page explicitly state they "do not expire" — you will not lose your credit when upgrading.

---

## Case DSP-2025-0058 — 60-Day Suspension Termination Notice

**Date:** 2025-06-10
**Customer:** Retro Inc (Starter Plan — Suspended since Apr 5, 2025)
**Status:** ESCALATED to Collections

### Background
- Account suspended: 2025-04-05 (after 2 failed payment retries and non-payment of Mar invoice INV-2025-00398)
- Reactivation fee ($15) not paid
- No contact from customer for 66 days
- Per Billing Policy: "Accounts suspended for more than 60 days may be terminated"
- Termination threshold reached: 2025-06-08

### Actions Taken
1. Final notice sent to customer's registered email + secondary contact
2. Data retention: Account data to be purged after 30 days per Compliance policy
3. Outstanding balance of $49.00 + $15 reactivation fee + $0.74 late fees = $64.74 referred to collections
4. Account marked for termination on 2025-06-20 if no response

### Escalation Level
This case has been escalated to the Accounts Receivable team and Legal Operations.
