# Acme Corp Security Handbook

## Password Requirements

All employee accounts must use passwords that are at least 14 characters
long, include a mix of upper and lowercase letters, numbers, and symbols.
Passwords must be rotated every 180 days. Reuse of the previous 10
passwords is prohibited.

## Two-Factor Authentication

Two-factor authentication (2FA) is mandatory for all production systems,
source-code repositories, and email. Approved 2FA methods are hardware
security keys (YubiKey), authenticator apps (Authy, 1Password), and SMS as
a last resort.

## Data Classification

Company data is classified into four tiers:
- Public: marketing materials, open-source code
- Internal: org charts, team wikis, policies
- Confidential: customer data, financials, contracts
- Restricted: source code for the core platform, security keys, payroll

Restricted data must be stored only on encrypted drives and transmitted
over TLS 1.3 or higher.

## Incident Response

Security incidents must be reported to security@acme.example within 1 hour
of discovery. The incident response team will acknowledge within 30 minutes
and coordinate remediation. Post-incident reviews are mandatory within 5
business days.

## Device Security

All laptops must have full-disk encryption enabled (FileVault on macOS,
BitLocker on Windows). Screen lock must engage after 5 minutes of
inactivity. Lost or stolen devices must be reported to IT within 2 hours.
