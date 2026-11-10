# IT Service Desk Runbook

> Sample knowledge-base content for this project. Replace it with approved company guidance before using the assistant for real support.

## Email Troubleshooting
If a user cannot send or receive email, confirm whether the issue affects one user or many users. Check network connectivity and the service status dashboard. For a single user, sign out and back in, verify mailbox quota, and test the web client. For widespread incidents, do not ask each user to reinstall software; escalate the suspected service incident.

For a single-user issue, note the approximate start time, whether sending, receiving, or both are affected, and whether the same issue occurs in the web client. Check for an obvious connectivity problem and any service notice available through the company's known status page. Do not ask the user to share their password or MFA code. If several people are affected, report the scope to the service desk rather than troubleshooting each workstation as an isolated case.

## Laptop Performance
For a slow laptop, first check CPU, memory, storage free space, startup applications, operating system updates, and endpoint security scan status. Do not disable security agents as a troubleshooting shortcut. Escalate when hardware diagnostics report failure or repeated shutdowns continue after software remediation.

Ask whether the slowdown is constant or limited to a particular application or task, and whether it began after a recent change. Save open work before restarting. Close applications the user is not using, then observe whether performance improves. Follow company-approved update and diagnostic prompts; do not disable endpoint protection, change security settings, or install unapproved cleanup tools. Escalate persistent slowness, hardware warnings, overheating, or repeated shutdowns to the IT Service Desk.

## VPN Connection Problems
Confirm that the device has internet access before troubleshooting the VPN. Have the user restart the approved VPN client and try a different company gateway if one is available in the client. Record the displayed error and whether the failure occurs before or after MFA. Do not bypass MFA, disable firewall or endpoint protection, or use an unapproved VPN. If the issue continues, contact the IT Service Desk and include the error text and approximate time.

## Account or Sign-In Problems
Confirm which company resource is affected and whether the user can sign in to other company services. For a forgotten password, refer to the Password Reset Policy in the NovaRetail Internal IT Support Handbook. This sample KB does not specify the identity portal address or identity-verification questions; direct users to a company-approved bookmark or intranet page, or to the IT Service Desk through a known channel. Do not invent a reset link or ask the user to disclose a password, MFA code, or recovery code.

If the user reports repeated failures or a possible account lock, discourage further guesses because five failed login attempts may temporarily lock the account. Route unresolved lockouts or failed self-service recovery to the IT Service Desk. For an unexpected MFA prompt, advise the user to deny it and report it.

## Reporting an Incident
When opening a ticket, describe the affected service or device, the observed behavior, when it began, how many people appear to be affected, and any troubleshooting already attempted. Include the exact error text or a screenshot when it contains no passwords, MFA codes, recovery codes, or other secrets. Use the company's approved ticketing channel; this sample runbook does not specify a ticket URL, phone number, or response-time commitment.

Assign priority using the levels listed in Ticket Priority. Escalate suspected security incidents through the established company security process; do not investigate by accessing another person's account or device.

## Ticket Priority
P1: company-wide outage or critical security incident. P2: major team blocked with no workaround. P3: individual user blocked or degraded with limited workaround. P4: standard request, information request, or low-impact issue.
