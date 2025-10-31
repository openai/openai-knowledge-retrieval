# Security Policy

## Supported Versions

This project is released as open source without a formal support policy. Security fixes will be
backported at the maintainers' discretion. We encourage users to follow the main branch for the
latest patches.

## Reporting a Vulnerability

If you discover a security issue, please contact us at [security@openai.com](mailto:security@openai.com)
with the details. We aim to acknowledge new reports within three business days.

Please **do not** file public GitHub issues for security vulnerabilities. Include the following
information in your email when possible:

- Description of the issue and potential impact
- Steps to reproduce or proof-of-concept exploit
- Any mitigations you have identified
- Suggested CVSS score (if available)

We will coordinate disclosure following a reasonable remediation period. If you are able to provide
a patch or mitigation, please include it.

## Best Practices for Users

- Rotate API keys and credentials regularly and store them outside of version control.
- Review `.env.example` for required configuration and avoid committing `.env` files.
- Run `trufflehog git file://.` (or similar tooling) before publishing forks to ensure secrets are
  not present in commit history.
- Periodically audit dependencies:

  ```bash
  pip install pip-audit && pip-audit
  npm audit --omit=dev
  ```

  (These commands require internet access.)

- Limit access to sample datasets that may contain sensitive information, and replace them with your
  own data before deploying publicly.
