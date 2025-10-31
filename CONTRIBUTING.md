# Contributing

Thanks for helping improve the OpenAI Knowledge Retrieval starter kit! This document explains how to set up your environment, propose changes, and keep the project healthy.

## Code of Conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please report
unacceptable behavior to [opensource@openai.com](mailto:opensource@openai.com).

## Development Environment

1. **Clone and bootstrap**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   make dev  # installs the project in editable mode with dev extras
   ```

2. **Install frontend dependencies (optional UI)**

   ```bash
   cd app/frontend
   npm install
   ```

3. **Copy environment variables**

   ```
   cp .env.example .env
   # edit the file and supply OPENAI_API_KEY plus any optional IDs
   ```

## Coding Standards

- Python 3.10+, type hints required for new code.
- Keep functions small and focused; extract shared logic into helpers or modules under `lib/`.
- For React code, favor functional components, hooks, and composition to avoid duplication.
- Document public functions and classes with concise docstrings where the intent is not obvious.
- Use descriptive kebab-case filenames for new frontend modules and snake_case for Python files.



## Commit and PR Process

- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, etc.) in the imperative mood.
- Keep pull requests focused and include a summary of the behavior change plus testing evidence.
- Ensure:

  ```bash
  ruff check .
  pytest -q
  npm run lint
  ```

  all pass locally (run the npm commands only if you touched UI code).

## Proposing Larger Changes

For significant work (new features or API changes), please open an issue to discuss the design
before submitting a PR. Include the problem statement, proposed solution, and any alternatives you
considered.

## Reporting Issues

Use GitHub issues for bugs or feature requests. Include reproduction steps, expected behavior, and
environment details (OS, Python/Node versions, configuration used).

## License

By contributing you agree that your contributions will be licensed under the MIT License.
