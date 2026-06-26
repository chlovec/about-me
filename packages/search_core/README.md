# Setup

1. Create the pyproject.toml files.
   - 1 at the project level
   - 1 for each python project added as member in the project level pyproject.toml

1. To create the project-level virtual environment, run the commands below at root directory

   ```\bash
   uv venv
   ```

1. Run the command below

   ```\bash
   uv sync
   ```

1. To add dependencies, use the example command below.

   ```\bash
   uv add numpy --project packages/search_core
   ```

1. Use the command below to run all the unit tests for the python projects

   ```\bash
   uv run pytest
   ```

1. Use the command below to run all the unit tests for search-core python projects

   ```\bash
   uv run --package search-core pytest
   ```
