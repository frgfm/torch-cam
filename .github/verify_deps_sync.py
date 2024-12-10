# Copyright (C) 2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from pathlib import Path

import tomllib
import yaml

PRECOMMIT_PATH = ".pre-commit-config.yaml"
PYPROJECT_PATH = "pyproject.toml"


def main():
    # Retrieve & parse all deps files
    deps_dict = {}
    # UV: Dockerfile, precommit, .github
    # Parse precommit
    with Path(PRECOMMIT_PATH).open("r") as f:
        precommit = yaml.safe_load(f)

    for repo in precommit["repos"]:
        if repo["repo"] == "https://github.com/astral-sh/uv-pre-commit":
            if "uv" not in deps_dict:
                deps_dict["uv"] = []
            deps_dict["uv"].append({"file": PRECOMMIT_PATH, "version": repo["rev"].lstrip("v")})
        elif repo["repo"] == "https://github.com/charliermarsh/ruff-pre-commit":
            if "ruff" not in deps_dict:
                deps_dict["ruff"] = []
            deps_dict["ruff"].append({"file": PRECOMMIT_PATH, "version": repo["rev"].lstrip("v")})

    # Parse pyproject.toml
    with Path(PYPROJECT_PATH).open("rb") as f:
        pyproject = tomllib.load(f)

    dev_deps = pyproject["project"]["optional-dependencies"]["quality"]
    for dep in dev_deps:
        if dep.startswith("ruff=="):
            if "ruff" not in deps_dict:
                deps_dict["ruff"] = []
            deps_dict["ruff"].append({"file": PYPROJECT_PATH, "version": dep.split("==")[1]})
        elif dep.startswith("mypy=="):
            if "mypy" not in deps_dict:
                deps_dict["mypy"] = []
            deps_dict["mypy"].append({"file": PYPROJECT_PATH, "version": dep.split("==")[1]})

    # Parse github/workflows/...
    for workflow_file in Path(".github/workflows").glob("*.yml"):
        with workflow_file.open("r") as f:
            workflow = yaml.safe_load(f)
            if "env" in workflow and "UV_VERSION" in workflow["env"]:
                if "uv" not in deps_dict:
                    deps_dict["uv"] = []
                deps_dict["uv"].append({
                    "file": str(workflow_file),
                    "version": workflow["env"]["UV_VERSION"].lstrip("v"),
                })

    # Assert all deps are in sync
    troubles = []
    for dep, versions in deps_dict.items():
        versions_ = {v["version"] for v in versions}
        if len(versions_) != 1:
            inv_dict = {v: set() for v in versions_}
            for version in versions:
                inv_dict[version["version"]].add(version["file"])
            troubles.extend([
                f"{dep}:",
                "\n".join(f"- '{v}': {', '.join(files)}" for v, files in inv_dict.items()),
            ])

    if len(troubles) > 0:
        raise AssertionError("Some dependencies are out of sync:\n\n" + "\n".join(troubles))


if __name__ == "__main__":
    main()
