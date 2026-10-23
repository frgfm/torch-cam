# Copyright (C) 2024-2025, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///

import logging
import sys
import tomllib
from pathlib import Path

import yaml

PRECOMMIT_CONFIG = ".pre-commit-config.yaml"
PYPROJECTS = ["./pyproject.toml"]
MINIMUM_UV_VERSION = "0.13.4"
EXPECTED_UV_BUILD_REQUIREMENT = f"uv_build>={MINIMUM_UV_VERSION},<1.0.0"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(levelname)s:     %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


def parse_dep_str(dep_str: str) -> dict[str, str]:
    # No version specified
    if all(dep_str.find(char) == -1 for char in ("<", ">", "=")):
        version_idx = len(dep_str)
    else:
        version_idx = min(idx for idx in (dep_str.find(char) for char in ("<", ">", "=")) if idx != -1)
    pkg_idx = dep_str.find("[")
    extra_idx = dep_str.find("]")
    return {
        "pkg": dep_str[: pkg_idx if pkg_idx != -1 else version_idx],
        "version": dep_str[version_idx:],
        "extras": [] if extra_idx == -1 else dep_str[pkg_idx + 1 : extra_idx].split(","),
    }


def parse_setup_uv_version(
    minimum_uv_version: tuple[int, ...],
) -> tuple[list[dict[str, str]], list[str]]:
    dependencies = []
    troubles = []
    setup_file = Path(".github/actions/setup/action.yml")
    with setup_file.open(encoding="utf-8") as f:
        setup = yaml.safe_load(f)
    uv_steps = [step for step in setup["runs"]["steps"] if step["uses"].startswith("astral-sh/setup-uv@")]
    if len(uv_steps) != 1:
        return [], [f"{setup_file}: expected one setup-uv step"]
    uv_version = str(uv_steps[0]["with"]["version"]).lstrip("v")
    dependencies.append({"file": str(setup_file), "version": f"=={uv_version}"})
    if tuple(map(int, uv_version.split("."))) < minimum_uv_version:
        troubles.append(f"{setup_file}: uv must be at least {MINIMUM_UV_VERSION}")
    return dependencies, troubles


def main():
    # Retrieve & parse all deps files
    deps_dict = {
        "uv": [],
        "ty": [],
        "pre-commit": [],
    }
    troubles = []
    minimum_uv_version = tuple(map(int, MINIMUM_UV_VERSION.split(".")))
    expected_uv_build_version = EXPECTED_UV_BUILD_REQUIREMENT.removeprefix("uv_build")
    # Parse precommit
    with Path(PRECOMMIT_CONFIG).open("r", encoding="utf-8") as f:
        precommit = yaml.safe_load(f)
    for repo in precommit["repos"]:
        if repo["repo"] == "https://github.com/astral-sh/uv-pre-commit":
            deps_dict["uv"].append({"file": PRECOMMIT_CONFIG, "version": f"=={repo['rev'].lstrip('v')}"})
    # Parse pyproject.toml
    for pyproject_path in PYPROJECTS:
        with Path(pyproject_path).open("rb") as f:
            pyproject = tomllib.load(f)

        # Check the build backend version used by this project.
        build_deps = [parse_dep_str(dep) for dep in pyproject["build-system"]["requires"]]
        uv_build_deps = [dep for dep in build_deps if dep["pkg"] == "uv_build"]
        if len(uv_build_deps) != 1 or uv_build_deps[0]["version"] != expected_uv_build_version:
            troubles.append(f"{pyproject_path}: build-system requires {EXPECTED_UV_BUILD_REQUIREMENT}")

        # Parse dependencies
        core_deps = [parse_dep_str(dep) for dep in pyproject["project"]["dependencies"]]
        core_deps = {dep["pkg"]: dep for dep in core_deps}
        for dep in deps_dict:  # noqa: PLC0206
            if dep in core_deps:
                deps_dict[dep].append({"file": pyproject_path, "version": core_deps[dep]["version"]})

        # Parse optional dependencies
        quality_deps = [parse_dep_str(dep) for dep in pyproject["project"]["optional-dependencies"]["quality"]]
        quality_deps = {dep["pkg"]: dep for dep in quality_deps}
        for dep in deps_dict:  # noqa: PLC0206
            if dep in quality_deps:
                deps_dict[dep].append({"file": pyproject_path, "version": quality_deps[dep]["version"]})

    # Parse github/workflows/...
    setup_uv_dependencies, setup_troubles = parse_setup_uv_version(minimum_uv_version)
    deps_dict["uv"].extend(setup_uv_dependencies)
    troubles.extend(setup_troubles)

    # Assert all deps are in sync
    for dep, versions in deps_dict.items():
        versions_ = {v["version"] for v in versions}
        if len(versions_) > 1:
            inv_dict = {v: set() for v in versions_}
            for version in versions:
                inv_dict[version["version"]].add(version["file"])
            troubles.extend([
                f"\033[31m{dep}\033[0m:",
                "\n".join(f"- '{v}': {', '.join(files)}" for v, files in inv_dict.items()),
            ])

    if len(troubles) > 0:
        raise AssertionError("Some dependencies are out of sync:\n\n" + "\n".join(troubles))
    logger.info("\033[32mAll dependencies are in sync!\033[0m")


if __name__ == "__main__":
    main()
