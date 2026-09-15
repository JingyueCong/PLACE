import os
from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_SEARCH_ROOT = os.path.relpath(PACKAGE_ROOT, Path.cwd())


def read_requirements():
    return [
        line
        for raw_line in (PACKAGE_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]

setup(
    name="recap-uld",
    version="1.0.0",
    packages=find_packages(where=PACKAGE_SEARCH_ROOT),
    package_dir={"": PACKAGE_SEARCH_ROOT},
    install_requires=read_requirements(),
    python_requires=">=3.10,<3.12",
    description="RECAP assistant training for the TOFU benchmark",
    keywords="machine unlearning, language models, TOFU, RECAP",
)
