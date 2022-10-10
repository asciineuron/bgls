# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup, find_packages


with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("dev_requirements.txt") as f:
    dev_requirements = f.read().splitlines()

# Save the source code in _version.py.
with open("pypackage/_version.py", "r") as f:
    version_file_source = f.read()

# Overwrite _version.py in the source distribution.
with open("pypackage/_version.py", "w") as f:
    f.write(f"__version__ = '{__version__}'\n")

setup(
    name="pypackage",
    version=__version__,
    install_requires=requirements,
    extras_require={"development": set(dev_requirements)},
    packages=find_packages(),
    include_package_data=True,
    description="A template Python Package hosted on GitHub.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Ryan LaRose",
    author_email="rlarose@umich.edu",
    license="Apache 2.0",
    url="https://github.com/rmlarose/PyPackage/",
    project_urls={
        "Bug Tracker": "https://github.com/rmlarose/PyPackage/issues/",
        "Documentation": "https://github.com/rmlarose/PyPackage/",
        "Source": "https://github.com/rmlarose/PyPackage/",
    },
    python_requires=">=3.7",
)

# Restore _version.py to its previous state.
with open("pypackage/_version.py", "w") as f:
    f.write(version_file_source)
