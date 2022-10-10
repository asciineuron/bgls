# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup, find_packages


with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("dev_requirements.txt") as f:
    dev_requirements = f.read().splitlines()

# Save the source code in _version.py.
with open("bgls/_version.py", "r") as f:
    version_file_source = f.read()

# Overwrite _version.py in the source distribution.
with open("bgls/_version.py", "w") as f:
    f.write(f"__version__ = '{__version__}'\n")

setup(
    name="bgls",
    version=__version__,
    install_requires=requirements,
    extras_require={"development": set(dev_requirements)},
    packages=find_packages(),
    include_package_data=True,
    description="Implementation of the gate-by-gate sampling algorithm.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Alex Shapiro",
    author_email="alexander.shapiro@epfl.ch",
    license="Apache 2.0",
    url="https://github.com/asciineuron/bgls",
    project_urls={
        "Bug Tracker": "https://github.com/asciineuron/bgls/issues/",
        "Documentation": "https://github.com/asciineuron/bgls",
        "Source": "https://github.com/asciineuron/bgls",
    },
    python_requires=">=3.7",
)

# Restore _version.py to its previous state.
with open("bgls/_version.py", "w") as f:
    f.write(version_file_source)
