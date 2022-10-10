# PyPackage

[![build](https://github.com/rmlarose/PyPackage/workflows/Python%20package/badge.svg)](https://github.com/rmlarose/PyPackage/actions)
[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github)](https://github.com/rmlarose/PyPackage)

PyPackage is a template for a Python package hosted on GitHub. Use it to create your own Python package.

## Instructions

### Get source files

Generate your repository from this template at https://github.com/rmlarose/PyPackage/generate.

### Make it your own

1. Refactor the `PyPackage` directory name to your package name.
    - **Recomended**: Use an IDE refactoring tool, e.g. PyCharm's refactor. This will update `import`s and paths in `setup.py`.

1. Update `setup.py` with your package's name, description, author, URLs, etc.

1. Try installing your package and make sure it still works.
    - Run `python -m pip install -e .` in the directory with `setup.py`.
    - In `python`, do `import <your package name>`.
    - *Tip*: You may need to "restart" your virtual environment to try another install. If so, run `pip freeze | xargs pip uninstall -y`.

1. Update the GitHub actions in `.github`.
    - Change `PyPackage` to your package name.

1. Update your mypy config, changing `exclude = <your package name>.*_test.py`.

1. Update your `README`.
    - Don't forget to update the badge links to your repository.

1. Push to GitHub and make sure your actions are still working properly.

1. Code away!
    - Scrap the simple `Object` and tests, add useful code, rename `module`, and add other modules.

## Questions, comments?

Did you try the above steps? Did they work? Did you have to do something different? Have any tips? Please let me know or edit this README in a pull request to make it easier for others.

