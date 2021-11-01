# Changing the documentation

The documentation of this project is built using `sphinx`. In order to install all the build dependencies, run the following command from the root folder of the repository:
```shell
pip install -e ".[docs]"
```

---
**NOTE**

You are only generating the documentation to inspect it locally. Only the source files are pushed to the remote repository, the documentation will be built automatically by the CI.

---

## Build the documentation

### Latest version

In most cases, you will only be changing the documentation of the latest version (dev version). In this case, you can build the documentation (the HTML files) with the following command:

```shell
sphinx-build docs/source docs/_build -a
```

Then open `docs/_build/index.html` in your web browser to navigate in it.


### Multi-version documentation

In rare cases, you might want to modify the documentation for other versions. You will then have to build the documentation for the multiple versions of the package, which you can do by running this command from the `docs` folder:
```shell
bash build.sh
```
