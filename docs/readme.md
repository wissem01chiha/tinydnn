# tinydnn documentation

A built version of documentation is available [here](http://tiny-dnn.readthedocs.io/en/latest/index.html).

## Local build

You can build html documents in your local machine if you prefer.
Assuming you have python already, install Sphinx and recommonmark at first:

```bash
$ pip install sphinx sphinx-autobuild
$ pip install recommonmark
```

#### Build on Windows
```bach
cd docs
make.bat html
```

#### Build on Linux
```bash
cd docs
make html
```