# Rogers: Fast, lightweight neighbor-list construction on GPU
*Won't you be my neighbor?*

Rogers is a Python package for constructing neighbor lists for atomistic systems. It is written entirely in JAX, in a single small files, and is meant to plug into any existing molecular dynamics engine. Rogers offers blazingly fast speeds on GPU (despite the simple implementation), and for large systems and cutoffs it outcompetes some of the fastest existing tools.

⚠️ **NOTE:** Rogers is very fast on GPU but comparatively slow on CPU. For CPU purposes, I recommend using another library, such as the excellent [vesin](https://github.com/Luthaf/vesin).

## Benchmarks
🏗️ Under construction!

## Installation
🏗️ Under construction!

## Build Dependencies
* [Sphinx](https://github.com/sphinx-doc/sphinx) and [Furo](https://github.com/pradyunsg/furo) to build documentation
* [pytest](https://docs.pytest.org/en/stable/) for tests
