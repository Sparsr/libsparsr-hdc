# Contributing

Thank you for looking at this library. A few things about how it is built and checked.

## Building

The library builds against an unpacked Sparsr SDK, the free download from the
[Sparsr Developer Zone](https://developers.sparsr.com/), and needs a RISC-V bare-metal
toolchain for its device kernels:

```sh
make SPARSR_SDK_ROOT=/path/to/sparsr-sdk all
make SPARSR_SDK_ROOT=/path/to/sparsr-sdk test
```

`README.md` has the rest.

## Pull requests

Open one against `main` and fill in the template. Keep a change to one thing. Squash on
merge: the history here is meant to be read.

## Continuous integration

CI fetches the SDK from a Sparsr release with a token this repository holds. A pull request
from a fork does not have that token, so its full check cannot run and reports why. A
maintainer runs it by pushing your branch to this repository. If you would rather run the
check yourself, register at the Developer Zone for the SDK and use the two commands above.
