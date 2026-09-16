# Sparsr HDC

Hyperdimensional Computing and Vector Symbolic Architecture operations on the Sparsr
processor, from C or C++: `hdc_bind`, `hdc_bundle`, `hdc_similarity` and `hdc_train`. Each one
runs as wide instructions on a Sparsr device. This tarball is the library, its header, and the
runtime it needs. It works on its own.

```
include/sparsr_hdc.h     the API, and the only header you include
include/sparsr.h         the Sparsr host API the library sits on
lib/libsparsr_hdc.so     the library
lib/libsparsr_*.so       the Sparsr runtime: host library, VM, and two more backends
bin/sparsr-vm            the Sparsr VM as a program, for the out-of-process backend
LICENSE                  MIT, for the library and its header
LICENSE-RUNTIME          the terms for the runtime files
THIRD-PARTY-NOTICES      the .NET runtime's notices, for the two binaries that contain it
```

## Building against it

```c
#include <sparsr_hdc.h>

int main(void) {
    hdc_init();

    uint64_t seed = 1;
    hdc_hypervector colour, red, record, recovered;
    hdc_random(&colour, 20, &seed);
    hdc_random(&red, 20, &seed);

    hdc_bind(&colour, &red, &record);           /* one wide XOR on the device */
    hdc_unbind(&record, &colour, &recovered);   /* XOR is its own inverse */

    hdc_shutdown();
    return 0;
}
```

With the tarball unpacked at `$SPARSR_HDC`:

```sh
gcc -I$SPARSR_HDC/include app.c -L$SPARSR_HDC/lib -lsparsr_hdc -lsparsr_host -lm \
    -Wl,-rpath,$SPARSR_HDC/lib -o app
SPARSR_BACKEND=vm ./app
```

You need a C compiler and nothing else. The kernels the library runs on the device are compiled
into `libsparsr_hdc.so` already, so there is no RISC-V toolchain to install.

## The backend

`SPARSR_BACKEND=vm` is required today. The library's device kernels are RV32I, and the runtime's
default backend, `softemu`, executes a different instruction set. `hdc_init()` refuses to come
up on a backend that cannot run the kernels, so a wrong setting fails at start rather than
returning wrong answers.

`SPARSR_BACKEND=vmproc` runs the same VM in a process of its own, which the runtime starts from
`bin/sparsr-vm`. Keep `bin/` beside `lib/`: the backend finds the program at `../bin` relative to
its own library.

## The device is the library's while it is up

From `hdc_init()` to `hdc_shutdown()` the library uses WMEM rows 0 to 31 and the front of
instruction memory. A Sparsr device has no allocator for either today, so an application that
loads its own kernel or keeps data in those rows would overwrite the library's, with no error on
either side. Give the library the device to itself.

## Limits worth knowing before you start

- A hypervector has to fit a compressed WMEM row: at most 48 non-zero four-byte lanes out of
  128. Bits spread across all 4096 positions have to be sparse; bits confined to 48 lanes can be
  at any density. `hdc_fits_device()` tells you which side of the line a vector is on.
- `hdc_bundle` is the union, exact and cheap. `hdc_bundle_majority` is the majority vote, built
  from bit-plane counters, and it takes up to 8191 members.
- XOR binding does not decorrelate sparse hypervectors: their XOR is nearly their union, which
  stays similar to both operands. Treat `hdc_bind` as an exact, reversible pairing.

The header documents every call, every status code, and what each one costs.
