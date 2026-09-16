# Recognising handwritten digits on Sparsr

This example classifies MNIST digits with **79.6% accuracy**, trained on all 60,000 training
images and tested on all 10,000 test images. It is one file of plain C, and every vector
operation in it runs on the Sparsr device.

There are no gradients, no training loop, and no floating-point arithmetic on the device.
Training is a single pass over the data. This is *hyperdimensional computing* (HDC), also
called a *vector symbolic architecture* (VSA): an image becomes one 4096-bit vector, a digit
class becomes one 4096-bit vector, and recognising a digit means finding the class vector
that the image vector resembles most.

## Run it

Build `libsparsr_hdc` first, then this example, both from this directory and both against
an unpacked Sparsr SDK:

```bash
make -C ../.. SPARSR_SDK_ROOT=/path/to/sparsr-sdk all
make SPARSR_SDK_ROOT=/path/to/sparsr-sdk
```

Then, from this directory. The MNIST files are about 55 MB and are in no repository, so you
supply them:

```bash
make run MNIST_DIR=/path/to/mnist
```

Or build and run the two steps separately:

```bash
make
SPARSR_BACKEND=vm ./build/mnist_hdc --data /path/to/mnist
```

A full run takes about half a minute. For a quick look, use a slice of the data:

```bash
SPARSR_BACKEND=vm ./build/mnist_hdc --data /path/to/mnist --train 6000 --test 1000
```

`make check` builds it and confirms it reports missing data correctly, which is what CI runs
on a machine that has no MNIST files.

### What you need

A host C compiler, and a checkout of this repository. **This example needs no RISC-V
toolchain**, unlike the SDK's own C-kernel example — the device programs it runs are
already compiled inside `libsparsr_hdc.so`, and the library loads them into instruction
memory for you. The toolchain is needed to *build* the library, not to use it.

**It builds against the repository, not against an unpacked SDK**, because this library is
not in the SDK. That tarball is the Sparsr *Kernel* SDK, and someone writing hyperdimensional
computing code is not the person it is for.

`SPARSR_BACKEND=vm` is required, and `make run` sets it. The library's device programs are
RV32I; the default `softemu` backend executes a different instruction set and would read them
as something else entirely.

### The data files

Four files, uncompressed:

```
train-images-idx3-ubyte   train-labels-idx1-ubyte
t10k-images-idx3-ubyte    t10k-labels-idx1-ubyte
```

Both common naming styles work — `train-images-idx3-ubyte` and `train-images.idx3-ubyte`. If
your copies end in `.gz`, `gunzip` them first. Point the program at the directory with
`--data`, or set `SPARSR_MNIST_DIR`.

## What it prints

```
Accuracy: 79.58%  (7958 of 10000 correct)

  digit   trained on   tested   correct   accuracy   prototype bits   lanes
      0         5923      980       882       90.0%              692      48
      1         6742     1135      1054       92.9%              694      48
      2         5958     1032       752       72.9%              714      48
      3         6131     1010       846       83.8%              733      48
      4         5842      982       771       78.5%              693      48
      5         5421      892       523       58.6%              701      48
      6         5918      958       804       83.9%              703      48
      7         6265     1028       839       81.6%              708      48
      8         5851      974       707       72.6%              703      48
      9         5949     1009       780       77.3%              696      48
```

Digit 1 is the easiest and digit 5 is the hardest, which is what an HDC classifier usually
finds on MNIST.

## The algorithm, in four steps

Each step is one call into `libsparsr_hdc`, and the library runs it as wide instructions on
the device.

**1. Item memory.** Give each of the 784 pixel positions a fixed random hypervector, drawn
once with `hdc_random_dense()` and never changed. Two positions get unrelated vectors, which
is what makes the encoding below say *which* pixels were lit.

**2. Encode.** An image is the **majority vote** of the hypervectors of its lit pixels,
computed by `hdc_bundle_majority()`. A bit of the result is set where more than half of those
pixels' vectors had it set. Two images of the same digit light similar pixels, so they encode
to similar vectors.

**3. Train.** A class prototype is the majority vote of every training image of that digit —
the same `hdc_bundle_majority()` call, over thousands of members instead of about 150. One
pass over the data, and each class is one vector.

**4. Classify.** Encode the test image, then score it against all ten prototypes with
`hdc_similarity()` and pick the best. Each score is **one wide instruction** — the
intersection and its population count are the same instruction, because the reduce is a mode
on the ALU's result bus. Nothing reads a 4096-bit row back to count its bits.

## Two things worth understanding before you write your own

### Why the codes are dense, and why they are 48 lanes wide

A hypervector reaches the device through a wide memory row, and **a row stores at
most 48 non-zero four-byte lanes out of 128**. The limit counts *lanes*, never set bits, and
it stores each occupied lane whole — so bits inside an occupied lane are free.

That makes two very different codes storable, and they are not equally good:

| Code | Built by | Set bits | Fits a row? |
| --- | --- | --- | --- |
| Sparse | `hdc_random(&v, 48, &seed)` | 48, scattered anywhere | always |
| Dense | `hdc_random_dense(&v, 48, &seed)` | about 768, inside 48 lanes | always |

This example uses the dense one, and it matters: a dense code carries about 768 bits of
signal where a sparse one carries 48, and it classifies far better. Every vector here — item,
encoded image, and prototype — stays inside those 48 lanes by construction, because a
majority vote can only set bits its members already had. That is why nothing in this example
ever comes back as `HDC_ERROR_TOO_DENSE`, and why the `lanes` column above reads 48 on every
row.

This is the constraint you will hit first when you write your own HDC code on Sparsr. Design
your encoding so its results stay storable, rather than discovering afterwards that they do
not.

### Why the vote, and not the union

`libsparsr_hdc` gives you two ways to superpose hypervectors, and picking the wrong one here
would produce a classifier that always guesses.

- `hdc_bundle()` is the **union**: OR every member together. Two wide instructions per
  member, exact, and it never loses a member. But it *grows*. OR a few hundred images of one
  digit together and every bit is set, so the prototype says nothing about that digit and
  every class looks identical.
- `hdc_bundle_majority()` is the **majority vote**. It stays informative however many members
  it has, which is exactly what a prototype needs.

The vote costs **8 to 13 instructions per member** against the union's 4 — more for a longer
bundle, because its ripple carry then runs deeper. That gap is not a quirk of the library.
Counting how many of N vectors set each of 4096 bit positions needs an instruction that
counts per bit position, and Sparsr's wide instruction set has none — so the device builds
4096 counters as bit-planes with a carry-save adder, one `WCSA` per plane, stopping once the
carry empties. The whole 60,000-image training set is one such vote per class, and it takes
0.2 seconds once the images are encoded. See [the library's cost
table](../../README.md#what-it-costs) for how those figures were measured.

## Options

| Option | Meaning |
| --- | --- |
| `--data DIR` | Where the MNIST files are. Falls back to `$SPARSR_MNIST_DIR`, then `./data`. |
| `--train N` | Training images to learn from. Default: all of them. |
| `--test N` | Test images to classify. Default: all of them. |
| `--lanes N` | Width of the code, 1 to 48 lanes. Default: 48. |
| `--seed N` | Seed for the item memory. Default: 1. |

`--lanes` is the interesting one. It is the density knob, and narrowing it costs accuracy
directly. Measured on 6,000 training and 1,000 test images:

| `--lanes` | Bits of the code | Accuracy |
| --- | --- | --- |
| 8 | 256 | 64.2% |
| 16 | 512 | 69.0% |
| 48 | 1536 | 75.6% |

48 is the widest a wide memory row can hold, so 1536 bits is the ceiling this
example can reach today.

```bash
SPARSR_BACKEND=vm ./build/mnist_hdc --data /path/to/mnist --lanes 8
```

Exit codes: `0` if it ran, `2` if the MNIST files were not found, `3` if something failed.
