# libsparsr_hdc

Hyperdimensional Computing / Vector Symbolic Architecture on Sparsr: `hdc_bind`,
`hdc_bundle`, `hdc_similarity` and `hdc_train`, each executed as wide instructions on a
Sparsr device.

Plain C, on the public host ABI. It runs on the Sparsr VM today, and on the card when the
card is ready. Building it takes a C compiler and the RISC-V toolchain, nothing from the
FPGA flow.

```c
#include <sparsr_hdc.h>

hdc_init();

uint64_t seed = 1;
hdc_hypervector colour, red, record, recovered;
hdc_random(&colour, 20, &seed);
hdc_random(&red, 20, &seed);

hdc_bind(&colour, &red, &record);           /* one WXOR on the device */
hdc_unbind(&record, &colour, &recovered);   /* XOR is its own inverse */

hdc_shutdown();
```

Building it takes an unpacked Sparsr SDK tarball, the free download from the Developer Zone,
and a RISC-V bare-metal toolchain for the kernels. The SDK's `include/`, `ldscripts/`,
`startup/` and `lib/` are what the Makefile reads, and `SPARSR_SDK_ROOT` is where it looks:

```sh
make SPARSR_SDK_ROOT=/path/to/sparsr-sdk all     # builds build/libsparsr_hdc.so
make SPARSR_SDK_ROOT=/path/to/sparsr-sdk test    # builds and runs the harness on the VM
```

## What it runs on

Whatever `libsparsr_host` is pointed at, through the `SPARSR_BACKEND` environment variable
— the same host library and the same selection rule as any other Sparsr program. It links
`-lsparsr_host` and includes `sparsr.h`, and it knows nothing about which device is on the
other side.

**`SPARSR_BACKEND=vm` is required today.** The kernels are RV32I, and the default `softemu`
backend executes MIPS words; it would read these images as something else entirely. That is
not special to this library — it is the rule for every C kernel, and it goes away when the
assembler and the hardware catch up.

## Kernels are loaded once

`hdc_init()` compiles nothing and sends three kernel images into instruction memory, once.
Every operation after that writes its data and starts a batch.

This is worth the bookkeeping because a frame is a fixed 256 words whether the kernel in it
is seven words or two hundred. Loading a kernel per call spends a whole frame re-sending
instructions the device already has:

| Operation | Frames, kernel per call | Frames, kernel pre-loaded |
| --- | --- | --- |
| `hdc_bind` | 5 | 4 |
| `hdc_similarity` | 5 | 4 |
| `hdc_classify`, N classes | 5N | 4N |

Counting a similarity on the device does not remove one of its four frames — the host still
has to collect the answer. What changes is what that frame carries: four bytes of data memory
instead of a 240-byte compressed row.

On the emulated link that is one transfer in five. On a card it is a round trip that bought
nothing.

Bundling gets the larger share of it. The bundle kernel is a **loop**: the host writes the
operand count into data memory and the kernel folds that many rows, so one pre-loaded
program serves a bundle of any size instead of a differently unrolled program per count.
It needs a CMEM row that varies at run time, which is the register-indirect wide load —
the addressing that used to be the software-emulator-only `WLR`, and since the wide
encoding was frozen is simply what the one wide load does when its base register is not
`x0`.

The kernels themselves are ordinary C, in `src/device/`, compiled by the stock RISC-V
bare-metal GCC against the SDK's `sparsr_intrinsics.h` exactly as the SDK's own C-kernel
example builds. Nothing here hand-encodes an instruction. The
images are then embedded in the library as word arrays, so an installed library has no data
directory to find at run time.

### What it reserves, and why that is a problem

**This library takes CMEM rows 0 to 31 — every row there is — and the front of instruction
memory, from `hdc_init()` until `hdc_shutdown()`.** `src/device/hdc_device_layout.h` is the
whole of that contract.

Nothing arbitrates it. There is no allocator for CMEM rows and none for instruction memory,
so an application's own kernel loaded at word 0, or any other library keeping data in those
rows, would be overwritten with no error on either side. Pre-loading makes that worse rather
than better, because the library then holds instruction memory permanently. A host-side
memory manager would fix it. It is planned and not built.

`torchhd-sparsr` used to be exactly that collision: it kept PyTorch tensors resident in these
same rows. That was settled by making that package a caller of this library rather than a
peer of it, so it now holds its tensors on the
host and this library is the only owner of CMEM. That is a workaround for the missing
allocator, not a replacement for it — the cost is a host round trip per operation, which is
what the memory manager would remove.

## The algebra, and what it costs

This is **sparse binary VSA**, one of the two regimes a compressed CMEM row can hold.
Binding is XOR and bundling is the union, which needs no tiebreak and costs one instruction
per member.

The better-known Binary Spatter Code defines bundling as a **majority vote** instead. This
library does not implement that, and the reason is worth stating precisely, because the
obvious reason is wrong.

It is *not* that dense data cannot be stored. A row charges for non-zero **lanes**, and it
stores each occupied lane whole — so bits inside an occupied lane are free, and a fully
dense code confined to 48 lanes fits at any density. `hdc_random_dense()` builds exactly
that. This corrects what an earlier draft of this file said; the measurement that pinned it
down came from the MNIST workload.

The real blocker is the majority itself. Counting how many of N hypervectors set each of
4096 positions, then thresholding at N/2, is not something Sparsr has an instruction for.

**The three wide-writing reduce modes are built** — `SEGCOUNT`, `PREFIXRANK` and
`THRESHACC` — **and they are the wrong shape.** They count per 32-bit lane, so they yield 128 counters. A majority needs
4,096 of them, one per bit position, each several bits wide; that does not fit in a wide
register and has to live as bit-planes. So those modes serve the segmented workloads that
asked for them, and not this one.

So `hdc_bundle_majority()` builds the counters out of the instructions that do exist, as
bit-planes with a carry-save adder: plane *j* holds bit *j* of all 4096 counters, and adding
a member is a ripple of full adders across the planes. Thirteen planes count to 8191, which
is the largest bundle it will take.

**Thirteen and not eleven, because a class is thousands of examples.** A prototype is one
vote over every training image of that class, and MNIST's largest digit has 6,742 of them.
Eleven planes stopped at 2,047, which is not a training set. Worse, a bit-plane counter that
overflows *wraps* rather than saturating -- a count of 6,000 read back as 1,904, fell under
the threshold, and would have produced an empty prototype with nothing to say it went wrong.
The host refuses anything past `HDC_MAJORITY_MAX_MEMBERS` so that cannot happen, but the
ceiling has to be high enough to be useful first.

### What it costs

**Every number below counts every instruction the device retired, not only the wide ones.**
The scalar loop around the ripple is real work, and reporting only the wide instructions is
the flattering way to present this rather than the honest one. `bench/` is the harness and
[its README](bench/README.md) explains the method; the short version is that it fits a slope
through several member counts instead of dividing one run's total, so the reset pass, the
threshold pass and the per-batch overhead do not get charged to the members.

| | instructions per member |
| --- | --- |
| `hdc_bundle` — union | **4.00** |
| `hdc_bundle_majority` — majority, 80-member bundle | **7.84** |
| `hdc_bundle_majority` — majority, 1,000-member bundle | **12.96** |

**Both of those majority figures used to be 28.84.** Two changes brought them down:

| majority kernel | 80 members | 80 dense members | 1,000 members |
| --- | --- | --- | --- |
| hand-built ripple, runs every plane | 28.84 | 28.84 | 28.96 |
| `WCSA`, runs every plane | 15.84 | 15.84 | 15.96 |
| `WCSA`, stops when the carry empties | **7.84** | **12.84** | **12.96** |

- **`WCSA` halves the ripple.** A full-adder stage is `carry = plane AND carry_in` and
  `plane = plane XOR carry_in` over one pair of operands, and that is exactly what `WCSA`
  computes. Thirteen planes, one
  instruction saved per plane: 28.84 − 13 = 15.84, which is what the bench reports. The
  answer is unchanged; only the count moves.
- **Stopping when the carry empties saves the rest, and how much depends on the bundle.**
  `_sparsr_wany` is one OR-reduce and the kernel asks every third
  plane. Four periods were built and measured; every third was best or within 2% of best in
  every column. The kernel's own comment carries that table.

**Why the majority cost depends on the bundle and the union's does not.** The union looks at
no data — one load and one OR per member, always. The majority's ripple stops when the carry
runs out, and the carry a member injects is the member itself, halving per plane. A long
bundle has larger counters, so its upper planes are populated and the carry runs deeper. A
dense member injects a heavier carry than a sparse one, for the same reason.

**A 2 to 3x ratio against the union is still the case for building the reduce unit**, and it
is the number that case was missing. It is a
smaller case than the 13.5x this file used to quote, because two thirds of that gap turned
out to be software. A threshold-accumulate mode would make the inner loop one instruction per
member.

One saving is left and it needs no new instruction:

- **Sizing the planes to the bundle.** Thirteen planes are carried even when the caller
  bundles nine members, where four would do. Bundling N members needs `ceil(log2(N+1))`
  planes, so this is a straight saving whenever N is small — just entry points the host picks
  between. Widening the counters to hold a real class made this worth more than it was:
  encoding one MNIST image is a vote over about 150 members, which needs eight planes and is
  charged for thirteen. The early exit already collects part of it, since planes above the
  counters' real width carry nothing and are skipped.

The counter planes live in **wide registers**, not CMEM rows. A plane is dense by
construction — about half its bits set, whatever the inputs looked like — so there is
nowhere else for it to go. That turns out to be the better place: a batch is a call and not
a reset, so the planes survive from one batch to the next and a bundle longer than CMEM
holds spends no rows on the accumulator at all.

**It is also the one operation here that reads a status word back.** Similarity reads data
memory too, but for its result rather than for a verdict — its kernel holds no store to a
compressed row and so has nothing that can fault. The other three predict a row overflow on
the host before sending anything, from a lane-occupancy map. A majority cannot be predicted
that way: its lane occupancy is not a function of its members' lane occupancy, so the only
way to know is to compute the vote, which is what the device is for.
`test_a_vote_too_dense_to_store_is_reported` is the worked case — three members each
occupying exactly 48 lanes, voting to something that occupies 72.

**Which regime to use is a real choice, and sparse is not always the better half.** Measured
on MNIST, 6000 train / 1000 test:

| Code | Fits a row? | Accuracy |
| --- | --- | --- |
| Dense BSC, 1536 bits (48 lanes) | yes | 65.3% |
| Dense BSC, 3072 bits | no | 74.8% |
| Sparse block code, B=32 | yes | 37.6% |

So sparse buys exact, cheap bundling, and dense buys accuracy. This library gives you both:
`hdc_random()` draws the sparse regime and `hdc_random_dense()` the dense one. Thinning does
not rescue the sparse path.

**The dense row above is a floor, not a ceiling.** `examples/mnist/` reaches
**75.6%** at the same 1,536 bits and the same 6,000 train / 1,000 test, and **79.6%** on the
full 60,000 training and 10,000 test images. So most of the gap is the encoding, not the data
volume: the example superposes with `hdc_bundle_majority()` at every step -- encoding an
image and building a class prototype are the same call -- instead of thinning a union that
has already saturated. Read the 65.3% as what that particular encoding managed, not as what
the dense regime can do.

- **Bind is `WXOR`.** Commutative, associative, its own inverse. `hdc_unbind` is the same
  call under the name that says what you meant.
- **Bundle is `WOR`.** Every member stays *exactly* contained in the bundle, so
  `hdc_similarity(member, bundle).overlap` equals the member's own weight. In BSC that
  containment is only statistical.
- **Similarity is one instruction.** `WAND` and its population count are the same wide
  instruction, because the reduce is a mode on the ALU's result bus rather than a second
  operation. It writes an ordinary 32-bit register, the kernel stores that to data memory,
  and the host reads four bytes.

  This used to be the most expensive thing the library did: the device ANDed the operands
  into a CMEM row and the host read all 4,096 bits back to count them. The device was never
  what stopped it — the reduce mode ran on the Sparsr VM well before C could ask for it.
  The SDK was: `SPARSR_WOP_R` pinned the mode field to zero, so the header defined
  `SPARSR_WMODE_POPCOUNT` and could emit nothing that used it. A later SDK release added
  `_sparsr_wreduce` and the named helpers over it, and this library then used
  `_sparsr_woverlap` here. The
  library never hand-rolled the `.insn` to get there early, which is why there is still no
  second instruction encoder in the tree.

  **The two operand weights `hdc_similarity_result` also reports stay on the host**, and that
  is a decision rather than what is left over. They count bits of hypervectors the caller
  already holds and the host has just touched to write the rows. Computing them on the device
  would add two wide instructions and two data words while removing no transfer. Only the
  overlap needs the device, because only the overlap needs both vectors in one place.

  **A backend that runs the wide ALU may still not implement the reduce.** The four
  scalar-writing modes exist on the Sparsr VM and the RTL has none of them, so such a device
  would run bind and bundle correctly, fault on the reduce, and leave every similarity
  reporting whatever data memory held. That is a property of the backend rather than of the
  call, so `hdc_init()` probes it once with a pair whose overlap, and both weights, are
  different known non-zero numbers — and returns `HDC_ERROR_DEVICE` rather than letting the
  library come up. One probe per process instead of a status word per similarity.

**XOR binding does not decorrelate sparse hypervectors.** This is the real cost of the
choice, and the library states it rather than hiding it. Two sparse vectors rarely share a
bit, so their XOR is very nearly their union — and a union stays strongly similar to both
operands. In dense BSC a bound pair looks like noise to both, which is what lets one bundle
hold many bound pairs and be probed for each one. Getting that here needs an operation that
*moves* bits instead of combining them: a wide rotate, which is planned and not built.
Until then, treat `hdc_bind` as an
exact reversible pairing, and keep bundles small enough that the union has not swallowed the
distinctions you care about. A test records this so it cannot quietly stop being true.

## Capacity is the limit you will hit

A CMEM row holds a 4096-bit vector compressed into 240 bytes as at most **48 non-zero
four-byte lanes** out of 128. The limit counts *lanes*, never set bits, and each occupied
lane is stored whole. So:

- A hypervector with at most 48 set bits always fits, whatever the bits are, because each set
  bit makes at most one lane non-zero. `hdc_random(&v, weight, &seed)` with `weight <= 48` is
  always storable. That is the sparse regime.
- A hypervector of **any** density fits as long as its bits stay inside 48 lanes — 1536 of
  the 4096 bits. That is the dense regime, and `hdc_random_dense(&v, 48, &seed)` builds it.
  `hdc_random` cannot: it scatters bits over all 4096 positions, so asking it for 1536 set
  bits gives roughly 128 occupied lanes and will not fit.
- A **union grows**, so bundling enough members overflows the row. That returns
  `HDC_ERROR_TOO_DENSE`, predicted on the host from a 128-bit lane-occupancy map *before*
  anything is sent — not discovered afterwards.
- `hdc_bundle` splits itself. CMEM has 32 rows and one holds the accumulator, so more than 31
  members run as several batches; `WOR` is associative, so the answer is the same.

Predicting the overflow is not belt and braces, it is the only option: **the public host ABI
cannot report a device fault.** `sparsr_execute_batch` returns `void` and `sparsr_read_status`
returns one done bit, which a faulting batch sets exactly like a successful one. A library
that trusted it would hand back the accumulator row as it stood before the refused store — a
valid-looking hypervector that silently lost members. A fix to the ABI is planned; until
then this library does not put itself in a position to need it.

Lifting the 48-lane ceiling is the uncompressed CMEM path, which is planned and not built.

## What is here

| Path | What it is |
| --- | --- |
| `include/sparsr_hdc.h` | The public C API. The only header a caller includes. |
| `src/sparsr_hdc.c` | The host side: kernel loading, the four operations, the memory. |
| `src/device/*.c` | The kernels, in C, compiled for RV32I. |
| `src/device/hdc_device_layout.h` | The CMEM rows and registers the host and the kernels share. |
| `scripts/embed_kernels.py` | Turns the compiled images into a header the library compiles in. |
| `test/test_hdc.c` | The harness. Runs every operation on a real device. |
| `examples/mnist/` | MNIST end to end on this library. Builds against the repo, not the SDK. |
| `packaging/README.md` | The README packaged inside the HDC tarball. |
| `tarball.tests/` | Builds a C program against the extracted HDC tarball and nothing else, and runs it on the VM. |
| `LICENSE`, `LICENSE-RUNTIME` | MIT for the library, and the terms for the runtime the tarball bundles. |

There is no mock in the tests, on purpose: the operations under test *are* wide
instructions, and a host-side reimplementation of them would test nothing.

## The worked example

`examples/mnist/` is this library end to end: handwritten digit recognition at **79.6% on the
full 10,000-image MNIST test set**, trained on all 60,000 training images, with every vector
operation running on the device. It is also the best worked argument for the dense regime —
the whole example lives inside 48 lanes by construction, so nothing in it can ever be too
dense to store.

```bash
make SPARSR_SDK_ROOT=/path/to/sparsr-sdk all
make -C examples/mnist SPARSR_SDK_ROOT=/path/to/sparsr-sdk run MNIST_DIR=/path/to/mnist
```

`make check` in that directory builds the example and confirms it reports missing data
correctly. That is what `hdc-tests.yml` runs, because CI has no MNIST files — they are about
55 MB and are in no repository.

**Someone *using* this library needs no RISC-V toolchain.** The kernels are compiled into the
`.so` when the library is built, so the toolchain is a prerequisite of building this
directory, not of linking against it.

### Why the example is here and not among the SDK's examples

Because the SDK's examples are staged into the SDK tarball, and **this library is deliberately
not in it**. The decision, 2026-08-27: that tarball is the Sparsr *Kernel* SDK, and the audience
for a high-level library is not the audience for kernel development, so the two ship
separately rather than being bundled into one download.

An earlier revision did package
`lib/libsparsr_hdc.so` and `include/sparsr_hdc.h` and put the example among the SDK's. That
was reversed. Two consequences worth knowing:

- **This example builds against this library's own build directory** and takes the runtime
  from the unpacked SDK `SPARSR_SDK_ROOT` names; the SDK's own examples build against an
  unpacked SDK alone.
- **This whole directory is held to the same content rules as the SDK tarball**, since it
  is published on its own, and a test in the SDK's own suite fails if the library reappears
  in the tarball. Both decisions are checks rather than notes.

## Not in this first iteration

- **No permutation or sequence encoding.** Needs the wide rotate, which is planned and not
  built.
- **The counter planes are not sized to the bundle.** Thirteen are carried whatever the
  member count. See "What it costs" above; the early exit already collects part of this, and
  the rest needs entry points rather than any new instruction.
- **No hypervector wider than 48 lanes**, in either regime. Lifting that is the uncompressed
  CMEM path, planned and not built. It is what caps the
  MNIST example at 1,536 of the 4,096 bit positions.
- **Not in the SDK tarball, on purpose.** See "Why the example lives here" above. It is packaged
  as its own tarball instead, `sparsr-hdc-*.tar.gz`, packaged by Sparsr's release workflow
  with the library, its header and the runtime, and `tarball.tests/` proves a C
  caller can build against that and nothing else. Building the library itself needs an
  unpacked SDK, see the top of this file.
