/**
 * @file sparsr_hdc.h
 * @brief Hyperdimensional Computing / Vector Symbolic Architecture on a Sparsr device.
 *
 * Four operations -- bind, bundle, similarity and train -- each executed as wide
 * instructions on Sparsr through libsparsr_host. Everything the device can do, the device
 * does -- counting included. A similarity is one instruction that reduces to a scalar, so
 * nothing reads a 4096-bit row back to the host just to count its bits.
 *
 * @section usage Using it
 *
 * @code
 * #include <sparsr_hdc.h>
 *
 * hdc_init();
 *
 * uint64_t seed = 1;
 * hdc_hypervector colour, red, record, recovered;
 * hdc_random(&colour, 20, &seed);
 * hdc_random(&red, 20, &seed);
 *
 * hdc_bind(&colour, &red, &record);        // one WXOR on the device
 * hdc_unbind(&record, &colour, &recovered); // XOR is its own inverse
 *
 * hdc_shutdown();
 * @endcode
 *
 * @section backend Which device it runs on
 *
 * Whichever one libsparsr_host is pointed at, through the `SPARSR_BACKEND` environment
 * variable. **The kernels are RV32I, so `SPARSR_BACKEND=vm` is required today**: the
 * default `softemu` backend executes MIPS words and would read these images as something
 * else entirely. That is not a limitation of this library -- it is the same rule as any
 * other C kernel, and it goes away when the assembler and the hardware catch up.
 *
 * @section algebra The algebra, and what it costs
 *
 * There are **two superposition operators here, and picking the right one is the first
 * decision you make**.
 *
 * - ::hdc_bundle is the **union**: OR every member together. Exact, no tiebreak, four
 *   instructions per member. It never loses a member -- but it grows, so a few hundred
 *   members saturate it and the result stops saying anything.
 * - ::hdc_bundle_majority is the **majority vote** of Binary Spatter Code: keep a bit where
 *   more than half the members had it. It stays informative however many members it has,
 *   which is what a class prototype needs, and it costs 8 to 13 instructions per member --
 *   more for a longer bundle, because its ripple carry then runs deeper.
 *
 * That 2 to 3x ratio is not incidental. A majority needs a count per bit position, and there are
 * 4096 of them. Sparsr has no instruction that counts that way. The wide-writing reduce
 * modes do count, but per 32-bit lane, which yields 128 counters rather than 4096 -- the
 * right tool for a segmented workload and the wrong one here. So the device builds the
 * counters out of AND, XOR and OR as bit-planes with a carry-save adder.
 *
 * **Sparse and dense are both storable, and which you want is a real choice.** What a
 * compressed row charges for is non-zero *lanes*, never set bits, and it stores each occupied
 * lane whole -- so bits inside an occupied lane are free, and a fully dense code confined to
 * 48 lanes fits at any density. ::hdc_random builds the sparse regime and ::hdc_random_dense
 * the dense one. Measured on MNIST: a dense code at 1536 bits -- the widest that fits a row --
 * classifies far better than a sparse block code of the same capacity. Sparse buys exact,
 * cheap bundling; dense buys accuracy. `examples/mnist` beside this library is the dense half
 * end to end.
 *
 * **What it costs: XOR over sparse hypervectors does not decorrelate them.** Two sparse
 * vectors rarely share a bit, so their XOR is very nearly their union, and a union stays
 * strongly similar to both operands. In dense BSC a bound pair looks like noise to both,
 * which is what lets one bundle hold many bound pairs and be probed for each. Getting that
 * here needs an operation that moves bits rather than combining them -- a wide rotate,
 * which Sparsr does not have yet. So treat hdc_bind() as an exact, reversible pairing, and
 * keep bundles small enough that the union has not swallowed the distinctions you need.
 *
 * @section capacity Capacity is the limit you will actually hit
 *
 * A WMEM row holds a 4096-bit vector compressed into 240 bytes as at most
 * ::HDC_MAX_STORABLE_LANES non-zero four-byte lanes out of 128. **The limit counts lanes,
 * never set bits**, and each occupied lane is stored whole. Two consequences, and the
 * second one is easy to miss:
 *
 * - A hypervector with 48 or fewer set bits always fits, whatever those bits are, because
 *   each set bit makes at most one lane non-zero. That is the sparse regime.
 * - A hypervector of *any* density fits as long as its set bits stay inside 48 lanes, which
 *   is 1536 of the 4096 bits. That is the dense regime, and ::hdc_random_dense builds it.
 *
 * A union grows, so bundling enough members will eventually spread past 48 lanes -- and
 * that is reported as ::HDC_ERROR_TOO_DENSE rather than returning a truncated hypervector
 * that looks perfectly valid.
 */

#ifndef SPARSR_HDC_H
#define SPARSR_HDC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Bits in a hypervector: the width of a Sparsr wide register. */
#define HDC_HYPERVECTOR_BITS 4096

/** Bytes in a hypervector. */
#define HDC_HYPERVECTOR_BYTES (HDC_HYPERVECTOR_BITS / 8)

/** Four-byte lanes a hypervector is divided into for compression. */
#define HDC_LANES (HDC_HYPERVECTOR_BYTES / 4)

/**
 * Non-zero lanes that fit one WMEM row. A denser hypervector cannot be stored on the
 * device at all, and every entry point that would have to store one refuses instead.
 */
#define HDC_MAX_STORABLE_LANES 48

/**
 * One 4096-bit hypervector, held on the host.
 *
 * Bit `i` is bit `i % 8` of byte `i / 8`, counting from the least significant bit. The
 * device treats the 512 bytes as an opaque block for every bitwise instruction, so this
 * ordering only has to be consistent -- but it has to be stated, because the population
 * counts this library reports depend on it.
 *
 * The struct is public and copyable on purpose: a hypervector is a value, and a caller
 * should be able to keep an example and the prototype it was bundled into side by side
 * without asking this library to manage either.
 */
typedef struct hdc_hypervector {
    uint8_t bytes[HDC_HYPERVECTOR_BYTES];
} hdc_hypervector;

/** How much two hypervectors overlap, as counts rather than as one number. */
typedef struct hdc_similarity_result {
    /** Bits set in both. This is what the device computed. */
    uint32_t overlap;
    /** Bits set in the first. */
    uint32_t left_weight;
    /** Bits set in the second. */
    uint32_t right_weight;
} hdc_similarity_result;

/** What went wrong, or ::HDC_OK. */
typedef enum hdc_status {
    HDC_OK = 0,
    /** hdc_init() has not run, or it failed. */
    HDC_ERROR_NOT_INITIALIZED,
    /** A null pointer, an out-of-range bit, or an empty set of vectors. */
    HDC_ERROR_INVALID_ARGUMENT,
    /** A hypervector needs more than ::HDC_MAX_STORABLE_LANES lanes, so the device cannot hold it. */
    HDC_ERROR_TOO_DENSE,
    /** More classes or labels than this library will track. */
    HDC_ERROR_CAPACITY,
    /** The device or the host library refused something. */
    HDC_ERROR_DEVICE,
    /** Out of host memory. */
    HDC_ERROR_OUT_OF_MEMORY
} hdc_status;

/** A short, stable description of a status code. Never null. */
const char *hdc_status_string(hdc_status status);

/**
 * Prepares the device and loads this library's kernels into instruction memory.
 *
 * The kernels stay loaded until hdc_shutdown(), so an operation costs its data and nothing
 * else. Call this once, before anything else here.
 *
 * It then asks the device one question and returns ::HDC_ERROR_DEVICE if the answer is
 * wrong: a similarity is a population-count reduce, and a backend can run every other wide
 * instruction without implementing that mode. Such a device would answer every similarity
 * with whatever data memory held, so this is checked once here rather than trusted.
 *
 * @warning It calls `sparsr_kernel_init()`, which clears every memory on the device, and
 * it then reserves WMEM rows 0 to 31 and the front of instruction memory for its own use.
 * Nothing arbitrates that today -- another library holding data in those rows will have it
 * overwritten, with no error on either side. A host-side memory manager that fixes it is
 * planned and not built.
 */
hdc_status hdc_init(void);

/** Releases what hdc_init() claimed. Safe to call when init never ran. */
void hdc_shutdown(void);

/** Sets every bit to zero. The zero hypervector is the identity of both bind and bundle. */
void hdc_zero(hdc_hypervector *vector);

/** Sets one bit. Returns ::HDC_ERROR_INVALID_ARGUMENT if @p bit is not below ::HDC_HYPERVECTOR_BITS. */
hdc_status hdc_set_bit(hdc_hypervector *vector, uint32_t bit);

/** Whether one bit is set. Zero for an out-of-range bit. */
int hdc_get_bit(const hdc_hypervector *vector, uint32_t bit);

/** Set bits. In sparse binary VSA this is the vector's weight. */
uint32_t hdc_weight(const hdc_hypervector *vector);

/** Non-zero four-byte lanes. This, not the weight, is what the compressed row charges for. */
uint32_t hdc_nonzero_lanes(const hdc_hypervector *vector);

/** Whether the device can hold this hypervector at all. */
int hdc_fits_device(const hdc_hypervector *vector);

/**
 * Draws a hypervector with exactly @p weight set bits, chosen uniformly without
 * replacement. This is how a caller mints an atomic symbol.
 *
 * Keep @p weight at or below ::HDC_MAX_STORABLE_LANES and the result always fits a WMEM
 * row, since each set bit makes at most one lane non-zero. Ask for more and it may still
 * fit, but nothing guarantees it -- check hdc_fits_device().
 *
 * @param seed  Caller-owned generator state, advanced by the call. Seeding it the same way
 *              twice gives the same draws, which is what makes a test reproducible. The
 *              generator is deliberately this library's own rather than `rand()`, so it
 *              neither depends on nor disturbs process-wide state.
 */
hdc_status hdc_random(hdc_hypervector *out, uint32_t weight, uint64_t *seed);

/**
 * Draws a **dense** hypervector confined to the first @p lanes lanes: every bit inside
 * them at even odds, every bit outside them zero.
 *
 * This is the other storable regime, and hdc_random() cannot reach it -- that one scatters
 * its bits over all 4096 positions, so asking it for 1536 set bits gives roughly 128
 * occupied lanes and will not fit. Here the width is `lanes * 32` bits and the lane count
 * is @p lanes by construction, so any @p lanes at or below ::HDC_MAX_STORABLE_LANES is
 * always storable however dense the result is.
 *
 * Use it when accuracy matters more than the cost of bundling. Measured on MNIST, a dense
 * code at the widest that fits (48 lanes, 1536 bits) classifies far better than a sparse
 * code of the same capacity. What it does not come with is a matching bundle: superposing
 * dense codes needs a majority vote, which hdc_bundle() does not do and Sparsr has no
 * instruction for. hdc_bundle() will happily OR them and saturate.
 *
 * @param lanes How many four-byte lanes the code occupies, 1 to ::HDC_MAX_STORABLE_LANES.
 */
hdc_status hdc_random_dense(hdc_hypervector *out, uint32_t lanes, uint64_t *seed);

/**
 * Binds two hypervectors into one that can be taken apart again: their WXOR, computed on
 * the device. Commutative, associative, and its own inverse.
 *
 * @p out may alias either input.
 */
hdc_status hdc_bind(const hdc_hypervector *left, const hdc_hypervector *right, hdc_hypervector *out);

/**
 * Recovers a value from `hdc_bind(value, key)`. XOR is its own inverse, so this is
 * hdc_bind() under the name that says what the caller meant.
 */
hdc_status hdc_unbind(const hdc_hypervector *bound, const hdc_hypervector *key, hdc_hypervector *out);

/**
 * Superposes hypervectors into one that still contains every member: their union,
 * computed as a chain of WOR on the device.
 *
 * More members than WMEM holds is fine -- the work is split into several batches, and OR
 * is associative so the answer does not change.
 *
 * @return ::HDC_ERROR_TOO_DENSE if a member, or the growing union, needs more than
 *         ::HDC_MAX_STORABLE_LANES lanes.
 */
hdc_status hdc_bundle(const hdc_hypervector *const *vectors, size_t count, hdc_hypervector *out);

/**
 * The largest bundle hdc_bundle_majority() will take. Longer refuses rather than wrapping.
 *
 * The counters are 13 bit-planes wide, and a bit-plane counter WRAPS when it overflows
 * rather than saturating -- so an unchecked vote over more members than this would come back
 * confidently wrong instead of refused. It is sized to hold a full class of a real training
 * set: the largest MNIST digit has 6,742 examples, and one prototype is one vote over all
 * of them.
 */
#define HDC_MAJORITY_MAX_MEMBERS 8191u

/**
 * @brief The majority vote of @p count hypervectors: a bit survives where more than half
 *        the members had it.
 *
 * This is what builds a class prototype, and it is what hdc_bundle() is not. A union
 * saturates -- fold a few hundred examples of one class together with OR and every bit is
 * set, so the prototype says nothing about the class. A majority stays informative however
 * many members it has.
 *
 * It is more expensive than a union, and the reason is architectural rather than incidental.
 * A majority needs a count per bit position, and the wide ISA has no instruction that counts
 * that way: the reduce modes that would (segmented counts, prefix rank, threshold-accumulate)
 * count per 32-bit lane, which gives 128 counters rather than 4096. So the device builds the
 * counters as bit-planes with a carry-save adder, one WCSA per plane per member, stopping
 * once the carry empties. That is 8 instructions per member for a short bundle and about 13
 * for a thousand-member one, against the union's 4. That ratio is the case for building the
 * unit -- see the library README for the measured table.
 *
 * @param vectors Members of the vote. None may be null and each must fit the device.
 * @param count   How many, from 1 to ::HDC_MAJORITY_MAX_MEMBERS.
 * @param out     The vote. May alias nothing in @p vectors.
 *
 * @return ::HDC_OK, or ::HDC_ERROR_TOO_DENSE if the RESULT does not fit a compressed row.
 *         That last case cannot be predicted before the vote is computed -- unlike a union,
 *         a majority's lane occupancy is not a function of its members' -- so it is reported
 *         after the fact rather than refused up front.
 */
hdc_status hdc_bundle_majority(const hdc_hypervector *const *vectors, size_t count, hdc_hypervector *out);

/**
 * Measures how much two hypervectors overlap.
 *
 * ::hdc_similarity_result.overlap comes back from the device as a scalar: the intersection
 * and its population count are one wide instruction, since the reduce is a mode on the ALU's
 * result bus rather than a second operation. Nothing reads the 4096-bit intersection back.
 *
 * The other two counts are host work, and deliberately so -- they count bits of hypervectors
 * the caller already holds, so the device could tell the host nothing it does not know while
 * costing a transfer to say it.
 */
hdc_status hdc_similarity(const hdc_hypervector *left, const hdc_hypervector *right, hdc_similarity_result *out);

/**
 * The overlap over the geometric mean of the two weights: 1 for identical hypervectors, 0
 * for disjoint ones. This is the binary form of cosine similarity, and the score
 * hdc_classify() ranks by. A hypervector with no set bits is similar to nothing, so this
 * is 0 rather than a division by zero.
 */
double hdc_cosine(const hdc_similarity_result *similarity);

/** The overlap over the union. Harsher than hdc_cosine() when the two weights differ a lot. */
double hdc_jaccard(const hdc_similarity_result *similarity);

/** One hypervector per class: what hdc_train() builds and hdc_classify() searches. */
typedef struct hdc_memory hdc_memory;

/** Classes one memory can hold. WMEM is the real limit on useful sizes long before this is. */
#define HDC_MAX_CLASSES 256

/**
 * Builds one prototype per class by bundling that class's examples, which is how an HDC
 * classifier is trained: no gradients, one pass, and a prototype is a hypervector like any
 * other.
 *
 * Labels are compared by value and copied, so the caller's strings need not outlive the
 * call. Classes keep the order they were first seen in, which is what makes a tie between
 * two prototypes resolve the same way on every run.
 *
 * @param labels   One label per example, @p count long.
 * @param vectors  One hypervector per example, @p count long.
 * @param out      Receives a memory the caller frees with hdc_memory_free().
 */
hdc_status hdc_train(const char *const *labels,
                     const hdc_hypervector *const *vectors,
                     size_t count,
                     hdc_memory **out);

/** Frees a memory. Null is ignored. */
void hdc_memory_free(hdc_memory *memory);

/** How many classes a memory holds. */
size_t hdc_memory_count(const hdc_memory *memory);

/** The label of one class, by index, or null if the index is out of range. */
const char *hdc_memory_label(const hdc_memory *memory, size_t index);

/** The prototype of one class, by index, or null if the index is out of range. */
const hdc_hypervector *hdc_memory_prototype(const hdc_memory *memory, size_t index);

/** The index of a class by label, or -1 if the memory has never seen it. */
int hdc_memory_find(const hdc_memory *memory, const char *label);

/**
 * Names the class whose prototype the query overlaps most.
 *
 * Ties go to the class seen first during training, so the answer does not move between
 * runs. Pass null for either output you do not want.
 *
 * @param best_index  Receives the winning class index, for hdc_memory_label().
 * @param best        Receives the winning class's similarity counts.
 */
hdc_status hdc_classify(const hdc_hypervector *query,
                        const hdc_memory *memory,
                        size_t *best_index,
                        hdc_similarity_result *best);

#ifdef __cplusplus
}
#endif

#endif /* SPARSR_HDC_H */
