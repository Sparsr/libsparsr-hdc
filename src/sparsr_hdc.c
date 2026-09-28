/*
 * Hyperdimensional Computing / Vector Symbolic Architecture on a Sparsr device.
 *
 * The public contract is in include/sparsr_hdc.h. This file is how it reaches the device,
 * and two decisions shape all of it.
 *
 * KERNELS ARE LOADED ONCE, NOT PER CALL
 *
 * hdc_init() writes all three kernels into instruction memory and remembers where each one
 * landed. An operation then writes its data and starts a batch. The alternative -- build
 * the kernel, send it, run it -- costs an extra frame every single call, and a frame is a
 * fixed 256 words whether the kernel is seven words or two hundred. On the emulated link
 * that is one transfer in five for a bind; on a real card it is a PCIe round trip that
 * bought nothing, because the same seven words were already sitting in instruction memory.
 *
 * It also means the bundle kernel has to be one program rather than a family of unrolled
 * ones. It reads its operand count out of data memory and loops, so a bundle of three and a
 * bundle of thirty run the same instructions.
 *
 * THE DEVICE COUNTS THE BITS
 *
 * Similarity is an intersection and a population count, and both now run on the device as
 * one instruction. The population-count reduce is a mode on the wide ALU's result bus, so
 * `popcount(a AND b)` writes an ordinary 32-bit register and the host reads one word of data
 * memory instead of a 4096-bit row.
 *
 * That read-back used to be the single most expensive thing this library did, and it was the
 * SDK rather than the device that kept it: the mode ran on the Sparsr VM well before any
 * intrinsic could emit it, because SPARSR_WOP_R pinned the mode field to zero. A later SDK
 * release added _sparsr_wreduce, this library used it, and the row read is gone.
 *
 * The host still counts the two operand weights hdc_similarity() reports, and that is a
 * decision rather than what is left of the old path. Those count host-owned data the host has
 * just touched; the device could tell it nothing it does not already know, and asking would
 * cost instructions and data words while removing no transfer. See hdc_similarity().
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sparsr.h"
#include "sparsr_hdc.h"

#include "hdc_device_layout.h"
#include "hdc_kernel_images.h"

/* How many times to poll for a batch to finish before giving up. A batch here is at most a
 * few hundred instructions, so anything near this bound means the device stopped
 * answering rather than that it is busy. */
#define HDC_STATUS_POLL_LIMIT 1000000u

#define HDC_LANE_BYTES 4u

/*
 * The public ceiling and the device's real one are two numbers in two headers, and they have
 * to agree: the public one is what a caller reads, the device one is how many bit-planes the
 * kernel actually unrolls. A public ceiling above the real one would let a vote wrap its
 * counters and hand back a confident wrong answer, which is precisely the failure the
 * ceiling exists to prevent. Nothing derives one from the other, so this pins them together
 * at compile time instead.
 */
_Static_assert(HDC_MAJORITY_MAX_MEMBERS == HDC_MAJORITY_MAX_COUNT,
               "the documented majority ceiling must match the counter planes the kernel unrolls");

typedef struct hdc_kernel {
    const uint32_t *image;
    uint32_t word_count;
    uint32_t imem_offset;
} hdc_kernel;

static struct {
    int ready;
    hdc_kernel bind;
    hdc_kernel intersect;
    hdc_kernel bundle;
    hdc_kernel majority;
} g_hdc;

struct hdc_memory {
    size_t count;
    char *labels[HDC_MAX_CLASSES];
    hdc_hypervector prototypes[HDC_MAX_CLASSES];
};

/* ---- status ---------------------------------------------------------------------- */

const char *hdc_status_string(hdc_status status) {
    switch (status) {
    case HDC_OK:                        return "HDC_OK";
    case HDC_ERROR_NOT_INITIALIZED:     return "HDC_ERROR_NOT_INITIALIZED";
    case HDC_ERROR_INVALID_ARGUMENT:    return "HDC_ERROR_INVALID_ARGUMENT";
    case HDC_ERROR_TOO_DENSE:           return "HDC_ERROR_TOO_DENSE";
    case HDC_ERROR_CAPACITY:            return "HDC_ERROR_CAPACITY";
    case HDC_ERROR_DEVICE:              return "HDC_ERROR_DEVICE";
    case HDC_ERROR_OUT_OF_MEMORY:       return "HDC_ERROR_OUT_OF_MEMORY";
    default:                            return "HDC_ERROR_UNKNOWN";
    }
}

/* ---- the hypervector value -------------------------------------------------------- */

void hdc_zero(hdc_hypervector *vector) {
    if (vector != NULL) memset(vector->bytes, 0, HDC_HYPERVECTOR_BYTES);
}

hdc_status hdc_set_bit(hdc_hypervector *vector, uint32_t bit) {
    if (vector == NULL || bit >= HDC_HYPERVECTOR_BITS) return HDC_ERROR_INVALID_ARGUMENT;

    vector->bytes[bit / 8u] |= (uint8_t)(1u << (bit % 8u));
    return HDC_OK;
}

int hdc_get_bit(const hdc_hypervector *vector, uint32_t bit) {
    if (vector == NULL || bit >= HDC_HYPERVECTOR_BITS) return 0;

    return (vector->bytes[bit / 8u] & (uint8_t)(1u << (bit % 8u))) != 0;
}

static uint32_t popcount_byte(uint8_t value) {
    /* Written out rather than using a builtin, because this library is meant to be read
     * and ported, and one byte at a time is not where its time goes. */
    uint32_t count = 0;
    while (value != 0) {
        count += (uint32_t)(value & 1u);
        value = (uint8_t)(value >> 1);
    }
    return count;
}

uint32_t hdc_weight(const hdc_hypervector *vector) {
    if (vector == NULL) return 0;

    uint32_t weight = 0;
    for (uint32_t i = 0; i < HDC_HYPERVECTOR_BYTES; ++i) weight += popcount_byte(vector->bytes[i]);
    return weight;
}

uint32_t hdc_nonzero_lanes(const hdc_hypervector *vector) {
    if (vector == NULL) return 0;

    uint32_t lanes = 0;
    for (uint32_t lane = 0; lane < HDC_LANES; ++lane) {
        const uint8_t *bytes = &vector->bytes[lane * HDC_LANE_BYTES];
        if (bytes[0] != 0 || bytes[1] != 0 || bytes[2] != 0 || bytes[3] != 0) ++lanes;
    }
    return lanes;
}

int hdc_fits_device(const hdc_hypervector *vector) {
    return vector != NULL && hdc_nonzero_lanes(vector) <= HDC_MAX_STORABLE_LANES;
}

hdc_status hdc_random(hdc_hypervector *out, uint32_t weight, uint64_t *seed) {
    if (out == NULL || seed == NULL) return HDC_ERROR_INVALID_ARGUMENT;
    if (weight > HDC_HYPERVECTOR_BITS) return HDC_ERROR_INVALID_ARGUMENT;

    /* A partial Fisher-Yates over all 4096 positions, so the draw is uniform and costs the
     * same whatever the weight. Rejection sampling degenerates as the weight approaches the
     * width, and a caller is free to ask for a dense hypervector even though the device
     * could not store it. */
    uint16_t positions[HDC_HYPERVECTOR_BITS]; /* 8 KB on the stack, and deliberately not static: a
                                               * shipped library must not hide shared state here. */
    for (uint32_t i = 0; i < HDC_HYPERVECTOR_BITS; ++i) positions[i] = (uint16_t)i;

    for (uint32_t i = 0; i < weight; ++i) {
        /* An ordinary 64-bit linear congruential generator, taking its result from the high
         * bits where the period is long. This library brings its own rather than calling
         * rand(), so that a caller's sequence is reproducible and process-wide state is
         * neither depended on nor disturbed. */
        *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t j = i + (uint32_t)((*seed >> 33) % (HDC_HYPERVECTOR_BITS - i));

        uint16_t swap = positions[i];
        positions[i] = positions[j];
        positions[j] = swap;
    }

    hdc_zero(out);
    for (uint32_t i = 0; i < weight; ++i) hdc_set_bit(out, positions[i]);
    return HDC_OK;
}

hdc_status hdc_random_dense(hdc_hypervector *out, uint32_t lanes, uint64_t *seed) {
    if (out == NULL || seed == NULL) return HDC_ERROR_INVALID_ARGUMENT;
    if (lanes == 0 || lanes > HDC_MAX_STORABLE_LANES) return HDC_ERROR_INVALID_ARGUMENT;

    /* Fill whole lanes with random bytes rather than placing individual bits. That is what
     * makes the lane count exactly `lanes` whatever the draw does, which is the property
     * that keeps the result storable at any density. */
    hdc_zero(out);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t value = (uint32_t)(*seed >> 32);

        for (uint32_t byte = 0; byte < HDC_LANE_BYTES; ++byte)
            out->bytes[lane * HDC_LANE_BYTES + byte] = (uint8_t)(value >> (8u * byte));
    }

    return HDC_OK;
}

/* ---- talking to the device --------------------------------------------------------- */

static hdc_status load_kernel(hdc_kernel *kernel, const uint32_t *image, uint32_t word_count, uint32_t *next_offset) {
    kernel->image = image;
    kernel->word_count = word_count;
    kernel->imem_offset = *next_offset;

    /* sparsr_load_batch does not take a const pointer, and cannot: the published ABI
     * predates this caller. Nothing writes through it. */
    sparsr_load_batch((uint32_t *)(uintptr_t)image, kernel->imem_offset, word_count);

    *next_offset += word_count;
    return HDC_OK;
}

/* Defined with the rest of similarity, since that is the only thing it checks. */
static hdc_status probe_device_can_count(void);

hdc_status hdc_init(void) {
    sparsr_kernel_init();

    uint32_t next_offset = 0;
    load_kernel(&g_hdc.bind, hdc_kernel_bind_image,
                (uint32_t)(sizeof hdc_kernel_bind_image / sizeof hdc_kernel_bind_image[0]), &next_offset);
    load_kernel(&g_hdc.intersect, hdc_kernel_intersect_image,
                (uint32_t)(sizeof hdc_kernel_intersect_image / sizeof hdc_kernel_intersect_image[0]), &next_offset);
    load_kernel(&g_hdc.bundle, hdc_kernel_bundle_image,
                (uint32_t)(sizeof hdc_kernel_bundle_image / sizeof hdc_kernel_bundle_image[0]), &next_offset);
    load_kernel(&g_hdc.majority, hdc_kernel_majority_image,
                (uint32_t)(sizeof hdc_kernel_majority_image / sizeof hdc_kernel_majority_image[0]), &next_offset);

    /* Ready before the probe, because the probe runs a real batch through the same helpers
     * every operation uses. If it fails, nothing here is usable and the library says so. */
    g_hdc.ready = 1;

    if (probe_device_can_count() != HDC_OK) {
        g_hdc.ready = 0;
        return HDC_ERROR_DEVICE;
    }

    return HDC_OK;
}

void hdc_shutdown(void) {
    /* Nothing to release on the host. The kernels stay in instruction memory until
     * somebody resets the device, which is exactly what makes this safe to call and then
     * call hdc_init() again. */
    g_hdc.ready = 0;
}

static hdc_status write_row(uint32_t row, const hdc_hypervector *vector) {
    if (!hdc_fits_device(vector)) return HDC_ERROR_TOO_DENSE;

    /* The host library compresses on the way in and takes a non-const pointer, so the
     * bytes are copied rather than cast away. A 512-byte copy is not what costs anything
     * here; the frame that follows is. */
    uint8_t scratch[HDC_HYPERVECTOR_BYTES];
    memcpy(scratch, vector->bytes, HDC_HYPERVECTOR_BYTES);
    sparsr_write_data_wmem(scratch, row);
    return HDC_OK;
}

static hdc_status read_row(uint32_t row, hdc_hypervector *out) {
    uint8_t *bytes = sparsr_read_data_wmem(row);
    if (bytes == NULL) return HDC_ERROR_DEVICE;

    memcpy(out->bytes, bytes, HDC_HYPERVECTOR_BYTES);
    return HDC_OK;
}

static hdc_status run_kernel(const hdc_kernel *kernel) {
    sparsr_execute_batch(kernel->imem_offset);

    for (uint32_t poll = 0; poll < HDC_STATUS_POLL_LIMIT; ++poll)
        if (sparsr_read_status() != 0) return HDC_OK;

    return HDC_ERROR_DEVICE;
}

/* ---- bind ------------------------------------------------------------------------- */

/*
 * Both two-operand kernels have the same shape, so they share this: write the operands,
 * run, read the result back.
 *
 * The result goes to a local before it reaches the caller's `out`, which is what lets `out`
 * alias either input. Callers do that constantly -- folding a running value against a key
 * is the normal way to use bind.
 */
static hdc_status combine(const hdc_kernel *kernel,
                          const hdc_hypervector *left,
                          const hdc_hypervector *right,
                          hdc_hypervector *out) {
    if (!g_hdc.ready) return HDC_ERROR_NOT_INITIALIZED;
    if (left == NULL || right == NULL || out == NULL) return HDC_ERROR_INVALID_ARGUMENT;

    hdc_status status = write_row(HDC_ROW_LEFT, left);
    if (status != HDC_OK) return status;

    status = write_row(HDC_ROW_RIGHT, right);
    if (status != HDC_OK) return status;

    status = run_kernel(kernel);
    if (status != HDC_OK) return status;

    hdc_hypervector result;
    status = read_row(HDC_ROW_RESULT, &result);
    if (status != HDC_OK) return status;

    *out = result;
    return HDC_OK;
}

hdc_status hdc_bind(const hdc_hypervector *left, const hdc_hypervector *right, hdc_hypervector *out) {
    return combine(&g_hdc.bind, left, right, out);
}

hdc_status hdc_unbind(const hdc_hypervector *bound, const hdc_hypervector *key, hdc_hypervector *out) {
    return hdc_bind(bound, key, out);
}

/* ---- bundle ------------------------------------------------------------------------ */

/*
 * Runs one phase of the majority kernel and reads back what it says happened.
 *
 * This is the one operation in the library that reads a status word, and the asymmetry is
 * deliberate -- see hdc_device_layout.h. The sentinel goes in before the batch so that a
 * batch which faulted partway, and therefore never reached its own status write, cannot be
 * read as the previous phase's success.
 */
static hdc_status run_majority(uint32_t mode, uint32_t operand_count, uint32_t total, uint32_t *device_status) {
    uint32_t words[4];
    words[0] = operand_count;                     /* HDC_DMEM_OPERAND_COUNT_WORD   */
    words[1] = mode;                              /* HDC_DMEM_MAJORITY_MODE_WORD   */
    words[2] = total;                             /* HDC_DMEM_MAJORITY_TOTAL_WORD  */
    words[3] = HDC_MAJORITY_STATUS_UNSET;         /* HDC_DMEM_MAJORITY_STATUS_WORD */
    sparsr_write_data_dmem(words, HDC_DMEM_OPERAND_COUNT_WORD, 4);

    hdc_status status = run_kernel(&g_hdc.majority);
    if (status != HDC_OK) return status;

    uint32_t *read_back = sparsr_read_data_dmem(HDC_DMEM_MAJORITY_STATUS_WORD, 1);
    if (read_back == NULL) return HDC_ERROR_DEVICE;

    *device_status = read_back[0];
    return HDC_OK;
}

hdc_status hdc_bundle_majority(const hdc_hypervector *const *vectors, size_t count, hdc_hypervector *out) {
    if (!g_hdc.ready) return HDC_ERROR_NOT_INITIALIZED;
    if (vectors == NULL || out == NULL || count == 0) return HDC_ERROR_INVALID_ARGUMENT;

    /* The counters are HDC_MAJORITY_COUNTER_PLANES bits wide and the kernel drops the carry
     * out of the top plane, so a longer bundle would wrap and produce a confident wrong
     * answer. Refused rather than truncated. */
    if (count > HDC_MAJORITY_MAX_COUNT) return HDC_ERROR_INVALID_ARGUMENT;

    for (size_t i = 0; i < count; ++i) {
        if (vectors[i] == NULL) return HDC_ERROR_INVALID_ARGUMENT;
        if (!hdc_fits_device(vectors[i])) return HDC_ERROR_TOO_DENSE;
    }

    /* The majority of one vector is that vector, and the device would only hand back what
     * it was given. */
    if (count == 1) {
        *out = *vectors[0];
        return HDC_OK;
    }

    /*
     * Unlike hdc_bundle, the result is NOT predicted on the host, and it cannot be. A union
     * only ever occupies lanes its members occupied, so a lane-occupancy map settles it
     * before anything is sent. A majority does not work that way: which lanes the vote
     * occupies is not a function of which lanes the inputs occupied, so the only way to
     * know is to compute the vote -- which is what the device is for. Hence the status word.
     */
    uint32_t device_status = HDC_MAJORITY_STATUS_UNSET;
    hdc_status status = run_majority(HDC_MAJORITY_MODE_RESET, 0, 0, &device_status);
    if (status != HDC_OK) return status;
    if (device_status != HDC_MAJORITY_STATUS_OK) return HDC_ERROR_DEVICE;

    size_t next = 0;
    while (next < count) {
        size_t take = count - next;
        if (take > HDC_BUNDLE_MAX_OPERANDS_PER_BATCH) take = HDC_BUNDLE_MAX_OPERANDS_PER_BATCH;

        for (size_t operand = 0; operand < take; ++operand) {
            status = write_row(HDC_ROW_BUNDLE_FIRST + (uint32_t)operand, vectors[next + operand]);
            if (status != HDC_OK) return status;
        }

        /* The counter planes live in wide registers and a batch does not clear them, which
         * is what lets this pick up where the last batch left off without spending a WMEM
         * row on the accumulator. */
        status = run_majority(HDC_MAJORITY_MODE_ACCUMULATE, (uint32_t)take, 0, &device_status);
        if (status != HDC_OK) return status;
        if (device_status != HDC_MAJORITY_STATUS_OK) return HDC_ERROR_DEVICE;

        next += take;
    }

    status = run_majority(HDC_MAJORITY_MODE_THRESHOLD, 0, (uint32_t)count, &device_status);
    if (status != HDC_OK) return status;
    if (device_status == HDC_MAJORITY_STATUS_TOO_DENSE) return HDC_ERROR_TOO_DENSE;
    if (device_status != HDC_MAJORITY_STATUS_OK) return HDC_ERROR_DEVICE;

    return read_row(HDC_ROW_LEFT, out);
}

hdc_status hdc_bundle(const hdc_hypervector *const *vectors, size_t count, hdc_hypervector *out) {
    if (!g_hdc.ready) return HDC_ERROR_NOT_INITIALIZED;
    if (vectors == NULL || out == NULL || count == 0) return HDC_ERROR_INVALID_ARGUMENT;

    /*
     * A union grows, so a bundle can outgrow a compressed row partway through even when
     * every member fits on its own. The device faults rather than truncating -- but the
     * public host ABI cannot report that: read_status carries one done bit, and a
     * faulting batch sets it exactly like a successful one. A caller would be handed the
     * accumulator row as it stood before the refused store: a valid-looking hypervector
     * that silently lost members.
     *
     * So the overflow is predicted here, before anything is sent. What decides it is which
     * *lanes* end up non-zero, not the bits inside them, and that is a 128-bit occupancy
     * map -- cheap to accumulate and far short of recomputing the union. Recomputing it
     * would answer the question too, but a library that works out the result on the host in
     * order to check the device is not using the device.
     */
    uint32_t union_lanes[HDC_LANES / 32] = {0};
    for (size_t i = 0; i < count; ++i) {
        if (vectors[i] == NULL) return HDC_ERROR_INVALID_ARGUMENT;
        if (!hdc_fits_device(vectors[i])) return HDC_ERROR_TOO_DENSE;

        for (uint32_t lane = 0; lane < HDC_LANES; ++lane) {
            const uint8_t *bytes = &vectors[i]->bytes[lane * HDC_LANE_BYTES];
            if (bytes[0] != 0 || bytes[1] != 0 || bytes[2] != 0 || bytes[3] != 0)
                union_lanes[lane / 32u] |= 1u << (lane % 32u);
        }
    }

    uint32_t occupied = 0;
    for (uint32_t word = 0; word < HDC_LANES / 32u; ++word)
        for (uint32_t bit = 0; bit < 32u; ++bit)
            occupied += (union_lanes[word] >> bit) & 1u;

    if (occupied > HDC_MAX_STORABLE_LANES) return HDC_ERROR_TOO_DENSE;

    /* Bundling one hypervector is the identity, and the device would only hand back what it
     * was given. Every other count runs on the device. */
    if (count == 1) {
        *out = *vectors[0];
        return HDC_OK;
    }

    hdc_status status = write_row(HDC_ROW_LEFT, vectors[0]);
    if (status != HDC_OK) return status;

    size_t next = 1;
    while (next < count) {
        size_t take = count - next;
        if (take > HDC_BUNDLE_MAX_OPERANDS_PER_BATCH) take = HDC_BUNDLE_MAX_OPERANDS_PER_BATCH;

        for (size_t operand = 0; operand < take; ++operand) {
            status = write_row(HDC_ROW_BUNDLE_FIRST + (uint32_t)operand, vectors[next + operand]);
            if (status != HDC_OK) return status;
        }

        uint32_t operand_count = (uint32_t)take;
        sparsr_write_data_dmem(&operand_count, HDC_DMEM_OPERAND_COUNT_WORD, 1);

        status = run_kernel(&g_hdc.bundle);
        if (status != HDC_OK) return status;

        next += take;
    }

    return read_row(HDC_ROW_LEFT, out);
}

/* ---- similarity --------------------------------------------------------------------- */

/*
 * Runs the intersect kernel over two rows the caller has already written, and returns the
 * overlap the device reduced to.
 *
 * No sentinel is written before the batch, and only probe_device_can_count() below does.
 * Nothing in this kernel can fault -- it holds no store to a compressed row -- so the word
 * read back after a successful batch is the word this batch wrote. See hdc_device_layout.h
 * for why that is worth one probe per process rather than a status word per call.
 */
static hdc_status run_intersect(uint32_t *overlap) {
    hdc_status status = run_kernel(&g_hdc.intersect);
    if (status != HDC_OK) return status;

    uint32_t *read_back = sparsr_read_data_dmem(HDC_DMEM_SIMILARITY_OVERLAP_WORD, 1);
    if (read_back == NULL) return HDC_ERROR_DEVICE;

    *overlap = read_back[0];
    return HDC_OK;
}

hdc_status hdc_similarity(const hdc_hypervector *left, const hdc_hypervector *right, hdc_similarity_result *out) {
    if (!g_hdc.ready) return HDC_ERROR_NOT_INITIALIZED;
    if (left == NULL || right == NULL || out == NULL) return HDC_ERROR_INVALID_ARGUMENT;

    hdc_status status = write_row(HDC_ROW_LEFT, left);
    if (status != HDC_OK) return status;

    status = write_row(HDC_ROW_RIGHT, right);
    if (status != HDC_OK) return status;

    uint32_t overlap = 0;
    status = run_intersect(&overlap);
    if (status != HDC_OK) return status;

    out->overlap = overlap;

    /* The two operand weights stay on the host, and that is a decision rather than an
     * omission. They count bits of hypervectors the host owns, has just touched to write
     * the rows, and therefore has in cache -- so asking the device for them would add two
     * wide instructions and two data words without removing a single transfer. Only the
     * overlap needs the device, because only the overlap needs both vectors in one place. */
    out->left_weight = hdc_weight(left);
    out->right_weight = hdc_weight(right);
    return HDC_OK;
}

/*
 * Asks the device one question it must get right, and refuses to come up if it does not.
 *
 * Similarity is a population-count reduce now, and a reduce is the one thing here a backend
 * can lack while running everything else perfectly. The four scalar-writing reduce modes
 * exist on the Sparsr VM; the RTL has none of them. On such a device bind and bundle
 * would work, the reduce would fault, nothing would write the overlap word, and every
 * similarity would report whatever data memory happened to hold -- a confident wrong number,
 * which is the failure mode this library goes out of its way to avoid.
 *
 * Whether a backend implements the mode is a property of the backend and not of the call, so
 * it is settled once here rather than costing a status word on every similarity. The sentinel
 * goes in first: a batch that faulted never reached its own store, so the sentinel surviving
 * is exactly the signal.
 *
 * The pair is chosen so that no plausible wrong answer passes. Left holds bits 0 to 10 and
 * right holds bits 4 to 16, so the overlap is 7 while the weights are 11 and 13 -- a device
 * that returns a constant, an operand weight, or nothing at all is caught rather than
 * believed. Both sit inside one lane, so neither can be refused as too dense.
 */
#define HDC_PROBE_LEFT_WEIGHT   11u
#define HDC_PROBE_RIGHT_FIRST    4u
#define HDC_PROBE_RIGHT_WEIGHT  13u
#define HDC_PROBE_OVERLAP        7u

static hdc_status probe_device_can_count(void) {
    hdc_hypervector left, right;
    hdc_zero(&left);
    hdc_zero(&right);
    for (uint32_t bit = 0; bit < HDC_PROBE_LEFT_WEIGHT; ++bit) hdc_set_bit(&left, bit);
    for (uint32_t bit = 0; bit < HDC_PROBE_RIGHT_WEIGHT; ++bit) hdc_set_bit(&right, HDC_PROBE_RIGHT_FIRST + bit);

    hdc_status status = write_row(HDC_ROW_LEFT, &left);
    if (status != HDC_OK) return status;

    status = write_row(HDC_ROW_RIGHT, &right);
    if (status != HDC_OK) return status;

    uint32_t sentinel = HDC_SIMILARITY_OVERLAP_UNSET;
    sparsr_write_data_dmem(&sentinel, HDC_DMEM_SIMILARITY_OVERLAP_WORD, 1);

    uint32_t overlap = HDC_SIMILARITY_OVERLAP_UNSET;
    status = run_intersect(&overlap);
    if (status != HDC_OK) return status;

    return overlap == HDC_PROBE_OVERLAP ? HDC_OK : HDC_ERROR_DEVICE;
}

double hdc_cosine(const hdc_similarity_result *similarity) {
    if (similarity == NULL) return 0.0;
    if (similarity->left_weight == 0 || similarity->right_weight == 0) return 0.0;

    return (double)similarity->overlap / sqrt((double)similarity->left_weight * (double)similarity->right_weight);
}

double hdc_jaccard(const hdc_similarity_result *similarity) {
    if (similarity == NULL) return 0.0;

    uint32_t combined = similarity->left_weight + similarity->right_weight - similarity->overlap;
    if (combined == 0) return 0.0;

    return (double)similarity->overlap / (double)combined;
}

/* ---- train and classify -------------------------------------------------------------- */

void hdc_memory_free(hdc_memory *memory) {
    if (memory == NULL) return;

    for (size_t i = 0; i < memory->count; ++i) free(memory->labels[i]);
    free(memory);
}

size_t hdc_memory_count(const hdc_memory *memory) {
    return memory == NULL ? 0 : memory->count;
}

const char *hdc_memory_label(const hdc_memory *memory, size_t index) {
    if (memory == NULL || index >= memory->count) return NULL;
    return memory->labels[index];
}

const hdc_hypervector *hdc_memory_prototype(const hdc_memory *memory, size_t index) {
    if (memory == NULL || index >= memory->count) return NULL;
    return &memory->prototypes[index];
}

int hdc_memory_find(const hdc_memory *memory, const char *label) {
    if (memory == NULL || label == NULL) return -1;

    for (size_t i = 0; i < memory->count; ++i)
        if (strcmp(memory->labels[i], label) == 0) return (int)i;

    return -1;
}

static char *duplicate(const char *text) {
    size_t length = strlen(text) + 1;
    char *copy = malloc(length);
    if (copy != NULL) memcpy(copy, text, length);
    return copy;
}

hdc_status hdc_train(const char *const *labels,
                     const hdc_hypervector *const *vectors,
                     size_t count,
                     hdc_memory **out) {
    if (!g_hdc.ready) return HDC_ERROR_NOT_INITIALIZED;
    if (labels == NULL || vectors == NULL || out == NULL || count == 0) return HDC_ERROR_INVALID_ARGUMENT;

    *out = NULL;

    hdc_memory *memory = calloc(1, sizeof *memory);
    if (memory == NULL) return HDC_ERROR_OUT_OF_MEMORY;

    /* Which class each example belongs to, and one scratch list to gather a class's members
     * into. Both are sized by the number of examples, so training on ten thousand of them
     * costs two small allocations rather than a table of every class against every class. */
    size_t *class_of = calloc(count, sizeof *class_of);
    const hdc_hypervector **members = calloc(count, sizeof *members);
    if (class_of == NULL || members == NULL) {
        free(class_of);
        free(members);
        hdc_memory_free(memory);
        return HDC_ERROR_OUT_OF_MEMORY;
    }

    /* Assign classes in the order the labels are first seen. The order matters: it is what
     * makes a tie between two prototypes resolve the same way on every run, so a classifier
     * is reproducible rather than reproducible by accident. */
    hdc_status status = HDC_OK;
    for (size_t i = 0; i < count; ++i) {
        if (labels[i] == NULL || vectors[i] == NULL) { status = HDC_ERROR_INVALID_ARGUMENT; goto done; }

        int index = hdc_memory_find(memory, labels[i]);
        if (index < 0) {
            if (memory->count == HDC_MAX_CLASSES) { status = HDC_ERROR_CAPACITY; goto done; }

            memory->labels[memory->count] = duplicate(labels[i]);
            if (memory->labels[memory->count] == NULL) { status = HDC_ERROR_OUT_OF_MEMORY; goto done; }

            index = (int)memory->count;
            ++memory->count;
        }

        class_of[i] = (size_t)index;
    }

    for (size_t class_index = 0; class_index < memory->count; ++class_index) {
        size_t gathered = 0;
        for (size_t i = 0; i < count; ++i)
            if (class_of[i] == class_index) members[gathered++] = vectors[i];

        status = hdc_bundle(members, gathered, &memory->prototypes[class_index]);
        if (status != HDC_OK) goto done;
    }

done:
    free(class_of);
    free(members);

    if (status != HDC_OK) {
        hdc_memory_free(memory);
        return status;
    }

    *out = memory;
    return HDC_OK;
}

hdc_status hdc_classify(const hdc_hypervector *query,
                        const hdc_memory *memory,
                        size_t *best_index,
                        hdc_similarity_result *best) {
    if (!g_hdc.ready) return HDC_ERROR_NOT_INITIALIZED;
    if (query == NULL || memory == NULL || memory->count == 0) return HDC_ERROR_INVALID_ARGUMENT;

    size_t winner = 0;
    double winning_score = -1.0;
    hdc_similarity_result winning_similarity = {0, 0, 0};

    for (size_t i = 0; i < memory->count; ++i) {
        hdc_similarity_result similarity;
        hdc_status status = hdc_similarity(query, &memory->prototypes[i], &similarity);
        if (status != HDC_OK) return status;

        double score = hdc_cosine(&similarity);
        if (score <= winning_score) continue;

        winning_score = score;
        winning_similarity = similarity;
        winner = i;
    }

    if (best_index != NULL) *best_index = winner;
    if (best != NULL) *best = winning_similarity;
    return HDC_OK;
}
