/*
 * The Sparsr HDC/VSA library, checked against a real device.
 *
 * Every case here runs the library's kernels on whatever libsparsr_host is pointed at,
 * which the Makefile sets to the Sparsr VM. There is no mock: the operations under test
 * are wide instructions, and a host-side reimplementation of them would test nothing.
 *
 * Failures print and the harness exits non-zero, so `make test` is the whole signal.
 *
 * Almost everything here goes through the public header alone, which is the point: the
 * tests exercise the library the way a caller does. One case is deliberately different --
 * `test_the_ripple_is_fused` reads the compiled kernel images, because the property it
 * checks is which instruction the hot loop uses, and no answer the library returns can
 * reveal that.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparsr_hdc.h"

/* The device-side half: the plane count and the generated images the library sends to
 * instruction memory. Only test_the_ripple_is_fused reads them. */
#include "hdc_device_layout.h"
#include "hdc_kernel_images.h"

/*
 * Two constants from the frozen wide encoding, repeated here rather than included.
 * The SDK's sparsr_intrinsics.h is the authority for both, but it defines them
 * inside `#if defined(__riscv)` and refuses any other target -- correctly, since it exists
 * to emit instructions and this machine cannot run them. This test only decodes words, so
 * it names the two fields it reads and cites where they come from.
 */
#define SPARSR_CUSTOM0    0x0Bu /**< opcode, inst[6:0] */
#define SPARSR_F3_WMULTI  0x1u  /**< funct3, inst[14:12]: the multi-result class */

static int g_checks = 0;
static int g_failures = 0;
static const char *g_case = "";

#define CASE(name) do { g_case = (name); } while (0)

#define CHECK(condition) do {                                                        \
    ++g_checks;                                                                      \
    if (!(condition)) {                                                              \
        ++g_failures;                                                                \
        printf("FAIL [%s] %s:%d: %s\n", g_case, __FILE__, __LINE__, #condition);     \
    }                                                                                \
} while (0)

#define CHECK_EQ_U32(actual, expected) do {                                          \
    uint32_t actual_value = (uint32_t)(actual);                                      \
    uint32_t expected_value = (uint32_t)(expected);                                  \
    ++g_checks;                                                                      \
    if (actual_value != expected_value) {                                            \
        ++g_failures;                                                                \
        printf("FAIL [%s] %s:%d: %s was %u, expected %u\n",                          \
               g_case, __FILE__, __LINE__, #actual, actual_value, expected_value);   \
    }                                                                                \
} while (0)

#define CHECK_STATUS(call, expected) do {                                            \
    hdc_status status_value = (call);                                                \
    ++g_checks;                                                                      \
    if (status_value != (expected)) {                                                \
        ++g_failures;                                                                \
        printf("FAIL [%s] %s:%d: %s returned %s, expected %s\n",                     \
               g_case, __FILE__, __LINE__, #call,                                    \
               hdc_status_string(status_value), hdc_status_string(expected));        \
    }                                                                                \
} while (0)

/* ---- helpers ------------------------------------------------------------------- */

static hdc_hypervector bits(const uint32_t *positions, size_t count) {
    hdc_hypervector vector;
    hdc_zero(&vector);
    for (size_t i = 0; i < count; ++i) hdc_set_bit(&vector, positions[i]);
    return vector;
}

/* One set bit in each of `count` distinct lanes, so the lane count is exactly `count`. */
static hdc_hypervector one_bit_per_lane(uint32_t first_lane, uint32_t count) {
    hdc_hypervector vector;
    hdc_zero(&vector);
    for (uint32_t i = 0; i < count; ++i) hdc_set_bit(&vector, (first_lane + i) * 32);
    return vector;
}

static int same(const hdc_hypervector *left, const hdc_hypervector *right) {
    return memcmp(left->bytes, right->bytes, HDC_HYPERVECTOR_BYTES) == 0;
}

/* ---- the hypervector value ------------------------------------------------------ */

static void test_hypervector_layout(void) {
    CASE("hypervector layout");

    uint32_t positions[] = {0, 9, 4095};
    hdc_hypervector vector = bits(positions, 3);

    CHECK_EQ_U32(vector.bytes[0], 0x01);
    CHECK_EQ_U32(vector.bytes[1], 0x02);
    CHECK_EQ_U32(vector.bytes[511], 0x80);
    CHECK_EQ_U32(hdc_weight(&vector), 3);
    CHECK(hdc_get_bit(&vector, 9));
    CHECK(!hdc_get_bit(&vector, 10));

    CHECK_STATUS(hdc_set_bit(&vector, HDC_HYPERVECTOR_BITS), HDC_ERROR_INVALID_ARGUMENT);
}

static void test_weight_is_not_lane_count(void) {
    CASE("weight is not lane count");

    hdc_hypervector packed;
    hdc_zero(&packed);
    for (uint32_t bit = 0; bit < 32; ++bit) hdc_set_bit(&packed, bit);

    CHECK_EQ_U32(hdc_weight(&packed), 32);
    CHECK_EQ_U32(hdc_nonzero_lanes(&packed), 1);

    hdc_hypervector spread = one_bit_per_lane(0, 4);
    CHECK_EQ_U32(hdc_weight(&spread), 4);
    CHECK_EQ_U32(hdc_nonzero_lanes(&spread), 4);
}

static void test_storability_rule(void) {
    CASE("storability rule");

    CHECK(hdc_fits_device(&(hdc_hypervector){{0}}));

    hdc_hypervector at_limit = one_bit_per_lane(0, HDC_MAX_STORABLE_LANES);
    CHECK_EQ_U32(hdc_nonzero_lanes(&at_limit), HDC_MAX_STORABLE_LANES);
    CHECK(hdc_fits_device(&at_limit));

    hdc_hypervector over = one_bit_per_lane(0, HDC_MAX_STORABLE_LANES + 1);
    CHECK(!hdc_fits_device(&over));

    /* The invariant a caller minting symbols relies on: 48 set bits always fit, whatever
     * the draw picks, because each set bit touches at most one lane. */
    uint64_t seed = 1;
    for (int trial = 0; trial < 50; ++trial) {
        hdc_hypervector drawn;
        CHECK_STATUS(hdc_random(&drawn, HDC_MAX_STORABLE_LANES, &seed), HDC_OK);
        CHECK_EQ_U32(hdc_weight(&drawn), HDC_MAX_STORABLE_LANES);
        CHECK(hdc_fits_device(&drawn));
    }
}

static void test_random_is_reproducible_and_varied(void) {
    CASE("random draws");

    uint64_t seed_a = 7;
    uint64_t seed_b = 7;
    hdc_hypervector first, second, repeat;

    CHECK_STATUS(hdc_random(&first, 20, &seed_a), HDC_OK);
    CHECK_STATUS(hdc_random(&second, 20, &seed_a), HDC_OK);
    CHECK_STATUS(hdc_random(&repeat, 20, &seed_b), HDC_OK);

    CHECK(same(&first, &repeat));
    CHECK(!same(&first, &second));

    hdc_hypervector too_wide;
    CHECK_STATUS(hdc_random(&too_wide, HDC_HYPERVECTOR_BITS + 1, &seed_a), HDC_ERROR_INVALID_ARGUMENT);
}

static void test_dense_codes_fit_when_confined_to_lanes(void) {
    CASE("dense codes within 48 lanes");

    /* The limit counts lanes, not set bits, and an occupied lane is stored whole -- so a
     * fully dense code confined to 48 lanes fits at any density. That is 1536 of the 4096
     * bits, and it is the regime hdc_random cannot reach. */
    uint64_t seed = 909;
    hdc_hypervector widest;
    CHECK_STATUS(hdc_random_dense(&widest, HDC_MAX_STORABLE_LANES, &seed), HDC_OK);

    CHECK_EQ_U32(hdc_nonzero_lanes(&widest), HDC_MAX_STORABLE_LANES);
    CHECK(hdc_fits_device(&widest));

    /* Roughly half of 1536 bits set: far denser than anything hdc_random could store, and
     * the bound is wide enough that the draw cannot fail it by chance. */
    uint32_t weight = hdc_weight(&widest);
    CHECK(weight > 600 && weight < 940);
    if (weight <= 600 || weight >= 940) printf("       (weight was %u)\n", weight);

    /* Nothing outside the occupied lanes. */
    for (uint32_t bit = HDC_MAX_STORABLE_LANES * 32u; bit < HDC_HYPERVECTOR_BITS; ++bit)
        if (hdc_get_bit(&widest, bit)) { CHECK(0); break; }

    /* It really does reach the device, which is the whole point of it being storable. */
    hdc_hypervector round_tripped;
    CHECK_STATUS(hdc_bind(&widest, &(hdc_hypervector){{0}}, &round_tripped), HDC_OK);
    CHECK(same(&round_tripped, &widest));

    hdc_hypervector narrow;
    CHECK_STATUS(hdc_random_dense(&narrow, 1, &seed), HDC_OK);
    CHECK_EQ_U32(hdc_nonzero_lanes(&narrow), 1);

    CHECK_STATUS(hdc_random_dense(&narrow, 0, &seed), HDC_ERROR_INVALID_ARGUMENT);
    CHECK_STATUS(hdc_random_dense(&narrow, HDC_MAX_STORABLE_LANES + 1, &seed), HDC_ERROR_INVALID_ARGUMENT);
}

/* ---- bind ----------------------------------------------------------------------- */

static void test_bind_is_symmetric_difference(void) {
    CASE("bind is symmetric difference");

    uint32_t left_bits[] = {1, 40, 100};
    uint32_t right_bits[] = {40, 100, 200};
    uint32_t expected_bits[] = {1, 200};

    hdc_hypervector left = bits(left_bits, 3);
    hdc_hypervector right = bits(right_bits, 3);
    hdc_hypervector expected = bits(expected_bits, 2);
    hdc_hypervector result;

    CHECK_STATUS(hdc_bind(&left, &right, &result), HDC_OK);
    CHECK(same(&result, &expected));
}

static void test_bind_is_commutative_and_self_inverse(void) {
    CASE("bind algebra");

    uint64_t seed = 31;
    hdc_hypervector value, key, forward, backward, recovered;
    hdc_random(&value, 18, &seed);
    hdc_random(&key, 18, &seed);

    CHECK_STATUS(hdc_bind(&value, &key, &forward), HDC_OK);
    CHECK_STATUS(hdc_bind(&key, &value, &backward), HDC_OK);
    CHECK(same(&forward, &backward));

    CHECK_STATUS(hdc_unbind(&forward, &key, &recovered), HDC_OK);
    CHECK(same(&recovered, &value));

    hdc_hypervector zero, with_zero, self;
    hdc_zero(&zero);
    CHECK_STATUS(hdc_bind(&value, &zero, &with_zero), HDC_OK);
    CHECK(same(&with_zero, &value));

    CHECK_STATUS(hdc_bind(&value, &value, &self), HDC_OK);
    CHECK(same(&self, &zero));
}

static void test_bind_output_may_alias_an_input(void) {
    CASE("bind aliasing");

    uint64_t seed = 41;
    hdc_hypervector value, key, expected;
    hdc_random(&value, 18, &seed);
    hdc_random(&key, 18, &seed);

    CHECK_STATUS(hdc_bind(&value, &key, &expected), HDC_OK);

    hdc_hypervector in_place = value;
    CHECK_STATUS(hdc_bind(&in_place, &key, &in_place), HDC_OK);
    CHECK(same(&in_place, &expected));
}

static void test_bind_refuses_a_dense_operand(void) {
    CASE("bind refuses dense");

    hdc_hypervector dense = one_bit_per_lane(0, HDC_MAX_STORABLE_LANES + 12);
    hdc_hypervector sparse = one_bit_per_lane(0, 1);
    hdc_hypervector result;

    CHECK_STATUS(hdc_bind(&dense, &sparse, &result), HDC_ERROR_TOO_DENSE);
}

/* ---- bundle --------------------------------------------------------------------- */

static void test_bundle_is_the_union(void) {
    CASE("bundle is the union");

    uint32_t first_bits[] = {1, 40};
    uint32_t second_bits[] = {40, 100};
    uint32_t expected_bits[] = {1, 40, 100};

    hdc_hypervector first = bits(first_bits, 2);
    hdc_hypervector second = bits(second_bits, 2);
    hdc_hypervector expected = bits(expected_bits, 3);
    const hdc_hypervector *members[] = {&first, &second};
    hdc_hypervector result;

    CHECK_STATUS(hdc_bundle(members, 2, &result), HDC_OK);
    CHECK(same(&result, &expected));
}

static void test_bundle_is_order_independent_and_idempotent(void) {
    CASE("bundle algebra");

    uint64_t seed = 91;
    hdc_hypervector a, b, c;
    hdc_random(&a, 10, &seed);
    hdc_random(&b, 10, &seed);
    hdc_random(&c, 10, &seed);

    const hdc_hypervector *forward[] = {&a, &b, &c};
    const hdc_hypervector *shuffled[] = {&c, &a, &b};
    hdc_hypervector one_way, other_way;

    CHECK_STATUS(hdc_bundle(forward, 3, &one_way), HDC_OK);
    CHECK_STATUS(hdc_bundle(shuffled, 3, &other_way), HDC_OK);
    CHECK(same(&one_way, &other_way));

    const hdc_hypervector *twice[] = {&a, &a};
    hdc_hypervector doubled;
    CHECK_STATUS(hdc_bundle(twice, 2, &doubled), HDC_OK);
    CHECK(same(&doubled, &a));

    const hdc_hypervector *alone[] = {&a};
    hdc_hypervector single;
    CHECK_STATUS(hdc_bundle(alone, 1, &single), HDC_OK);
    CHECK(same(&single, &a));

    CHECK_STATUS(hdc_bundle(alone, 0, &single), HDC_ERROR_INVALID_ARGUMENT);
}

static void test_a_member_stays_contained_in_the_bundle(void) {
    CASE("bundle contains its members");

    /* What makes bundling useful: every member is still recoverable by similarity
     * afterwards. With a union the containment is exact, not statistical. */
    uint64_t seed = 111;
    hdc_hypervector member, other, third, bundle;
    hdc_random(&member, 12, &seed);
    hdc_random(&other, 12, &seed);
    hdc_random(&third, 12, &seed);

    const hdc_hypervector *members[] = {&member, &other, &third};
    CHECK_STATUS(hdc_bundle(members, 3, &bundle), HDC_OK);

    hdc_similarity_result similarity;
    CHECK_STATUS(hdc_similarity(&member, &bundle, &similarity), HDC_OK);
    CHECK_EQ_U32(similarity.overlap, hdc_weight(&member));
}

static void test_bundle_beyond_one_batch(void) {
    CASE("bundle across batches");

    /* CMEM has 32 rows and one holds the running accumulator, so 40 members cannot all be
     * resident at once. The library splits the work; the answer must not change. */
    hdc_hypervector members[40];
    const hdc_hypervector *pointers[40];
    for (uint32_t i = 0; i < 40; ++i) {
        members[i] = one_bit_per_lane(i, 1);
        pointers[i] = &members[i];
    }

    hdc_hypervector bundle;
    CHECK_STATUS(hdc_bundle(pointers, 40, &bundle), HDC_OK);
    CHECK_EQ_U32(hdc_weight(&bundle), 40);
    CHECK_EQ_U32(hdc_nonzero_lanes(&bundle), 40);
}

static void test_bundle_past_capacity_is_reported(void) {
    CASE("bundle past capacity");

    hdc_hypervector members[60];
    const hdc_hypervector *pointers[60];
    for (uint32_t i = 0; i < 60; ++i) {
        members[i] = one_bit_per_lane(i, 1);
        pointers[i] = &members[i];
    }

    hdc_hypervector bundle;
    CHECK_STATUS(hdc_bundle(pointers, 60, &bundle), HDC_ERROR_TOO_DENSE);
}

/* ---- similarity ----------------------------------------------------------------- */

static void test_similarity_counts_the_shared_bits(void) {
    CASE("similarity counts");

    uint32_t left_bits[] = {1, 40, 100};
    uint32_t right_bits[] = {40, 100, 200};
    hdc_hypervector left = bits(left_bits, 3);
    hdc_hypervector right = bits(right_bits, 3);

    hdc_similarity_result similarity;
    CHECK_STATUS(hdc_similarity(&left, &right, &similarity), HDC_OK);

    CHECK_EQ_U32(similarity.overlap, 2);
    CHECK_EQ_U32(similarity.left_weight, 3);
    CHECK_EQ_U32(similarity.right_weight, 3);
    CHECK(hdc_cosine(&similarity) > 0.66 && hdc_cosine(&similarity) < 0.67);
    CHECK(hdc_jaccard(&similarity) > 0.49 && hdc_jaccard(&similarity) < 0.51);
}

static void test_similarity_endpoints(void) {
    CASE("similarity endpoints");

    uint64_t seed = 131;
    hdc_hypervector vector, zero;
    hdc_random(&vector, 20, &seed);
    hdc_zero(&zero);

    hdc_similarity_result identical, with_zero, disjoint;
    CHECK_STATUS(hdc_similarity(&vector, &vector, &identical), HDC_OK);
    CHECK_EQ_U32(identical.overlap, 20);
    CHECK(hdc_cosine(&identical) > 0.999999);

    CHECK_STATUS(hdc_similarity(&vector, &zero, &with_zero), HDC_OK);
    CHECK_EQ_U32(with_zero.overlap, 0);
    CHECK(hdc_cosine(&with_zero) == 0.0);
    CHECK(hdc_jaccard(&with_zero) == 0.0);

    uint32_t left_bits[] = {1, 2};
    uint32_t right_bits[] = {3, 4};
    hdc_hypervector left = bits(left_bits, 2);
    hdc_hypervector right = bits(right_bits, 2);
    CHECK_STATUS(hdc_similarity(&left, &right, &disjoint), HDC_OK);
    CHECK_EQ_U32(disjoint.overlap, 0);
    CHECK(hdc_cosine(&disjoint) == 0.0);
}

/*
 * The counts, against a reference written a different way.
 *
 * The overlap is produced by the device's population-count reduce and arrives as a scalar.
 * That makes it worth checking against something that shares no code with it: the reference
 * below walks bit positions through the public hdc_get_bit(), where hdc_weight() walks bytes
 * through a table-free popcount. Two implementations that disagree cannot both be right.
 *
 * The dense pairs are the part the rest of the suite never reaches. Every other similarity
 * case counts at most 20 bits, so a reduce that saturated, sign-extended, or counted only
 * part of the 4096 bits would pass all of them. Here an overlap runs into the hundreds.
 */
static uint32_t reference_weight(const hdc_hypervector *vector) {
    uint32_t count = 0;
    for (uint32_t bit = 0; bit < HDC_HYPERVECTOR_BITS; ++bit) count += (uint32_t)hdc_get_bit(vector, bit);
    return count;
}

static uint32_t reference_overlap(const hdc_hypervector *left, const hdc_hypervector *right) {
    uint32_t count = 0;
    for (uint32_t bit = 0; bit < HDC_HYPERVECTOR_BITS; ++bit)
        if (hdc_get_bit(left, bit) && hdc_get_bit(right, bit)) ++count;
    return count;
}

static void check_similarity_against_reference(const hdc_hypervector *left, const hdc_hypervector *right) {
    hdc_similarity_result similarity;
    CHECK_STATUS(hdc_similarity(left, right, &similarity), HDC_OK);

    CHECK_EQ_U32(similarity.overlap, reference_overlap(left, right));
    CHECK_EQ_U32(similarity.left_weight, reference_weight(left));
    CHECK_EQ_U32(similarity.right_weight, reference_weight(right));
}

static void test_similarity_matches_a_host_reference(void) {
    CASE("similarity matches host reference");

    uint64_t seed = 977;

    /* Sparse against sparse: the regime the library targets, where an overlap is a handful
     * of bits and often zero. */
    for (int trial = 0; trial < 4; ++trial) {
        hdc_hypervector left, right;
        hdc_random(&left, HDC_MAX_STORABLE_LANES, &seed);
        hdc_random(&right, HDC_MAX_STORABLE_LANES, &seed);
        check_similarity_against_reference(&left, &right);
    }

    /* Dense against dense: both codes fill all 48 storable lanes, so each carries roughly
     * 768 set bits and they share roughly 384. This is the count the device has to get
     * right and no other case here exercises. */
    for (int trial = 0; trial < 4; ++trial) {
        hdc_hypervector left, right;
        hdc_random_dense(&left, HDC_MAX_STORABLE_LANES, &seed);
        hdc_random_dense(&right, HDC_MAX_STORABLE_LANES, &seed);

        /* Guard the guard: a reference that reported nothing would agree with a device that
         * computed nothing, and the case would pass while testing neither. */
        CHECK(reference_overlap(&left, &right) > 100);
        check_similarity_against_reference(&left, &right);
    }

    /* Dense against sparse, so the two weights differ by more than an order of magnitude
     * and a result that mixed them up cannot hide. */
    for (int trial = 0; trial < 2; ++trial) {
        hdc_hypervector dense, sparse;
        hdc_random_dense(&dense, HDC_MAX_STORABLE_LANES, &seed);
        hdc_random(&sparse, HDC_MAX_STORABLE_LANES, &seed);
        check_similarity_against_reference(&dense, &sparse);
        check_similarity_against_reference(&sparse, &dense);
    }

    /* A vector against itself, dense: the overlap is the whole weight, which is the largest
     * count similarity can report from a storable pair. */
    hdc_hypervector self;
    hdc_random_dense(&self, HDC_MAX_STORABLE_LANES, &seed);
    check_similarity_against_reference(&self, &self);
}

static void test_unrelated_symbols_are_nearly_orthogonal(void) {
    CASE("symbols are near-orthogonal");

    /* The property the whole of HDC rests on: independently drawn symbols barely overlap,
     * so one bundle can hold many of them without confusing them. 20 bits out of 4096
     * twice over gives an expected 0.1 shared bits per pair, about 2 across 20 pairs. The
     * bound is well above that and far below the 400 two identical vectors would give. */
    uint64_t seed = 151;
    uint32_t total_overlap = 0;
    for (int trial = 0; trial < 20; ++trial) {
        hdc_hypervector left, right;
        hdc_similarity_result similarity;
        hdc_random(&left, 20, &seed);
        hdc_random(&right, 20, &seed);
        CHECK_STATUS(hdc_similarity(&left, &right, &similarity), HDC_OK);
        total_overlap += similarity.overlap;
    }

    CHECK(total_overlap <= 10);
    if (total_overlap > 10) printf("       (20 pairs overlapped %u times in total)\n", total_overlap);
}

static void test_xor_binding_does_not_decorrelate(void) {
    CASE("xor binding does not decorrelate");

    /* Recorded because it is a real limit of this algebra, not an accident. In dense BSC,
     * XOR moves the result far from both operands, which is what lets a bundle of bound
     * pairs be probed one pair at a time. Two sparse vectors rarely share a bit, so their
     * XOR is very nearly their union -- and a union stays strongly similar to each member.
     * Binding here is exact and invertible, but it is not a randomiser. */
    uint64_t seed = 161;
    hdc_hypervector role, filler, bound;
    hdc_random(&role, 20, &seed);
    hdc_random(&filler, 20, &seed);

    CHECK_STATUS(hdc_bind(&role, &filler, &bound), HDC_OK);

    hdc_similarity_result against_role;
    CHECK_STATUS(hdc_similarity(&bound, &role, &against_role), HDC_OK);
    CHECK(hdc_cosine(&against_role) > 0.5);
    CHECK(!same(&bound, &role));
    CHECK(!same(&bound, &filler));
}

/* ---- train and classify ---------------------------------------------------------- */

static void test_a_prototype_is_the_bundle_of_its_examples(void) {
    CASE("prototype is a bundle");

    uint32_t first_bits[] = {1, 40};
    uint32_t second_bits[] = {40, 100};
    uint32_t expected_bits[] = {1, 40, 100};
    hdc_hypervector first = bits(first_bits, 2);
    hdc_hypervector second = bits(second_bits, 2);
    hdc_hypervector expected = bits(expected_bits, 3);

    const char *labels[] = {"cat", "cat"};
    const hdc_hypervector *vectors[] = {&first, &second};
    hdc_memory *memory = NULL;

    CHECK_STATUS(hdc_train(labels, vectors, 2, &memory), HDC_OK);
    CHECK_EQ_U32(hdc_memory_count(memory), 1);
    CHECK(hdc_memory_prototype(memory, 0) != NULL && same(hdc_memory_prototype(memory, 0), &expected));
    CHECK(strcmp(hdc_memory_label(memory, 0), "cat") == 0);
    hdc_memory_free(memory);
}

static void test_labels_keep_first_seen_order(void) {
    CASE("label order");

    uint32_t dog_bits[] = {2};
    uint32_t cat_bits[] = {1};
    uint32_t dog_two_bits[] = {4};
    hdc_hypervector dog = bits(dog_bits, 1);
    hdc_hypervector cat = bits(cat_bits, 1);
    hdc_hypervector dog_two = bits(dog_two_bits, 1);

    const char *labels[] = {"dog", "cat", "dog"};
    const hdc_hypervector *vectors[] = {&dog, &cat, &dog_two};
    hdc_memory *memory = NULL;

    CHECK_STATUS(hdc_train(labels, vectors, 3, &memory), HDC_OK);
    CHECK_EQ_U32(hdc_memory_count(memory), 2);
    CHECK(strcmp(hdc_memory_label(memory, 0), "dog") == 0);
    CHECK(strcmp(hdc_memory_label(memory, 1), "cat") == 0);
    CHECK(hdc_memory_find(memory, "cat") == 1);
    CHECK(hdc_memory_find(memory, "fox") == -1);
    CHECK(hdc_memory_label(memory, 99) == NULL);

    hdc_memory_free(memory);

    CHECK_STATUS(hdc_train(labels, vectors, 0, &memory), HDC_ERROR_INVALID_ARGUMENT);
}

static void test_a_training_example_classifies_as_its_own_class(void) {
    CASE("classify a training example");

    uint32_t cat_bits[] = {1, 40, 100};
    uint32_t dog_bits[] = {200, 300, 400};
    hdc_hypervector cat = bits(cat_bits, 3);
    hdc_hypervector dog = bits(dog_bits, 3);

    const char *labels[] = {"cat", "dog"};
    const hdc_hypervector *vectors[] = {&cat, &dog};
    hdc_memory *memory = NULL;
    CHECK_STATUS(hdc_train(labels, vectors, 2, &memory), HDC_OK);

    size_t best = 99;
    hdc_similarity_result similarity;
    CHECK_STATUS(hdc_classify(&cat, memory, &best, &similarity), HDC_OK);
    CHECK(best < hdc_memory_count(memory) && strcmp(hdc_memory_label(memory, best), "cat") == 0);
    CHECK_EQ_U32(similarity.overlap, 3);

    hdc_memory_free(memory);
}

/* An example of a class: `drawn` bits taken from that class's own pool, plus one bit of
 * noise nothing else has. Two classes with disjoint pools give examples that resemble
 * their own class and not the other. */
static hdc_hypervector class_example(uint64_t *seed, const uint32_t *pool, uint32_t pool_size,
                                     uint32_t drawn, uint32_t noise_bit) {
    uint32_t shuffled[64];
    for (uint32_t i = 0; i < pool_size; ++i) shuffled[i] = pool[i];

    for (uint32_t i = 0; i < drawn; ++i) {
        *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t j = i + (uint32_t)((*seed >> 33) % (pool_size - i));
        uint32_t swap = shuffled[i];
        shuffled[i] = shuffled[j];
        shuffled[j] = swap;
    }

    hdc_hypervector vector;
    hdc_zero(&vector);
    for (uint32_t i = 0; i < drawn; ++i) hdc_set_bit(&vector, shuffled[i]);
    hdc_set_bit(&vector, noise_bit);
    return vector;
}

static void test_an_unseen_example_classifies_by_resemblance(void) {
    CASE("classify an unseen example");

    uint32_t cat_pool[12], dog_pool[12];
    for (uint32_t i = 0; i < 12; ++i) {
        cat_pool[i] = i * 32;
        dog_pool[i] = (16 + i) * 32;
    }

    uint64_t seed = 171;
    hdc_hypervector examples[8];
    const char *labels[8];
    const hdc_hypervector *vectors[8];

    for (uint32_t i = 0; i < 4; ++i) {
        examples[2 * i] = class_example(&seed, cat_pool, 12, 8, 3000 + i);
        labels[2 * i] = "cat";
        vectors[2 * i] = &examples[2 * i];

        examples[2 * i + 1] = class_example(&seed, dog_pool, 12, 8, 3100 + i);
        labels[2 * i + 1] = "dog";
        vectors[2 * i + 1] = &examples[2 * i + 1];
    }

    hdc_memory *memory = NULL;
    CHECK_STATUS(hdc_train(labels, vectors, 8, &memory), HDC_OK);

    hdc_hypervector unseen_cat = class_example(&seed, cat_pool, 12, 8, 3200);
    hdc_hypervector unseen_dog = class_example(&seed, dog_pool, 12, 8, 3300);

    size_t best = 0;
    CHECK_STATUS(hdc_classify(&unseen_cat, memory, &best, NULL), HDC_OK);
    CHECK(strcmp(hdc_memory_label(memory, best), "cat") == 0);

    CHECK_STATUS(hdc_classify(&unseen_dog, memory, &best, NULL), HDC_OK);
    CHECK(strcmp(hdc_memory_label(memory, best), "dog") == 0);

    hdc_memory_free(memory);
}

/* ---- argument checking ----------------------------------------------------------- */

static void test_null_arguments_are_refused(void) {
    CASE("null arguments");

    hdc_hypervector vector = one_bit_per_lane(0, 1);
    hdc_hypervector result;
    hdc_similarity_result similarity;

    CHECK_STATUS(hdc_bind(NULL, &vector, &result), HDC_ERROR_INVALID_ARGUMENT);
    CHECK_STATUS(hdc_bind(&vector, NULL, &result), HDC_ERROR_INVALID_ARGUMENT);
    CHECK_STATUS(hdc_bind(&vector, &vector, NULL), HDC_ERROR_INVALID_ARGUMENT);
    CHECK_STATUS(hdc_bundle(NULL, 1, &result), HDC_ERROR_INVALID_ARGUMENT);
    CHECK_STATUS(hdc_similarity(&vector, &vector, NULL), HDC_ERROR_INVALID_ARGUMENT);
    CHECK_STATUS(hdc_classify(&vector, NULL, NULL, &similarity), HDC_ERROR_INVALID_ARGUMENT);

    CHECK(hdc_memory_count(NULL) == 0);
    CHECK(hdc_memory_label(NULL, 0) == NULL);
    CHECK(hdc_memory_prototype(NULL, 0) == NULL);
    CHECK(hdc_memory_find(NULL, "cat") == -1);
    hdc_memory_free(NULL);

    CHECK(hdc_status_string(HDC_OK) != NULL);
    CHECK(hdc_status_string((hdc_status)999) != NULL);
}

/* ---- main ------------------------------------------------------------------------ */

/* ---- the majority vote ----------------------------------------------------------- */

/*
 * The reference the device is checked against. Deliberately written the obvious way -- count
 * the members that set each bit, keep the bit where the count passed half -- because the
 * whole point of the device version is that it reaches the same answer without a per-bit
 * counter existing anywhere in the instruction set.
 */
static hdc_hypervector reference_majority(const hdc_hypervector *const *vectors, size_t count) {
    hdc_hypervector result;
    hdc_zero(&result);

    for (uint32_t bit = 0; bit < HDC_HYPERVECTOR_BITS; ++bit) {
        size_t votes = 0;
        for (size_t i = 0; i < count; ++i)
            if (hdc_get_bit(vectors[i], bit)) ++votes;

        if (votes * 2u > count) hdc_set_bit(&result, bit);
    }

    return result;
}

static void test_majority_keeps_what_most_members_have(void) {
    CASE("majority keeps what most members have");

    /* Bit 5 is in all three, bit 9 in two, bit 40 in one. Only the first two survive. */
    uint32_t a_bits[] = {5, 9, 40};
    uint32_t b_bits[] = {5, 9};
    uint32_t c_bits[] = {5};
    hdc_hypervector a = bits(a_bits, 3), b = bits(b_bits, 2), c = bits(c_bits, 1);
    const hdc_hypervector *members[] = {&a, &b, &c};

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(members, 3, &vote), HDC_OK);

    CHECK(hdc_get_bit(&vote, 5));
    CHECK(hdc_get_bit(&vote, 9));
    CHECK(!hdc_get_bit(&vote, 40));
    CHECK_EQ_U32(hdc_weight(&vote), 2);
}

static void test_majority_of_identical_members_is_that_member(void) {
    CASE("majority of identical members is that member");

    uint64_t seed = 0x5150u;
    hdc_hypervector value;
    CHECK_STATUS(hdc_random(&value, 20, &seed), HDC_OK);

    const hdc_hypervector *members[7];
    for (size_t i = 0; i < 7; ++i) members[i] = &value;

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(members, 7, &vote), HDC_OK);
    CHECK(memcmp(vote.bytes, value.bytes, HDC_HYPERVECTOR_BYTES) == 0);
}

static void test_majority_is_not_the_union(void) {
    CASE("majority is not the union");

    /*
     * The distinction the whole operation exists for. A union of many sparse vectors keeps
     * every bit any member had; a majority keeps almost none of them, because a bit one
     * member set is not a bit most members set.
     */
    uint64_t seed = 0x1234u;
    hdc_hypervector members[9];
    const hdc_hypervector *pointers[9];
    for (size_t i = 0; i < 9; ++i) {
        CHECK_STATUS(hdc_random(&members[i], 4, &seed), HDC_OK);
        pointers[i] = &members[i];
    }

    hdc_hypervector vote, union_of;
    CHECK_STATUS(hdc_bundle_majority(pointers, 9, &vote), HDC_OK);
    CHECK_STATUS(hdc_bundle(pointers, 9, &union_of), HDC_OK);

    /* Nine vectors of four bits each over 4096 positions almost never agree anywhere, so
     * the vote is empty while the union has kept everything. */
    CHECK_EQ_U32(hdc_weight(&vote), 0);
    CHECK(hdc_weight(&union_of) > 30);
}

static void test_majority_matches_the_reference(void) {
    CASE("majority matches the reference");

    /*
     * Counts either side of HDC_BUNDLE_MAX_OPERANDS_PER_BATCH, which is 31. Anything above
     * it runs as several batches, and the counter planes have to survive from one batch to
     * the next for that to come out right -- they live in wide registers precisely because
     * a batch does not clear those. 32 and 33 are the cases that would catch it if that
     * stopped being true.
     */
    const size_t counts[] = {2, 3, 4, 5, 16, 31, 32, 33, 64};

    for (size_t index = 0; index < sizeof counts / sizeof counts[0]; ++index) {
        size_t count = counts[index];
        uint64_t seed = 0xA5A5u + count;

        hdc_hypervector *members = malloc(count * sizeof *members);
        const hdc_hypervector **pointers = malloc(count * sizeof *pointers);
        CHECK(members != NULL && pointers != NULL);
        if (members == NULL || pointers == NULL) { free(members); free(pointers); return; }

        /* Enough overlap that the vote is not trivially empty: every member draws from a
         * narrow band of positions, so bits genuinely compete. */
        for (size_t i = 0; i < count; ++i) {
            hdc_zero(&members[i]);
            for (uint32_t bit = 0; bit < 60; ++bit) {
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                if ((seed >> 33) % 2u == 0u) hdc_set_bit(&members[i], bit);
            }
            pointers[i] = &members[i];
        }

        hdc_hypervector expected = reference_majority(pointers, count);
        hdc_hypervector vote;
        CHECK_STATUS(hdc_bundle_majority(pointers, count, &vote), HDC_OK);

        if (memcmp(vote.bytes, expected.bytes, HDC_HYPERVECTOR_BYTES) != 0) {
            ++g_failures;
            printf("FAIL [%s] count %zu: device weight %u, reference weight %u\n",
                   g_case, count, hdc_weight(&vote), hdc_weight(&expected));
        }
        ++g_checks;

        free(members);
        free(pointers);
    }
}

static void test_majority_of_one_is_the_identity(void) {
    CASE("majority of one is the identity");

    uint64_t seed = 0x99u;
    hdc_hypervector only;
    CHECK_STATUS(hdc_random(&only, 12, &seed), HDC_OK);
    const hdc_hypervector *members[] = {&only};

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(members, 1, &vote), HDC_OK);
    CHECK(memcmp(vote.bytes, only.bytes, HDC_HYPERVECTOR_BYTES) == 0);
}

static void test_majority_counts_a_full_class_of_examples(void) {
    CASE("majority counts a full class of examples");

    /*
     * A class in a real training set is thousands of examples, not dozens. MNIST's largest
     * digit has 6,742 of them, and building one prototype means one vote over all of them.
     *
     * The counters have to be wide enough to hold that count. They are bit-planes, so a
     * count that overflows does not saturate -- it WRAPS, and the vote then comes back
     * confidently wrong rather than refused. With 11 planes a count of 6,000 wraps to 1,904,
     * which is below the 3,001 threshold, so every bit two thirds of the class agreed on
     * would have been dropped and the prototype would carry almost nothing.
     *
     * So this bundles a full class and checks a bit above the threshold survives while one
     * below it does not.
     */
    const size_t count = 6000;
    const size_t above = 4000;   /* two thirds of the class: a clear majority */
    const size_t below = 2000;   /* one third: a clear minority              */

    hdc_hypervector with_both, with_neither;
    hdc_zero(&with_both);
    hdc_zero(&with_neither);
    hdc_set_bit(&with_both, 7);
    hdc_set_bit(&with_both, 39);

    hdc_hypervector only_majority = with_both;
    hdc_zero(&only_majority);
    hdc_set_bit(&only_majority, 7);

    const hdc_hypervector **members = malloc(count * sizeof *members);
    CHECK(members != NULL);
    if (members == NULL) return;

    for (size_t i = 0; i < count; ++i)
        members[i] = i < below ? &with_both : (i < above ? &only_majority : &with_neither);

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(members, count, &vote), HDC_OK);

    CHECK(hdc_get_bit(&vote, 7));    /* set in 4,000 of 6,000 */
    CHECK(!hdc_get_bit(&vote, 39));  /* set in 2,000 of 6,000 */
    CHECK_EQ_U32(hdc_weight(&vote), 1);

    free(members);
}

static void test_majority_refuses_more_members_than_the_counters_hold(void) {
    CASE("majority refuses more members than the counters hold");

    /*
     * The counter planes are 13 bits wide and the kernel drops the carry out of the top
     * one, so member 8192 would wrap a saturated counter back to zero and flip bits the
     * wrong way with no sign of trouble. Refused on the host instead.
     */
    hdc_hypervector value;
    hdc_zero(&value);
    hdc_set_bit(&value, 3);

    size_t count = HDC_MAJORITY_MAX_MEMBERS + 1u;
    const hdc_hypervector **pointers = malloc(count * sizeof *pointers);
    CHECK(pointers != NULL);
    if (pointers == NULL) return;
    for (size_t i = 0; i < count; ++i) pointers[i] = &value;

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(pointers, count, &vote), HDC_ERROR_INVALID_ARGUMENT);
    free(pointers);
}

static void test_a_vote_too_dense_to_store_is_reported(void) {
    CASE("a vote too dense to store is reported");

    /*
     * The case the host cannot predict, and the reason this operation reads a status word
     * when the other four do not.
     *
     * Three members over 72 lanes. Each lane is filled in exactly two of the three, chosen
     * round-robin, so every member occupies 72 * 2/3 = 48 lanes -- exactly what a compressed
     * row holds, so each one is storable on its own. But every lane has a majority behind
     * it, so the vote occupies all 72 and does not fit.
     *
     * That is the whole argument in one fixture: a majority's lane occupancy is not a
     * function of its members' lane occupancy, so no amount of host-side inspection settles
     * it in advance. hdc_bundle can predict its overflow for free; this cannot.
     */
    const uint32_t lanes = 72;
    hdc_hypervector members[3];
    const hdc_hypervector *pointers[3];
    for (size_t i = 0; i < 3; ++i) {
        hdc_zero(&members[i]);
        pointers[i] = &members[i];
    }

    for (uint32_t lane = 0; lane < lanes; ++lane) {
        size_t first = lane % 3u;
        size_t second = (first + 1u) % 3u;
        for (uint32_t bit = 0; bit < 32u; ++bit) {
            hdc_set_bit(&members[first], lane * 32u + bit);
            hdc_set_bit(&members[second], lane * 32u + bit);
        }
    }

    for (size_t i = 0; i < 3; ++i) {
        CHECK_EQ_U32(hdc_nonzero_lanes(&members[i]), HDC_MAX_STORABLE_LANES);
        CHECK(hdc_fits_device(&members[i]));
    }

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(pointers, 3, &vote), HDC_ERROR_TOO_DENSE);

    /* And the same three members bundled as a union are refused too, but by the host
     * before anything is sent -- the two paths reach the same verdict by different routes. */
    hdc_hypervector union_of;
    CHECK_STATUS(hdc_bundle(pointers, 3, &union_of), HDC_ERROR_TOO_DENSE);
}

static void test_a_vote_that_stays_inside_the_lanes_is_stored(void) {
    CASE("a vote that stays inside the lanes is stored");

    /* The other half of the pair: members dense within the first 48 lanes vote to something
     * dense within those same 48 lanes, which fits. Dense data is not the problem; dense
     * data spread over too many lanes is. */
    uint64_t seed = 0x7777u;
    hdc_hypervector members[5];
    const hdc_hypervector *pointers[5];
    for (size_t i = 0; i < 5; ++i) {
        CHECK_STATUS(hdc_random_dense(&members[i], HDC_MAX_STORABLE_LANES, &seed), HDC_OK);
        CHECK(hdc_fits_device(&members[i]));
        pointers[i] = &members[i];
    }

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(pointers, 5, &vote), HDC_OK);
    CHECK(hdc_nonzero_lanes(&vote) <= HDC_MAX_STORABLE_LANES);

    hdc_hypervector expected = reference_majority(pointers, 5);
    CHECK(memcmp(vote.bytes, expected.bytes, HDC_HYPERVECTOR_BYTES) == 0);
}

/*
 * A majority over three members resolves each position by "what most of them had", and the
 * third member decides every position the first two disagree on. That is exactly the
 * multiplexer torchhd's BSC bundle is defined as -- `where(a == b, a, tiebreak)` -- so
 * torchhd-sparsr computes its bundle by calling hdc_bundle_majority() with three members
 * rather than carrying a kernel of its own.
 *
 * The identity is worth pinning here because nothing else would notice it breaking. The
 * threshold is `total / 2 + 1`, so a change to "at least half" would still pass every other
 * majority test in this file -- 3/2 + 1 and 3/2 are 2 and 1 -- and would silently turn
 * torchhd's bundle into a union.
 */
static void test_majority_of_three_selects_between_two(void) {
    CASE("majority of three selects between two");

    uint32_t a_bits[] = {0, 1, 10, 21};
    uint32_t b_bits[] = {0, 1, 11, 20};
    uint32_t tiebreak_bits[] = {10, 11, 99};
    hdc_hypervector a = bits(a_bits, 4);
    hdc_hypervector b = bits(b_bits, 4);
    hdc_hypervector tiebreak = bits(tiebreak_bits, 3);
    const hdc_hypervector *members[] = {&a, &b, &tiebreak};

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(members, 3, &vote), HDC_OK);

    /* Where a and b agree, the vote is what they agreed on -- the tiebreak is outvoted, so
     * bit 99 does not survive being set in it alone. Where they disagree, the tiebreak
     * decides: it holds 10 and 11 and not 20 or 21. */
    uint32_t expected_bits[] = {0, 1, 10, 11};
    hdc_hypervector expected = bits(expected_bits, 4);
    CHECK(same(&vote, &expected));

    /* Stated again as the property, over random members, so it holds beyond the hand-picked
     * case: agreement is preserved and disagreement is handed to the third member. */
    uint64_t seed = 0xB5C3u;
    for (int trial = 0; trial < 8; ++trial) {
        CHECK_STATUS(hdc_random(&a, 12, &seed), HDC_OK);
        CHECK_STATUS(hdc_random(&b, 12, &seed), HDC_OK);
        CHECK_STATUS(hdc_random(&tiebreak, 12, &seed), HDC_OK);
        CHECK_STATUS(hdc_bundle_majority(members, 3, &vote), HDC_OK);

        for (uint32_t bit = 0; bit < HDC_HYPERVECTOR_BITS; ++bit) {
            int left = hdc_get_bit(&a, bit);
            int right = hdc_get_bit(&b, bit);
            int selected = (left == right) ? left : hdc_get_bit(&tiebreak, bit);
            if (hdc_get_bit(&vote, bit) != selected) {
                CHECK_EQ_U32(bit, HDC_HYPERVECTOR_BITS); /* prints the offending position */
                break;
            }
        }
        ++g_checks;
    }
}

/*
 * The ripple is one instruction per plane, not two.
 *
 * A full-adder stage is `carry_out = plane AND carry_in` and `plane = plane XOR carry_in`
 * over the same pair of operands, and `WCSA` is exactly that pair in one instruction. The
 * answer is identical either way, so no behavioural test in this file can tell the two
 * apart -- which is why this one reads the compiled kernel instead.
 *
 * It matters because the accumulate loop is the hottest path in the library: it runs once
 * per member, and a member is one training example. Dropping back to the WAND/WXOR pair
 * would roughly double what a bundle costs and every other test here would still pass.
 *
 * The check is on the image the library actually sends to the device, so it fails if the
 * kernel stops using WCSA, and equally if the build stops compiling the kernel that does.
 */
static void test_the_ripple_is_fused(void) {
    CASE("the ripple is fused");

    /* Every kernel image is a real image, so a generator that emitted an empty array
     * cannot make the count below pass by having nothing to count. */
    CHECK(sizeof hdc_kernel_bind_image / sizeof hdc_kernel_bind_image[0] > 0);
    CHECK(sizeof hdc_kernel_intersect_image / sizeof hdc_kernel_intersect_image[0] > 0);
    CHECK(sizeof hdc_kernel_bundle_image / sizeof hdc_kernel_bundle_image[0] > 0);
    CHECK(sizeof hdc_kernel_majority_image / sizeof hdc_kernel_majority_image[0] > 0);

    /* custom-0 with funct3 = WMULTI is the multi-result class, and WCSA is the only
     * instruction in it. Decoding rather than pattern-matching a whole word keeps this
     * insensitive to which registers the kernel happens to name. */
    size_t fused = 0;
    size_t words = sizeof hdc_kernel_majority_image / sizeof hdc_kernel_majority_image[0];
    for (size_t i = 0; i < words; ++i) {
        uint32_t word = hdc_kernel_majority_image[i];
        if ((word & 0x7Fu) == SPARSR_CUSTOM0 && ((word >> 12) & 0x7u) == SPARSR_F3_WMULTI) ++fused;
    }

    /* One per counter plane. The ripple is unrolled by macro, because a wide register
     * number is an instruction immediate and cannot come from a loop variable, so the
     * stages are countable in the image. */
    CHECK_EQ_U32((uint32_t)fused, HDC_MAJORITY_COUNTER_PLANES);
}

/*
 * The vote is right whether the carry dies early or runs to the top plane.
 *
 * Cutting the ripple short once the carry is empty is a real change of control flow, and
 * the depth the carry reaches is a property of the data: it is the OR of all 4096 counters,
 * so it only empties once every one of them has stopped carrying. The cases below reach
 * different depths on purpose, and every one of them is checked against the host reference.
 *
 * `identical` is the worst case and the one an early exit must not get wrong -- every
 * member sets the same bits, so those counters carry at every plane and the ripple runs the
 * whole way up on most members.
 */
static void test_majority_is_right_at_every_carry_depth(void) {
    CASE("majority is right at every carry depth");

    /*
     * The carry a member injects is the member itself, and it halves per plane, so its
     * depth is about log2(weight): one bit dies at the first plane, forty last about five.
     * The weights stop at HDC_MAX_STORABLE_LANES because hdc_random scatters its bits over
     * all 4096 positions, so a heavier member occupies more lanes than a row holds and is
     * refused before any of this runs. The dense block below reaches the deeper carries.
     */
    const uint32_t weights[] = {1, 2, 5, 40};
    const size_t counts[] = {3, 9, 33, 70};

    for (size_t w = 0; w < sizeof weights / sizeof weights[0]; ++w) {
        for (size_t c = 0; c < sizeof counts / sizeof counts[0]; ++c) {
            size_t count = counts[c];
            uint64_t seed = 0xC5A0u + weights[w] * 131u + count;

            hdc_hypervector *members = malloc(count * sizeof *members);
            const hdc_hypervector **pointers = malloc(count * sizeof *pointers);
            CHECK(members != NULL && pointers != NULL);
            if (members == NULL || pointers == NULL) { free(members); free(pointers); return; }

            for (size_t i = 0; i < count; ++i) {
                CHECK_STATUS(hdc_random(&members[i], weights[w], &seed), HDC_OK);
                pointers[i] = &members[i];
            }

            hdc_hypervector expected = reference_majority(pointers, count);
            hdc_hypervector vote;
            CHECK_STATUS(hdc_bundle_majority(pointers, count, &vote), HDC_OK);
            CHECK(same(&vote, &expected));

            free(members);
            free(pointers);
        }
    }

    /*
     * The dense regime, which is where the carry runs deepest. A code confined to 48 lanes
     * carries about 768 set bits, so the carry it injects survives roughly ten of the
     * thirteen planes -- and it is storable at any density, because a row charges for lanes
     * and not for bits.
     */
    for (size_t count = 3; count <= 35; count += 16) {
        uint64_t dense_seed = 0x5EED0u + count;

        hdc_hypervector *members = malloc(count * sizeof *members);
        const hdc_hypervector **pointers = malloc(count * sizeof *pointers);
        CHECK(members != NULL && pointers != NULL);
        if (members == NULL || pointers == NULL) { free(members); free(pointers); return; }

        for (size_t i = 0; i < count; ++i) {
            CHECK_STATUS(hdc_random_dense(&members[i], HDC_MAX_STORABLE_LANES, &dense_seed), HDC_OK);
            pointers[i] = &members[i];
        }

        hdc_hypervector expected = reference_majority(pointers, count);
        hdc_hypervector dense_vote;
        CHECK_STATUS(hdc_bundle_majority(pointers, count, &dense_vote), HDC_OK);
        CHECK(same(&dense_vote, &expected));

        free(members);
        free(pointers);
    }

    /*
     * Members drawn from a narrow band of 60 positions. Every counter in the band is hit by
     * about half the members, so the counters climb high and the carry runs deep -- while
     * the vote stays inside two lanes and is comfortably storable. This is the case a
     * too-eager exit test would get wrong without any of the cases above noticing.
     */
    for (size_t count = 3; count <= 99; count += 32) {
        uint64_t band_seed = 0x2B4Du + count;

        hdc_hypervector *members = malloc(count * sizeof *members);
        const hdc_hypervector **pointers = malloc(count * sizeof *pointers);
        CHECK(members != NULL && pointers != NULL);
        if (members == NULL || pointers == NULL) { free(members); free(pointers); return; }

        for (size_t i = 0; i < count; ++i) {
            hdc_zero(&members[i]);
            for (uint32_t bit = 0; bit < 60; ++bit) {
                band_seed = band_seed * 6364136223846793005ull + 1442695040888963407ull;
                if ((band_seed >> 33) % 2u == 0u) hdc_set_bit(&members[i], bit);
            }
            pointers[i] = &members[i];
        }

        hdc_hypervector expected = reference_majority(pointers, count);
        hdc_hypervector band_vote;
        CHECK_STATUS(hdc_bundle_majority(pointers, count, &band_vote), HDC_OK);
        CHECK(same(&band_vote, &expected));

        free(members);
        free(pointers);
    }

    /* Every member identical: the carry reaches the top plane on nearly every member, so
     * an exit test that fired early would drop counts and lose the vote. */
    uint64_t seed = 0x7C1Du;
    hdc_hypervector shared;
    CHECK_STATUS(hdc_random(&shared, 40, &seed), HDC_OK);

    const size_t identical_count = 64;
    const hdc_hypervector **identical = malloc(identical_count * sizeof *identical);
    CHECK(identical != NULL);
    if (identical == NULL) return;
    for (size_t i = 0; i < identical_count; ++i) identical[i] = &shared;

    hdc_hypervector vote;
    CHECK_STATUS(hdc_bundle_majority(identical, identical_count, &vote), HDC_OK);
    CHECK(same(&vote, &shared));
    free(identical);
}

int main(void) {
    hdc_status started = hdc_init();
    if (started != HDC_OK) {
        printf("FAIL: hdc_init returned %s\n", hdc_status_string(started));
        printf("      The kernels are RV32I, so this needs SPARSR_BACKEND=vm.\n");
        printf("      init also probes the population-count reduce and refuses a backend\n");
        printf("      that cannot run it, so a wide ALU alone is not enough.\n");
        return 1;
    }

    test_hypervector_layout();
    test_weight_is_not_lane_count();
    test_storability_rule();
    test_random_is_reproducible_and_varied();
    test_dense_codes_fit_when_confined_to_lanes();

    test_bind_is_symmetric_difference();
    test_bind_is_commutative_and_self_inverse();
    test_bind_output_may_alias_an_input();
    test_bind_refuses_a_dense_operand();

    test_bundle_is_the_union();
    test_bundle_is_order_independent_and_idempotent();
    test_a_member_stays_contained_in_the_bundle();
    test_bundle_beyond_one_batch();
    test_bundle_past_capacity_is_reported();

    test_majority_keeps_what_most_members_have();
    test_majority_of_identical_members_is_that_member();
    test_majority_is_not_the_union();
    test_majority_matches_the_reference();
    test_majority_of_one_is_the_identity();
    test_majority_counts_a_full_class_of_examples();
    test_majority_refuses_more_members_than_the_counters_hold();
    test_a_vote_too_dense_to_store_is_reported();
    test_a_vote_that_stays_inside_the_lanes_is_stored();
    test_majority_of_three_selects_between_two();
    test_the_ripple_is_fused();
    test_majority_is_right_at_every_carry_depth();

    test_similarity_counts_the_shared_bits();
    test_similarity_endpoints();
    test_similarity_matches_a_host_reference();
    test_unrelated_symbols_are_nearly_orthogonal();
    test_xor_binding_does_not_decorrelate();

    test_a_prototype_is_the_bundle_of_its_examples();
    test_labels_keep_first_seen_order();
    test_a_training_example_classifies_as_its_own_class();
    test_an_unseen_example_classifies_by_resemblance();

    test_null_arguments_are_refused();

    hdc_shutdown();

    printf("%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) printf("TEST PASSED\n");
    return g_failures == 0 ? 0 : 1;
}
