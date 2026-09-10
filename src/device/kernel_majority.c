/*
 * The N-way majority vote, on the device.
 *
 * This is the operation that builds a class prototype, and it is the one HDC operation
 * Sparsr has no instruction for. `hdc_bundle` is a union: OR every member together. That is
 * exact and cheap, but it saturates -- bundle a few hundred examples of a digit and every
 * bit is set, so the prototype carries nothing. A prototype needs the *majority*: keep a
 * bit where more than half the members had it.
 *
 * WHY IT IS THIS EXPENSIVE
 *
 * A majority needs a count per bit position, and there are 4096 of them. The frozen wide
 * ISA allocates three reduce modes that would do it in the hardware -- segmented per-lane
 * counts, prefix rank, threshold-accumulate -- and none is built. So the counters
 * are built out of the instructions that do exist, as bit-planes with a carry-save adder:
 * plane j holds bit j of all 4096 counters, and adding a vector is a ripple of full adders
 * across the planes. One wide instruction per plane per operand, since WCSA fuses the
 * adder's two halves.
 *
 * That cost is the point as much as the result is. It is the number that says what a
 * dedicated reduce unit would be worth, and before this kernel existed there was no number.
 *
 * WHY THE PLANES LIVE IN WIDE REGISTERS
 *
 * A counter plane is dense by construction -- about half its bits are set whatever the
 * inputs looked like -- so it cannot be parked in a compressed CMEM row between batches.
 * Wide registers are the only place it can go, and that turns out to be the better place: a
 * batch is a call and not a reset, so the planes survive from one batch to the next and a
 * bundle longer than CMEM holds costs no rows at all. See hdc_device_layout.h.
 *
 * THE THREE MODES
 *
 * The host drives this in phases, because the operands arrive a CMEM-worth at a time:
 * reset once, accumulate as many times as it takes, threshold once. The mode word says
 * which, so one pre-loaded image serves all three rather than three images at three
 * instruction offsets.
 */

#include "sparsr_intrinsics.h"

#include "hdc_device_layout.h"

/* Placed at fixed DMEM addresses by the link line. */
extern volatile uint32_t hdc_operand_count[];
extern volatile uint32_t hdc_majority_mode[];
extern volatile uint32_t hdc_majority_total[];
extern volatile uint32_t hdc_majority_status[];

/*
 * A wide register index has to be a compile-time constant: GCC has no register class for
 * the wide file and cannot allocate, spill or index it. So every loop over the counter
 * planes is unrolled here by macro rather than written as a `for`, and the two lists below
 * are the unrolling -- ascending for the adder, descending for the comparison. Changing
 * HDC_MAJORITY_COUNTER_PLANES means changing both lists to match, which the host checks.
 */
#define HDC_PLANES_ASCENDING(MACRO)                                                        \
    MACRO(0) MACRO(1) MACRO(2) MACRO(3) MACRO(4) MACRO(5) MACRO(6)                         \
    MACRO(7) MACRO(8) MACRO(9) MACRO(10) MACRO(11) MACRO(12)

#define HDC_PLANES_DESCENDING(MACRO)                                                       \
    MACRO(12) MACRO(11) MACRO(10) MACRO(9) MACRO(8) MACRO(7)                               \
    MACRO(6) MACRO(5) MACRO(4) MACRO(3) MACRO(2) MACRO(1) MACRO(0)

/* Planes run from HDC_WIDE_COUNTER_FIRST upwards, so plane j is that register plus j. */
#define HDC_PLANE(j) ((SparsrWideReg)(HDC_WIDE_COUNTER_FIRST + (j)))

/* ---- reset: zero every plane ------------------------------------------------------- */

/* XOR of a register with itself is zero, whatever it held. No constant needed, and no
 * dependence on a CMEM row a caller would have to have written first. */
#define HDC_RESET_PLANE(j) _sparsr_wxor(HDC_PLANE(j), HDC_PLANE(j), HDC_PLANE(j));

static void majority_reset(void) {
    HDC_PLANES_ASCENDING(HDC_RESET_PLANE)
}

/* ---- accumulate: fold `count` CMEM rows into the planes ---------------------------- */

/*
 * One full-adder step per plane, one wide instruction:
 *
 *     carry_out = plane AND carry_in
 *     plane     = plane XOR carry_in
 *
 * That pair over one pair of operands is exactly what WCSA computes, so a stage is one
 * instruction and not two. The answer is identical -- this changes what
 * the ripple costs and nothing about what it produces.
 *
 * WCSA reads both sources before it writes either destination, which is what lets the sum
 * go back into the plane it came from. The carry still alternates between two registers so
 * neither output has to be copied anywhere; see hdc_device_layout.h for why that is worth a
 * register. Written as two instructions the AND had to come first, because the XOR
 * overwrote the plane the AND needed. Fused, the ordering question does not arise.
 *
 * THE RIPPLE STOPS ONCE THE CARRY IS EMPTY, AND IT IS TESTED EVERY THIRD PLANE.
 *
 * Once the carry vector holds no bits, every remaining stage is a no-op, and `_sparsr_wany`
 * asks exactly that in one OR-reduce. How often to ask is a real trade
 * and it was settled by measuring, not by arguing: a test costs two instructions -- the
 * reduce and the branch -- and each plane it lets us skip now saves one, because a stage is
 * one instruction. Testing too often loses more than it skips.
 *
 * Four periods were built and measured, all instructions counted, in retired instructions
 * per member (bench/ is the harness and the library README carries the table):
 *
 *                          68-93 sparse   68-93 dense   962-992 sparse
 *     no early exit               15.84         15.84            15.96
 *     test every plane            11.22         20.53            16.85
 *     test every 2 planes         10.02         14.84            13.48
 *     test every 3 planes          7.84         12.84            12.96   <- this kernel
 *     test every 4 planes          8.84         14.84            12.74
 *
 * Every third plane is best or within two percent of best in all three columns, so that is
 * what this does. Testing every fourth plane wins the long-bundle column by 1.7% and loses
 * the other two by 11% and 16%.
 *
 * WHY THE NUMBER DEPENDS ON THE BUNDLE. Depth is a property of the data. The carry a member
 * injects is the member itself, and it survives plane j only where that plane's bit was
 * already set, so it halves per plane. A long bundle has larger counters, so its upper
 * planes are populated and the carry runs deeper -- which is why a thousand members cost
 * more per member than eighty. The no-early-exit row is flat for the same reason in reverse:
 * it looks at no data, so it costs the same whatever the members are. That flatness is also
 * the cross-check on the whole table -- fusing removes exactly one instruction per plane, so
 * 28.84 minus thirteen planes is 15.84, and it is.
 *
 * The early exit's first job was to measure the saving. The saving is real, and it is smaller than the pre-fusing estimate of it --
 * that estimate assumed a skipped plane saved two instructions, which was true before WCSA
 * and is not true now.
 */
#define HDC_ADD_PLANE(j, CARRY_IN, CARRY_OUT)                                              \
    _sparsr_wcsa(HDC_PLANE(j), CARRY_OUT, HDC_PLANE(j), CARRY_IN);

static void majority_accumulate(uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        /* The operand is the carry into plane 0: adding a vector is adding one to every
         * counter whose bit is set. */
        _sparsr_wlr(HDC_WIDE_CARRY, HDC_ROW_BUNDLE_FIRST + i, 0);

        HDC_ADD_PLANE(0,  HDC_WIDE_CARRY,     HDC_WIDE_CARRY_ALT)
        HDC_ADD_PLANE(1,  HDC_WIDE_CARRY_ALT, HDC_WIDE_CARRY)
        HDC_ADD_PLANE(2,  HDC_WIDE_CARRY,     HDC_WIDE_CARRY_ALT)
        if (!_sparsr_wany(HDC_WIDE_CARRY_ALT)) continue;

        HDC_ADD_PLANE(3,  HDC_WIDE_CARRY_ALT, HDC_WIDE_CARRY)
        HDC_ADD_PLANE(4,  HDC_WIDE_CARRY,     HDC_WIDE_CARRY_ALT)
        HDC_ADD_PLANE(5,  HDC_WIDE_CARRY_ALT, HDC_WIDE_CARRY)
        if (!_sparsr_wany(HDC_WIDE_CARRY)) continue;

        HDC_ADD_PLANE(6,  HDC_WIDE_CARRY,     HDC_WIDE_CARRY_ALT)
        HDC_ADD_PLANE(7,  HDC_WIDE_CARRY_ALT, HDC_WIDE_CARRY)
        HDC_ADD_PLANE(8,  HDC_WIDE_CARRY,     HDC_WIDE_CARRY_ALT)
        if (!_sparsr_wany(HDC_WIDE_CARRY_ALT)) continue;

        HDC_ADD_PLANE(9,  HDC_WIDE_CARRY_ALT, HDC_WIDE_CARRY)
        HDC_ADD_PLANE(10, HDC_WIDE_CARRY,     HDC_WIDE_CARRY_ALT)
        HDC_ADD_PLANE(11, HDC_WIDE_CARRY_ALT, HDC_WIDE_CARRY)
        if (!_sparsr_wany(HDC_WIDE_CARRY)) continue;

        HDC_ADD_PLANE(12, HDC_WIDE_CARRY,     HDC_WIDE_CARRY_ALT)
        /* The carry out of the top plane is dropped. It can only be set if the count passed
         * HDC_MAJORITY_MAX_COUNT, and the host refuses a bundle that long. */
    }
}

/* ---- threshold: keep a bit where its count reached the threshold -------------------- */

/*
 * A bitwise magnitude comparison, 4096 of them at once. Each position's count is spread
 * across the planes, so the comparison walks the planes from the top down carrying two
 * masks: `greater` for positions already known to exceed the threshold, `equal` for
 * positions still tied with it.
 *
 *     threshold bit is 0:  greater |= equal AND plane  ;  equal &= NOT plane
 *     threshold bit is 1:  equal   &= plane
 *
 * and the answer is `greater OR equal`, which is "count >= threshold".
 *
 * `threshold` is a run-time value, so the choice per plane is a real branch -- but it is a
 * branch on a scalar register, taken once per plane and not once per bit. NOT is spelled
 * NOR(x, x), because WNOT is allocated by the freeze and nothing implements it.
 */
#define HDC_COMPARE_PLANE(j)                                                               \
    if (((threshold >> (j)) & 1u) == 0u) {                                                 \
        _sparsr_wand(HDC_WIDE_TEMP, HDC_WIDE_EQUAL, HDC_PLANE(j));                         \
        _sparsr_wor(HDC_WIDE_GREATER, HDC_WIDE_GREATER, HDC_WIDE_TEMP);                    \
        _sparsr_wnor(HDC_WIDE_TEMP, HDC_PLANE(j), HDC_PLANE(j));                           \
        _sparsr_wand(HDC_WIDE_EQUAL, HDC_WIDE_EQUAL, HDC_WIDE_TEMP);                       \
    } else {                                                                               \
        _sparsr_wand(HDC_WIDE_EQUAL, HDC_WIDE_EQUAL, HDC_PLANE(j));                        \
    }

static void majority_threshold(uint32_t total) {
    /* Strictly more than half, which is the majority rule: count >= floor(total/2) + 1. */
    uint32_t threshold = (total / 2u) + 1u;

    /* greater = 0, equal = all ones. XOR with self gives zero; NOR of zero with itself
     * gives all ones. Neither needs a constant fetched from anywhere. */
    _sparsr_wxor(HDC_WIDE_GREATER, HDC_WIDE_GREATER, HDC_WIDE_GREATER);
    _sparsr_wnor(HDC_WIDE_ONES, HDC_WIDE_GREATER, HDC_WIDE_GREATER);
    _sparsr_wand(HDC_WIDE_EQUAL, HDC_WIDE_ONES, HDC_WIDE_ONES);

    HDC_PLANES_DESCENDING(HDC_COMPARE_PLANE)

    _sparsr_wor(HDC_WIDE_RESULT, HDC_WIDE_GREATER, HDC_WIDE_EQUAL);
    _sparsr_ws(HDC_WIDE_RESULT, HDC_ROW_LEFT);
}

void kernel_main(void) {
    uint32_t mode = hdc_majority_mode[0];

    if (mode == HDC_MAJORITY_MODE_RESET) {
        majority_reset();
        hdc_majority_status[0] = HDC_MAJORITY_STATUS_OK;
        return;
    }

    if (mode == HDC_MAJORITY_MODE_ACCUMULATE) {
        majority_accumulate(hdc_operand_count[0]);
        hdc_majority_status[0] = HDC_MAJORITY_STATUS_OK;
        return;
    }

    if (mode == HDC_MAJORITY_MODE_THRESHOLD) {
        /*
         * The store is the one instruction here that can fault: a vote is not sparse just
         * because its inputs were, and a result of more than 48 non-zero lanes does not fit
         * a compressed row. The device refuses it rather than truncating, but the host ABI
         * cannot see that -- so the status word is written BEFORE the store, and
         * overwritten with OK after it. A refused store ends the batch, leaving TOO_DENSE
         * behind for the host to find.
         */
        hdc_majority_status[0] = HDC_MAJORITY_STATUS_TOO_DENSE;
        majority_threshold(hdc_majority_total[0]);
        hdc_majority_status[0] = HDC_MAJORITY_STATUS_OK;
        return;
    }

    hdc_majority_status[0] = HDC_MAJORITY_STATUS_BAD_MODE;
}
