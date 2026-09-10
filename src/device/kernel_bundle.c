/*
 * Bundling, on the device: fold N hypervectors into their union.
 *
 * This one is a loop, and the loop is the point. The host writes the operand count into
 * data memory and starts the batch; the kernel folds that many rows without the host
 * sending an instruction per operand. One pre-loaded kernel therefore serves a bundle of
 * any size, instead of a differently unrolled program for every count.
 *
 * The loop needs a CMEM row that varies at run time, which is what the register-indirect
 * wide load gives it -- `_sparsr_wlr` takes a base the compiler puts in a register, where
 * `_sparsr_wl` bakes the row into the instruction word. This is the addressing that used
 * to be the software-emulator-only WLR; since the encoding freeze there is one wide load
 * and `rs1 = x0` is the absolute form of it.
 *
 * The accumulator lives in a CMEM row rather than staying in a wide register between
 * batches, so a bundle longer than CMEM holds can run as several batches and pick up where
 * it left off. OR is associative, so that changes no answer.
 *
 * Nothing here bounds the count: the host guarantees it, since it is the host that chose
 * how many rows to fill. A count larger than CMEM would read rows the host never wrote,
 * which is a bug in the host and not something the kernel can detect.
 */

#include "sparsr_intrinsics.h"

#include "hdc_device_layout.h"

/* Placed at a fixed DMEM address by the link line, which is how the SDK's own tests give a
 * kernel a known place to read from. */
extern volatile uint32_t hdc_operand_count[];

void kernel_main(void) {
    uint32_t count = hdc_operand_count[0];

    _sparsr_wl(HDC_WIDE_ACCUMULATOR, HDC_ROW_LEFT);

    for (uint32_t i = 0; i < count; ++i) {
        _sparsr_wlr(HDC_WIDE_RIGHT, HDC_ROW_BUNDLE_FIRST + i, 0);
        _sparsr_wor(HDC_WIDE_ACCUMULATOR, HDC_WIDE_ACCUMULATOR, HDC_WIDE_RIGHT);
    }

    _sparsr_ws(HDC_WIDE_ACCUMULATOR, HDC_ROW_LEFT);
}
