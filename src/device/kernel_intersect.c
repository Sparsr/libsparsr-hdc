/*
 * Similarity, on the device: how many bit positions two hypervectors both set.
 *
 * This kernel used to compute only half of that. It ANDed the two operands into a wide
 * register, stored the 4096-bit result to a CMEM row, and left the host to read all of it
 * back and count the bits. The count was the cheap part; the read was the most expensive
 * thing this library did.
 *
 * It is now one instruction. The population-count reduce is a mode on the wide ALU's result
 * bus rather than a separate operation, so `popcount(a AND b)` is a single wide instruction
 * that writes an ordinary 32-bit register -- and a store of that register is four bytes of
 * data memory instead of a compressed row. The host reads one word.
 *
 * WHY IT COULD NOT BE WRITTEN THIS WAY BEFORE. The mode ran on the Sparsr VM well before
 * the SDK exposed a way to ask for it from C: SPARSR_WOP_R pinned the mode field to zero and
 * declared no output operand, so the header defined SPARSR_WMODE_POPCOUNT and could emit
 * nothing that used it. A later SDK release added _sparsr_wreduce and the named helpers
 * over it, of which _sparsr_woverlap is exactly this. Hand-rolling the `.insn` here was
 * deliberately refused in the meantime: it would have put a second instruction encoder in
 * the tree, to go stale the next time an encoding moved.
 *
 * NOTHING HERE CAN FAULT. There is no store to a compressed row any more, which was the one
 * instruction in this kernel that could refuse -- so the operand rows the host just wrote
 * are loaded, reduced, and the answer is a word in data memory. The only way this kernel
 * fails is a backend that does not implement the reduce mode at all, and hdc_init() settles
 * that once for the process rather than checking it on every call. See hdc_device_layout.h.
 *
 * The two operand weights hdc_similarity() also reports are NOT computed here. They are
 * counts of data the host already owns and has in cache, so the device would spend two more
 * wide instructions and two more data words to tell the host something it can work out
 * without a transfer. Only the overlap needs the device, because only the overlap needs both
 * vectors to meet.
 */

#include "sparsr_intrinsics.h"

#include "hdc_device_layout.h"

/* Placed at a fixed DMEM address by the link line, the same way the bundle and majority
 * kernels are given their words. Must match HDC_DMEM_SIMILARITY_OVERLAP_ADDRESS. */
extern volatile uint32_t hdc_similarity_overlap[];

void kernel_main(void) {
    _sparsr_wl(HDC_WIDE_ACCUMULATOR, HDC_ROW_LEFT);
    _sparsr_wl(HDC_WIDE_RIGHT, HDC_ROW_RIGHT);

    hdc_similarity_overlap[0] = _sparsr_woverlap(HDC_WIDE_ACCUMULATOR, HDC_WIDE_RIGHT);
}
