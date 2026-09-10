/*
 * Binding, on the device: the exclusive-OR of two hypervectors.
 *
 * Three wide instructions and a return. Binding is what pairs a role with a filler in a
 * Vector Symbolic Architecture, and XOR is its own inverse, so the same kernel unbinds.
 *
 * This kernel is loaded once, by hdc_init(), and stays in instruction memory for the life
 * of the process. Every hdc_bind() call then writes its two operands and starts the batch,
 * rather than re-sending seven words that are already there.
 */

#include "sparsr_intrinsics.h"

#include "hdc_device_layout.h"

void kernel_main(void) {
    _sparsr_wl(HDC_WIDE_ACCUMULATOR, HDC_ROW_LEFT);
    _sparsr_wl(HDC_WIDE_RIGHT, HDC_ROW_RIGHT);

    _sparsr_wxor(HDC_WIDE_RESULT, HDC_WIDE_ACCUMULATOR, HDC_WIDE_RIGHT);

    _sparsr_ws(HDC_WIDE_RESULT, HDC_ROW_RESULT);
}
