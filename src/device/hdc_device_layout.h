/*
 * The device resources this library uses, named once so the host and the kernels cannot
 * disagree about them.
 *
 * Both sides include this file: the host to know where to put operands and find results,
 * the kernels to know where to load and store. A row number that appeared twice, once in a
 * kernel and once in the host, would be a silent wrong answer rather than a build error.
 *
 * WHAT THIS RESERVES, AND WHY THAT IS A PROBLEM TODAY
 *
 * Nothing arbitrates access to a Sparsr device. There is no allocator for CMEM rows and
 * none for instruction memory, so a library that wants either simply takes it and hopes
 * nothing else wanted the same. This library takes CMEM rows 0 to 31 -- every row there is
 * -- and the front of instruction memory for its kernels, which it holds from
 * hdc_init() until hdc_shutdown().
 *
 * That is fine while it is the only thing talking to the device, and wrong the moment it is
 * not: an application's own kernel is loaded at word 0 by default, and the two would
 * overwrite each other with no error on either side. A host-side memory manager would fix
 * it; until one exists, this header is the whole of the contract, and it is
 * written down rather than assumed.
 *
 * torchhd-sparsr used to be the other half of that collision, keeping PyTorch tensors
 * resident in these same rows. It was made a caller of this library instead of a peer,
 * so it now holds its tensors on the host and this library owns CMEM alone.
 */

#ifndef SPARSR_HDC_DEVICE_LAYOUT_H
#define SPARSR_HDC_DEVICE_LAYOUT_H

/* CMEM rows. The two-operand kernels use LEFT, RIGHT and RESULT; the bundle kernel uses
 * LEFT as its running accumulator and RIGHT upwards as its operands. The two never run at
 * the same time, so the overlap costs nothing and keeps the reserved window small. */
#define HDC_ROW_LEFT    0u
#define HDC_ROW_RIGHT   1u
#define HDC_ROW_RESULT  2u

/* The first CMEM row a bundle takes its operands from, and how many it can hold at once.
 * CMEM has 32 rows and the accumulator keeps one, so a longer bundle runs as several
 * batches. OR is associative, so splitting it changes no answer. */
#define HDC_ROW_BUNDLE_FIRST  HDC_ROW_RIGHT
#define HDC_BUNDLE_MAX_OPERANDS_PER_BATCH  31u

/*
 * THE SAME FOUR WORDS, NAMED TWO WAYS.
 *
 * Data memory sits at 0x8000_0000 in the flat address map, so that is what
 * a kernel's pointer holds and what the link line's --defsym has to say. The host does not
 * use flat addresses: sparsr_write_data_dmem and sparsr_read_data_dmem take a
 * region-relative word index, and that ABI is unchanged. Both forms are derived from the
 * same offset below so the two cannot drift apart.
 *
 * Getting this wrong is not a silent wrong answer any more, which is the point of the map:
 * a kernel that reads offset 0x0 as if it were a flat address reads instruction memory,
 * and the device traps the batch instead of handing back whatever was there.
 */
#define HDC_DMEM_BASE  0x80000000u

/* Byte offset in DMEM of the word holding how many operands the bundle kernel should
 * fold in. The host writes it before each batch. */
#define HDC_DMEM_OPERAND_COUNT_OFFSET  0x0u

/* Byte offset in DMEM of the word the majority kernel is told to do, and the word it
 * writes back. See "why only this kernel reports a status" below. */
#define HDC_DMEM_MAJORITY_MODE_OFFSET    0x4u
#define HDC_DMEM_MAJORITY_TOTAL_OFFSET   0x8u
#define HDC_DMEM_MAJORITY_STATUS_OFFSET  0xCu

/* Byte offset in DMEM of the overlap the intersect kernel reduces to. This is the whole
 * result of a similarity: one scalar, not a 4096-bit row the host has to count. */
#define HDC_DMEM_SIMILARITY_OVERLAP_OFFSET  0x10u

/* What a kernel names: the flat address. These must match the --defsym values in the
 * Makefile, which is the one place the linker learns them. */
#define HDC_DMEM_OPERAND_COUNT_ADDRESS    (HDC_DMEM_BASE + HDC_DMEM_OPERAND_COUNT_OFFSET)
#define HDC_DMEM_MAJORITY_MODE_ADDRESS    (HDC_DMEM_BASE + HDC_DMEM_MAJORITY_MODE_OFFSET)
#define HDC_DMEM_MAJORITY_TOTAL_ADDRESS   (HDC_DMEM_BASE + HDC_DMEM_MAJORITY_TOTAL_OFFSET)
#define HDC_DMEM_MAJORITY_STATUS_ADDRESS  (HDC_DMEM_BASE + HDC_DMEM_MAJORITY_STATUS_OFFSET)
#define HDC_DMEM_SIMILARITY_OVERLAP_ADDRESS (HDC_DMEM_BASE + HDC_DMEM_SIMILARITY_OVERLAP_OFFSET)

/* The first DMEM byte a kernel's own data may use. The five control words above sit below
 * it, and the Makefile passes this to the linker as __bss_origin so a kernel's .bss cannot
 * land on top of them -- the startup code clears .bss at the start of every batch, and it
 * would otherwise wipe the operand count the host had just written. Adding a control word
 * means raising this and the --defsym in the Makefile together. */
#define HDC_DMEM_KERNEL_DATA_OFFSET  0x14u

/* What the host names: the region-relative word index. */
#define HDC_DMEM_OPERAND_COUNT_WORD    (HDC_DMEM_OPERAND_COUNT_OFFSET / 4u)
#define HDC_DMEM_MAJORITY_MODE_WORD    (HDC_DMEM_MAJORITY_MODE_OFFSET / 4u)
#define HDC_DMEM_MAJORITY_TOTAL_WORD   (HDC_DMEM_MAJORITY_TOTAL_OFFSET / 4u)
#define HDC_DMEM_MAJORITY_STATUS_WORD  (HDC_DMEM_MAJORITY_STATUS_OFFSET / 4u)
#define HDC_DMEM_SIMILARITY_OVERLAP_WORD  (HDC_DMEM_SIMILARITY_OVERLAP_OFFSET / 4u)

/* No overlap can be this, since a hypervector is 4096 bits, so it says "the kernel did not
 * write here". hdc_init() puts it in before its one probe batch, which is what turns a
 * backend that cannot run the reduce into an init failure rather than an overlap of zero. */
#define HDC_SIMILARITY_OVERLAP_UNSET  0xFFFFFFFFu

/* What the majority kernel should do this batch. */
#define HDC_MAJORITY_MODE_RESET       0u  /* zero the counter planes                       */
#define HDC_MAJORITY_MODE_ACCUMULATE  1u  /* fold in HDC_DMEM_OPERAND_COUNT_ADDRESS rows   */
#define HDC_MAJORITY_MODE_THRESHOLD   2u  /* compare against the total and store the vote  */

#define HDC_MAJORITY_STATUS_UNSET  0xFFFFFFFFu
#define HDC_MAJORITY_STATUS_OK     0u
#define HDC_MAJORITY_STATUS_TOO_DENSE  1u  /* the vote did not fit a compressed row */
#define HDC_MAJORITY_STATUS_BAD_MODE   2u

/* Wide registers. w1 is the left operand and the bundle accumulator, w2 the right operand,
 * w3 the result of a two-operand kernel. */
#define HDC_WIDE_ACCUMULATOR  SPARSR_W1
#define HDC_WIDE_RIGHT        SPARSR_W2
#define HDC_WIDE_RESULT       SPARSR_W3

/*
 * The majority vote's counter planes: a contiguous block of wide registers, w4 upwards.
 *
 * WHY WIDE REGISTERS AND NOT CMEM ROWS. Counting set bits per position across N vectors
 * needs ceil(log2(N+1)) bit-planes, and a counter plane is dense by construction -- about
 * half its bits are set whatever the inputs looked like. Dense data does not fit a
 * compressed CMEM row, so there is nowhere else for the planes to go. That turns out to be
 * the better place anyway: a batch is a call and not a reset, so a wide register keeps its
 * value across batches, and the accumulator survives a bundle longer than CMEM holds
 * without spending a single row. `hdc_bundle`'s accumulator round-trips through a row only
 * because a union needs no state wider than one vector.
 *
 * Thirteen planes count to 8191, which is HDC_MAJORITY_MAX_COUNT. Each plane costs one
 * register and one wide instruction per operand -- WCSA fuses the full adder's two halves --
 * and a plane is skipped entirely once the ripple carry has emptied.
 *
 * WHY THIRTEEN AND NOT ELEVEN. Eleven planes counted to 2047, which is a plausible-looking
 * number and too small for the job this operation exists to do: a class prototype is a vote
 * over every training example of that class, and a real training set has thousands. MNIST's
 * largest digit has 6,742. Worse, an overflow here does not saturate -- bit-planes wrap, so
 * a count of 6,000 read back as 1,904, fell under the threshold, and the prototype came back
 * confidently empty rather than refused. The host refuses anything past HDC_MAJORITY_MAX_COUNT
 * so that cannot happen, but the ceiling has to be high enough to be useful first.
 */
#define HDC_MAJORITY_COUNTER_PLANES  13u
#define HDC_WIDE_COUNTER_FIRST       SPARSR_W4
#define HDC_MAJORITY_MAX_COUNT       ((1u << HDC_MAJORITY_COUNTER_PLANES) - 1u)

/* Scratch above the planes: w17 to w22.
 *
 * There are two carry registers rather than one because a full adder produces two values
 * from the same pair of operands, and WCSA writes both at once: the sum goes back into the
 * plane and the carry has to go somewhere else. With one carry register the new carry would
 * land on the old one it was computed from. Alternating between two means neither output is
 * ever copied, and there is no wide move instruction to copy with -- it would be an AND of a
 * register with itself, an extra instruction on the hottest path in the library. */
#define HDC_WIDE_CARRY     SPARSR_W17
#define HDC_WIDE_CARRY_ALT SPARSR_W18
#define HDC_WIDE_GREATER   SPARSR_W19
#define HDC_WIDE_EQUAL     SPARSR_W20
#define HDC_WIDE_ONES      SPARSR_W21
#define HDC_WIDE_TEMP      SPARSR_W22

/*
 * WHY ONLY THE MAJORITY KERNEL REPORTS A STATUS, AND THE OTHER FOUR DO NOT.
 *
 * The host ABI cannot report a device fault: sparsr_execute_batch returns void and
 * read_status carries one done bit that a faulting batch sets exactly like a successful one.
 * A kernel that wants to be honest has to write a status word into DMEM and have
 * the host read it back, which costs two extra frames per batch.
 *
 * bind, intersect and bundle do not pay that, and it is arithmetic rather than taste. Their
 * only realistic fault is a row that does not fit, and the host predicts that for free from
 * a lane-occupancy map before sending anything -- a bind is five frames, so a status read
 * would be a 40% increase to detect something already known.
 *
 * INTERSECT READS DMEM, BUT NOT FOR A STATUS. The intersect kernel now reduces to
 * a scalar overlap and the host reads it from HDC_DMEM_SIMILARITY_OVERLAP_WORD. That read
 * REPLACES the CMEM row read it used to do, so similarity costs the same frames as before
 * and moves four bytes back instead of 240. It carries no status word, and adding one would
 * make similarity more expensive than it was rather than less.
 *
 * What a per-call status would have guarded is a backend that runs the wide ALU but has not
 * built the reduce modes: the reduce faults, nothing writes the word, and the host reads a
 * stale count. That is a real risk -- the four scalar-writing modes exist on the Sparsr VM
 * and the RTL has none of them -- but it is a property of the backend and not of
 * the call, so hdc_init() probes it once with a known pair and refuses to come up if the
 * answer is wrong. One probe per process rather than one status read per similarity.
 *
 * The majority vote is the case where the host genuinely cannot predict it. The thresholded
 * result of a vote over sparse inputs is not sparse in general, and unlike a union its lane
 * occupancy is not a function of the inputs' lane occupancy: predicting it means computing
 * the vote on the host, which is the thing the device is for. So here the status word earns
 * its two frames, and it is read once per bundle rather than once per operand.
 */

#endif /* SPARSR_HDC_DEVICE_LAYOUT_H */
