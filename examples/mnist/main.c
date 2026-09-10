/*
 * Handwritten digit recognition on Sparsr, in C, using nothing but libsparsr_hdc.
 *
 * This is hyperdimensional computing (HDC), also called vector symbolic architecture. It
 * classifies MNIST digits with no gradients, no training loop and no floating point: every
 * image becomes one 4096-bit vector, every digit class becomes one 4096-bit vector, and a
 * digit is recognised by finding the class vector its image resembles most.
 *
 * WHAT RUNS WHERE
 *
 * Every vector operation in this file is a call into libsparsr_hdc, and every one of those
 * runs on the Sparsr device. This program does three things itself, and none of them is
 * arithmetic on a hypervector:
 *
 *   1. reads the MNIST files,
 *   2. decides which vectors go into which bundle,
 *   3. prints the answer.
 *
 * That is the point of the example. The algorithm lives in the library; this is a driver.
 *
 * THE ALGORITHM, IN FOUR STEPS
 *
 * 1. ITEM MEMORY. Give each of the 784 pixel positions a fixed random hypervector, drawn
 *    once and never changed. Two positions get unrelated vectors, which is what makes the
 *    encoding below carry information about *which* pixels were lit.
 *
 * 2. ENCODE. An image is the majority vote of the hypervectors of its lit pixels. A bit of
 *    the result is set where more than half of those pixels' vectors had it set. Two images
 *    of the same digit light up similar pixels, so they encode to similar vectors.
 *
 * 3. TRAIN. A class prototype is the majority vote of every training image of that digit.
 *    One pass over the data, and each class is one vector.
 *
 * 4. CLASSIFY. Encode the test image and pick the prototype it overlaps most. Each of the
 *    ten scores is one wide instruction: the intersection and its population count are the
 *    same instruction, so nothing reads a 4096-bit row back to the host to count its bits.
 *
 * WHY THE CODES ARE DENSE, AND WHY THEY ARE 48 LANES WIDE
 *
 * A hypervector reaches the device through a co-processor memory row, and a row stores at
 * most 48 non-zero four-byte lanes out of 128. The limit counts LANES, not set bits, and it
 * stores each occupied lane whole -- so bits inside an occupied lane are free.
 *
 * That makes two very different codes storable, and they are not equally good:
 *
 *   - SPARSE: at most 48 set bits, scattered anywhere. hdc_random() draws these.
 *   - DENSE:  any density at all, as long as the bits stay inside 48 lanes -- 1536 of the
 *             4096 positions. hdc_random_dense() draws these.
 *
 * This example uses the dense one, and it matters: a dense code carries about 768 bits of
 * signal where a sparse one carries 48, and it classifies far better. Every vector here --
 * item, encoded image and prototype -- stays inside those 48 lanes by construction, so
 * nothing this program builds can ever be too dense to store.
 *
 * WHAT IT COSTS
 *
 * A majority vote costs more than a union on Sparsr -- 8 to 13 instructions per member
 * against four -- because the wide instruction set has no instruction that counts per bit
 * position. The device builds 4096 counters as bit-planes with a carry-save adder instead,
 * one WCSA per plane, stopping once the carry empties. A union is cheaper, but a union
 * saturates: OR a few hundred images together and every bit is set, so the prototype says
 * nothing about the class. The vote is what a prototype needs.
 */

#include <sparsr_hdc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* MNIST: 28 x 28 greyscale images of the digits 0 to 9. */
#define IMAGE_PIXELS 784
#define DIGIT_CLASSES 10

/* A pixel counts as lit at or above mid-grey. MNIST is nearly black-and-white already, so
 * thresholding here loses very little and removes a parameter nobody wants to tune. */
#define PIXEL_LIT_THRESHOLD 128

/* Exit codes, so a script driving this can tell "no data" from "it went wrong". */
#define EXIT_NO_DATA 2
#define EXIT_FAILED 3

/* ---- reading the MNIST files ------------------------------------------------------- */

/*
 * IDX is the format the MNIST files ship in: a big-endian header, then the raw bytes. An
 * image file has a 16-byte header and then one byte per pixel; a label file has an 8-byte
 * header and then one byte per label.
 */
#define IDX_IMAGE_HEADER_BYTES 16
#define IDX_LABEL_HEADER_BYTES 8

typedef struct dataset {
    unsigned char *pixels; /* count * IMAGE_PIXELS, one byte per pixel, 0 or 1 */
    unsigned char *labels; /* count bytes, each 0 to 9                        */
    size_t count;
} dataset;

static unsigned char *read_file(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return NULL;

    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return NULL; }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return NULL; }

    unsigned char *bytes = malloc((size_t)length);
    if (bytes == NULL) { fclose(file); return NULL; }

    if (fread(bytes, 1, (size_t)length, file) != (size_t)length) {
        free(bytes);
        fclose(file);
        return NULL;
    }

    fclose(file);
    *size = (size_t)length;
    return bytes;
}

static uint32_t read_big_endian(const unsigned char *bytes) {
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | bytes[3];
}

/*
 * Both naming conventions in the wild are accepted. The files are named
 * `train-images-idx3-ubyte` where they are downloaded by hand and `train-images.idx3-ubyte`
 * by some tools, and a customer who has already got them should not have to rename anything.
 */
static unsigned char *read_idx(const char *directory, const char *stem, const char *suffix, size_t *size) {
    char path[1024];

    snprintf(path, sizeof path, "%s/%s-%s", directory, stem, suffix);
    unsigned char *bytes = read_file(path, size);
    if (bytes != NULL) return bytes;

    snprintf(path, sizeof path, "%s/%s.%s", directory, stem, suffix);
    return read_file(path, size);
}

/*
 * Loads at most `limit` images and their labels, thresholded to one bit per pixel.
 * Returns 0 and leaves `out` untouched when the files are not there.
 */
static int load_dataset(const char *directory, int training, size_t limit, dataset *out) {
    const char *image_stem = training ? "train-images" : "t10k-images";
    const char *label_stem = training ? "train-labels" : "t10k-labels";

    size_t image_size = 0, label_size = 0;
    unsigned char *image_bytes = read_idx(directory, image_stem, "idx3-ubyte", &image_size);
    unsigned char *label_bytes = read_idx(directory, label_stem, "idx1-ubyte", &label_size);

    if (image_bytes == NULL || label_bytes == NULL) {
        free(image_bytes);
        free(label_bytes);
        return 0;
    }

    if (image_size < IDX_IMAGE_HEADER_BYTES || label_size < IDX_LABEL_HEADER_BYTES) {
        fprintf(stderr, "The MNIST files in '%s' are too short to be valid.\n", directory);
        free(image_bytes);
        free(label_bytes);
        return 0;
    }

    size_t available = read_big_endian(image_bytes + 4);
    size_t labels_available = read_big_endian(label_bytes + 4);
    if (labels_available < available) available = labels_available;

    /* Refuse rather than read past the end if a file is truncated. */
    if (image_size < IDX_IMAGE_HEADER_BYTES + available * IMAGE_PIXELS ||
        label_size < IDX_LABEL_HEADER_BYTES + available) {
        fprintf(stderr, "The MNIST files in '%s' are truncated.\n", directory);
        free(image_bytes);
        free(label_bytes);
        return 0;
    }

    size_t count = limit < available ? limit : available;

    out->pixels = malloc(count * IMAGE_PIXELS);
    out->labels = malloc(count);
    if (out->pixels == NULL || out->labels == NULL) {
        fprintf(stderr, "Out of memory reading '%s'.\n", directory);
        free(out->pixels);
        free(out->labels);
        free(image_bytes);
        free(label_bytes);
        return 0;
    }

    for (size_t image = 0; image < count; ++image) {
        const unsigned char *source = image_bytes + IDX_IMAGE_HEADER_BYTES + image * IMAGE_PIXELS;
        unsigned char *target = out->pixels + image * IMAGE_PIXELS;
        for (size_t pixel = 0; pixel < IMAGE_PIXELS; ++pixel)
            target[pixel] = source[pixel] >= PIXEL_LIT_THRESHOLD ? 1u : 0u;
        out->labels[image] = label_bytes[IDX_LABEL_HEADER_BYTES + image];
    }

    out->count = count;
    free(image_bytes);
    free(label_bytes);
    return 1;
}

static void free_dataset(dataset *set) {
    free(set->pixels);
    free(set->labels);
    set->pixels = NULL;
    set->labels = NULL;
    set->count = 0;
}

/* ---- the four steps ---------------------------------------------------------------- */

/*
 * Step 1: the item memory. One fixed hypervector per pixel position, drawn once.
 *
 * `lanes` is the width of the code. Every draw is confined to the same first `lanes` lanes,
 * which is what keeps everything built out of them storable: a majority vote can only set
 * bits its members had, so a vote over these vectors occupies those lanes and no others.
 */
static hdc_hypervector *build_item_memory(uint32_t lanes, uint64_t seed) {
    hdc_hypervector *items = malloc(IMAGE_PIXELS * sizeof *items);
    if (items == NULL) return NULL;

    for (size_t pixel = 0; pixel < IMAGE_PIXELS; ++pixel) {
        hdc_status status = hdc_random_dense(&items[pixel], lanes, &seed);
        if (status != HDC_OK) {
            fprintf(stderr, "Drawing the item memory failed: %s\n", hdc_status_string(status));
            free(items);
            return NULL;
        }
    }

    return items;
}

/*
 * Step 2: encode one image.
 *
 * The lit pixels' vectors are voted on. An image lights about 150 of the 784 pixels, so this
 * is a vote over about 150 members and the result keeps the bits that most of them shared.
 *
 * `members` is scratch the caller owns, so encoding 60,000 images does not allocate 60,000
 * times.
 */
static hdc_status encode_image(const unsigned char *image,
                               const hdc_hypervector *items,
                               const hdc_hypervector **members,
                               hdc_hypervector *out) {
    size_t lit = 0;
    for (size_t pixel = 0; pixel < IMAGE_PIXELS; ++pixel)
        if (image[pixel] != 0) members[lit++] = &items[pixel];

    /* A blank image has nothing to vote on. It encodes to the zero hypervector, which
     * resembles nothing and will simply be classified badly -- which is correct. */
    if (lit == 0) {
        hdc_zero(out);
        return HDC_OK;
    }

    return hdc_bundle_majority(members, lit, out);
}

/* ---- the program -------------------------------------------------------------------- */

typedef struct options {
    const char *data_directory;
    size_t train_images;
    size_t test_images;
    uint32_t lanes;
    uint64_t seed;
} options;

static void print_usage(const char *program) {
    printf("Usage: %s [options]\n\n", program);
    printf("  --data DIR     Where the MNIST idx files are. Defaults to $SPARSR_MNIST_DIR, then ./data.\n");
    printf("  --train N      Training images to learn from. Default: all of them.\n");
    printf("  --test N       Test images to classify. Default: all of them.\n");
    printf("  --lanes N      Width of the code, in 32-bit lanes, 1 to %d. Default: %d.\n",
           HDC_MAX_STORABLE_LANES, HDC_MAX_STORABLE_LANES);
    printf("  --seed N       Seed for the item memory. Default: 1.\n");
    printf("  --help         This message.\n");
}

static void explain_missing_data(const char *directory) {
    fprintf(stderr, "Could not find the MNIST files in '%s'.\n\n", directory);
    fprintf(stderr, "This example needs the four original MNIST idx files:\n");
    fprintf(stderr, "  train-images-idx3-ubyte   train-labels-idx1-ubyte\n");
    fprintf(stderr, "  t10k-images-idx3-ubyte    t10k-labels-idx1-ubyte\n\n");
    fprintf(stderr, "They are about 55 MB in total and are not shipped with this example.\n");
    fprintf(stderr, "Put them in a directory and point this program at it:\n\n");
    fprintf(stderr, "  ./build/mnist_hdc --data /path/to/mnist\n");
    fprintf(stderr, "  SPARSR_MNIST_DIR=/path/to/mnist ./build/mnist_hdc\n\n");
    fprintf(stderr, "The files must be uncompressed. If yours end in .gz, gunzip them first.\n");
}

/*
 * Progress on one line while a person is watching, and nothing at all when the output is a
 * file or a CI log -- where a carriage return does not erase anything and the counters would
 * pile up into one unreadable line.
 */
static void report_progress(const char *what, size_t done, size_t total) {
    static int to_a_terminal = -1;
    if (to_a_terminal < 0) to_a_terminal = isatty(fileno(stdout));
    if (!to_a_terminal) return;

    printf("\r  %-10s %zu / %zu", what, done, total);
    fflush(stdout);
}

static double seconds_now(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

static int parse_options(int argc, char **argv, options *out, int *should_exit) {
    *should_exit = 0;

    for (int argument = 1; argument < argc; ++argument) {
        const char *name = argv[argument];
        const char *value = argument + 1 < argc ? argv[argument + 1] : NULL;

        if (strcmp(name, "--help") == 0 || strcmp(name, "-h") == 0) {
            print_usage(argv[0]);
            *should_exit = 1;
            return 1;
        }

        if (value == NULL) {
            fprintf(stderr, "%s needs a value.\n", name);
            return 0;
        }

        if (strcmp(name, "--data") == 0) out->data_directory = value;
        else if (strcmp(name, "--train") == 0) out->train_images = strtoul(value, NULL, 10);
        else if (strcmp(name, "--test") == 0) out->test_images = strtoul(value, NULL, 10);
        else if (strcmp(name, "--lanes") == 0) out->lanes = (uint32_t)strtoul(value, NULL, 10);
        else if (strcmp(name, "--seed") == 0) out->seed = strtoull(value, NULL, 10);
        else {
            fprintf(stderr, "Unknown option '%s'. Try --help.\n", name);
            return 0;
        }

        ++argument;
    }

    if (out->lanes == 0 || out->lanes > HDC_MAX_STORABLE_LANES) {
        fprintf(stderr, "--lanes must be between 1 and %d: a wider code cannot be stored on the device.\n",
                HDC_MAX_STORABLE_LANES);
        return 0;
    }

    return 1;
}

int main(int argc, char **argv) {
    options chosen = {
        .data_directory = NULL,
        .train_images = (size_t)-1,
        .test_images = (size_t)-1,
        .lanes = HDC_MAX_STORABLE_LANES,
        .seed = 1,
    };

    int should_exit = 0;
    if (!parse_options(argc, argv, &chosen, &should_exit)) return EXIT_FAILED;
    if (should_exit) return 0;

    const char *directory = chosen.data_directory;
    if (directory == NULL) directory = getenv("SPARSR_MNIST_DIR");
    if (directory == NULL || directory[0] == '\0') directory = "data";

    dataset training = {0};
    dataset test = {0};
    if (!load_dataset(directory, 1, chosen.train_images, &training) ||
        !load_dataset(directory, 0, chosen.test_images, &test)) {
        free_dataset(&training);
        explain_missing_data(directory);
        return EXIT_NO_DATA;
    }

    /*
     * One vote per class, over every training image of that class. The counters the device
     * builds are 13 bit-planes wide, so a class may hold up to HDC_MAJORITY_MAX_MEMBERS
     * examples. MNIST's largest digit has 6,742, comfortably inside that.
     */
    if (training.count > DIGIT_CLASSES * (size_t)HDC_MAJORITY_MAX_MEMBERS) {
        fprintf(stderr, "More training images than the device's vote counters can hold.\n");
        free_dataset(&training);
        free_dataset(&test);
        return EXIT_FAILED;
    }

    printf("Sparsr HDC digit recognition\n");
    printf("  data       %s\n", directory);
    printf("  training   %zu images\n", training.count);
    printf("  test       %zu images\n", test.count);
    printf("  code       %u lanes, %u of the %d bit positions\n\n",
           chosen.lanes, chosen.lanes * 32u, HDC_HYPERVECTOR_BITS);

    hdc_status status = hdc_init();
    if (status != HDC_OK) {
        fprintf(stderr, "hdc_init failed: %s\n", hdc_status_string(status));
        fprintf(stderr, "This library's kernels are RV32I, so it needs SPARSR_BACKEND=vm.\n");
        free_dataset(&training);
        free_dataset(&test);
        return EXIT_FAILED;
    }

    int exit_code = EXIT_FAILED;

    /* Scratch big enough for either kind of vote: the lit pixels of one image, or every
     * training image of one class. */
    size_t widest_vote = training.count > IMAGE_PIXELS ? training.count : IMAGE_PIXELS;
    const hdc_hypervector **members = malloc(widest_vote * sizeof *members);
    hdc_hypervector *encoded = malloc(training.count * sizeof *encoded);
    hdc_hypervector *items = build_item_memory(chosen.lanes, chosen.seed);

    if (members == NULL || encoded == NULL || items == NULL) {
        fprintf(stderr, "Out of memory.\n");
        goto done;
    }

    /* ---- step 2: encode the training set ---- */

    double started = seconds_now();
    for (size_t image = 0; image < training.count; ++image) {
        status = encode_image(training.pixels + image * IMAGE_PIXELS, items, members, &encoded[image]);
        if (status != HDC_OK) {
            fprintf(stderr, "\nEncoding training image %zu failed: %s\n", image, hdc_status_string(status));
            goto done;
        }
        if ((image + 1) % 1000 == 0 || image + 1 == training.count)
            report_progress("encoding", image + 1, training.count);
    }
    double encode_seconds = seconds_now() - started;
    printf("\r  encoding   %zu images   (%.1f s)\n", training.count, encode_seconds);

    /* ---- step 3: one prototype per digit ---- */

    started = seconds_now();
    hdc_hypervector prototypes[DIGIT_CLASSES];
    size_t class_members[DIGIT_CLASSES];

    for (int digit = 0; digit < DIGIT_CLASSES; ++digit) {
        size_t count = 0;
        for (size_t image = 0; image < training.count; ++image)
            if (training.labels[image] == digit) members[count++] = &encoded[image];

        class_members[digit] = count;

        if (count == 0) {
            /* A class with no examples has an empty prototype, and hdc_similarity scores it
             * zero against everything, so it simply never wins. */
            hdc_zero(&prototypes[digit]);
            continue;
        }

        status = hdc_bundle_majority(members, count, &prototypes[digit]);
        if (status != HDC_OK) {
            fprintf(stderr, "Building the prototype for digit %d failed: %s\n", digit, hdc_status_string(status));
            goto done;
        }
    }
    double train_seconds = seconds_now() - started;
    printf("  training   10 prototypes   (%.1f s)\n", train_seconds);

    /* ---- step 4: classify ---- */

    started = seconds_now();
    size_t correct = 0;
    size_t correct_per_digit[DIGIT_CLASSES] = {0};
    size_t tested_per_digit[DIGIT_CLASSES] = {0};

    for (size_t image = 0; image < test.count; ++image) {
        hdc_hypervector query;
        status = encode_image(test.pixels + image * IMAGE_PIXELS, items, members, &query);
        if (status != HDC_OK) {
            fprintf(stderr, "\nEncoding test image %zu failed: %s\n", image, hdc_status_string(status));
            goto done;
        }

        int best = -1;
        double best_score = -1.0;
        for (int digit = 0; digit < DIGIT_CLASSES; ++digit) {
            hdc_similarity_result overlap;
            status = hdc_similarity(&query, &prototypes[digit], &overlap);
            if (status != HDC_OK) {
                fprintf(stderr, "\nComparing against digit %d failed: %s\n", digit, hdc_status_string(status));
                goto done;
            }

            double score = hdc_cosine(&overlap);
            if (score <= best_score) continue;
            best_score = score;
            best = digit;
        }

        int truth = test.labels[image];
        ++tested_per_digit[truth];
        if (best == truth) { ++correct; ++correct_per_digit[truth]; }

        if ((image + 1) % 200 == 0 || image + 1 == test.count)
            report_progress("testing", image + 1, test.count);
    }
    double test_seconds = seconds_now() - started;
    printf("\r  testing    %zu images   (%.1f s)\n\n", test.count, test_seconds);

    /* ---- the answer ---- */

    printf("Accuracy: %.2f%%  (%zu of %zu correct)\n\n", test.count == 0 ? 0.0 : 100.0 * (double)correct / (double)test.count,
           correct, test.count);

    printf("  digit   trained on   tested   correct   accuracy   prototype bits   lanes\n");
    for (int digit = 0; digit < DIGIT_CLASSES; ++digit) {
        double digit_accuracy = tested_per_digit[digit] == 0
            ? 0.0
            : 100.0 * (double)correct_per_digit[digit] / (double)tested_per_digit[digit];
        printf("  %5d   %10zu   %6zu   %7zu   %8.1f%%   %14u   %5u\n",
               digit, class_members[digit], tested_per_digit[digit], correct_per_digit[digit],
               digit_accuracy, hdc_weight(&prototypes[digit]), hdc_nonzero_lanes(&prototypes[digit]));
    }

    /* Every prototype sits at exactly `lanes` occupied lanes, which is why nothing here ever
     * came back as HDC_ERROR_TOO_DENSE. Printing it is the cheapest way to show that the
     * storability rule was respected rather than got lucky. */
    printf("\n  Every prototype occupies %u of the %d lanes a row can hold, so all of them\n",
           chosen.lanes, HDC_MAX_STORABLE_LANES);
    printf("  live on the device. Widening the code past %d lanes would not fit.\n", HDC_MAX_STORABLE_LANES);

    printf("\n  Time: %.1f s encoding, %.1f s training, %.1f s testing.\n",
           encode_seconds, train_seconds, test_seconds);

    exit_code = 0;

done:
    free(items);
    free(encoded);
    free(members);
    hdc_shutdown();
    free_dataset(&training);
    free_dataset(&test);
    return exit_code;
}
