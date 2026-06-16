// Standalone reproducer for a ptxas (-O>=1) miscompilation. See README.md.
//
// `ranking_forward` loops `candidate_position` over `0..candidate_count` per
// thread and stores the best-scoring index in `best_position`, so the result
// is always in `[0, candidate_count)`. ptxas -O>=1 promotes the counter into
// the warp-uniform datapath; the body diverges per thread, so one thread
// reads the counter incoherently and stores `candidate_count` (out of range).
// -O0 is correct.
//
// Hand transcription of a CubeCL-generated kernel with a plain pointer ABI
// and literal shapes, so it has no CubeCL/burn/Rust/cudarc dependency.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MAX_PEAKS 128
#define CANDIDATE_CAPACITY (MAX_PEAKS * 2)        // 256
#define DP_CAPACITY (MAX_PEAKS * 2 + 1)           // 257
#define INVALID 0xFFFFFFFFu

// ---------------------------------------------------------------------------
// Device helpers (transcribed from the CubeCL #[cube] helpers).
// ---------------------------------------------------------------------------

__device__ void insert_modified_neighbor(unsigned *neighbor_a, unsigned *neighbor_b,
                                          unsigned edge, unsigned neighbor) {
    unsigned ei = edge;
    if (neighbor_a[ei] == INVALID) {
        neighbor_a[ei] = neighbor;
    } else if (neighbor_a[ei] != neighbor) {
        neighbor_b[ei] = neighbor;
    }
}

__device__ unsigned first_modified_neighbor_not_from(unsigned neighbor_a, unsigned neighbor_b,
                                                     unsigned from) {
    unsigned next = INVALID;
    if (neighbor_a != INVALID && neighbor_a != from) {
        next = neighbor_a;
    } else if (neighbor_b != INVALID && neighbor_b != from) {
        next = neighbor_b;
    }
    return next;
}

__device__ unsigned collect_modified_candidates(
    const float *mz, const float *left_products, unsigned left_row, unsigned left_peaks,
    float left_precursor_value, const float *right_products, unsigned right_row,
    unsigned right_peaks, float right_precursor_value, float tolerance,
    unsigned *candidate_left, unsigned *candidate_right) {
    float zero = 0.0f;
    unsigned candidate_count = 0u;

    unsigned right_cursor = 0u;
    for (unsigned left_peak = 0; left_peak < left_peaks; ++left_peak) {
        if (left_products[left_peak] > zero) {
            float left_value = mz[left_row * MAX_PEAKS + left_peak];
            while (right_cursor < right_peaks && right_products[right_cursor] <= zero) {
                right_cursor += 1u;
            }
            while (right_cursor < right_peaks) {
                if (right_products[right_cursor] <= zero) {
                    right_cursor += 1u;
                } else {
                    float right_value = mz[right_row * MAX_PEAKS + right_cursor];
                    float delta = left_value - right_value;
                    if (delta > tolerance) {
                        right_cursor += 1u;
                    } else if (fabsf(delta) <= tolerance) {
                        if (candidate_count < CANDIDATE_CAPACITY) {
                            candidate_left[candidate_count] = left_peak;
                            candidate_right[candidate_count] = right_cursor;
                            candidate_count += 1u;
                        }
                        right_cursor += 1u;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    if (right_precursor_value < left_precursor_value - tolerance ||
        right_precursor_value > left_precursor_value + tolerance) {
        unsigned shifted_right_cursor = 0u;
        for (unsigned left_peak = 0; left_peak < left_peaks; ++left_peak) {
            if (left_products[left_peak] > zero) {
                float left_value = mz[left_row * MAX_PEAKS + left_peak] - left_precursor_value;
                while (shifted_right_cursor < right_peaks &&
                       right_products[shifted_right_cursor] <= zero) {
                    shifted_right_cursor += 1u;
                }
                while (shifted_right_cursor < right_peaks) {
                    if (right_products[shifted_right_cursor] <= zero) {
                        shifted_right_cursor += 1u;
                    } else {
                        float right_value =
                            mz[right_row * MAX_PEAKS + shifted_right_cursor] - right_precursor_value;
                        float delta = left_value - right_value;
                        if (delta > tolerance) {
                            shifted_right_cursor += 1u;
                        } else if (fabsf(delta) <= tolerance) {
                            if (candidate_count < CANDIDATE_CAPACITY) {
                                candidate_left[candidate_count] = left_peak;
                                candidate_right[candidate_count] = shifted_right_cursor;
                                candidate_count += 1u;
                            }
                            shifted_right_cursor += 1u;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }

    return candidate_count;
}

__device__ unsigned sort_and_dedupe_modified_candidates(unsigned *candidate_left,
                                                        unsigned *candidate_right,
                                                        unsigned candidate_count) {
    unsigned count = candidate_count;
    unsigned unique_count = 0u;
    if (count > 0u) {
        for (unsigned sort_index = 1u; sort_index < count; ++sort_index) {
            unsigned key_left = candidate_left[sort_index];
            unsigned key_right = candidate_right[sort_index];
            unsigned insert_index = sort_index;
            while (insert_index > 0u) {
                unsigned previous = insert_index - 1u;
                unsigned previous_left = candidate_left[previous];
                unsigned previous_right = candidate_right[previous];
                if (previous_left > key_left ||
                    (previous_left == key_left && previous_right > key_right)) {
                    candidate_left[insert_index] = previous_left;
                    candidate_right[insert_index] = previous_right;
                    insert_index -= 1u;
                } else {
                    break;
                }
            }
            candidate_left[insert_index] = key_left;
            candidate_right[insert_index] = key_right;
        }

        for (unsigned read_index = 0u; read_index < count; ++read_index) {
            unsigned current_left = candidate_left[read_index];
            unsigned current_right = candidate_right[read_index];
            if (read_index == 0u || current_left != candidate_left[read_index - 1u] ||
                current_right != candidate_right[read_index - 1u]) {
                candidate_left[unique_count] = current_left;
                candidate_right[unique_count] = current_right;
                unique_count += 1u;
            }
        }
    }
    return unique_count;
}

__device__ void build_modified_conflict_graph(
    const unsigned *candidate_left, const unsigned *candidate_right, unsigned candidate_count,
    unsigned *left_slot_a, unsigned *left_slot_b, unsigned *right_slot_a, unsigned *right_slot_b,
    unsigned *neighbor_a, unsigned *neighbor_b, unsigned *visited) {
    unsigned count = candidate_count;

    for (unsigned peak = 0; peak < MAX_PEAKS; ++peak) {
        left_slot_a[peak] = INVALID;
        left_slot_b[peak] = INVALID;
        right_slot_a[peak] = INVALID;
        right_slot_b[peak] = INVALID;
    }

    for (unsigned edge = 0; edge < count; ++edge) {
        neighbor_a[edge] = INVALID;
        neighbor_b[edge] = INVALID;
        visited[edge] = 0u;

        unsigned left_peak = candidate_left[edge];
        if (left_slot_a[left_peak] == INVALID) {
            left_slot_a[left_peak] = edge;
        } else {
            left_slot_b[left_peak] = edge;
        }

        unsigned right_peak = candidate_right[edge];
        if (right_slot_a[right_peak] == INVALID) {
            right_slot_a[right_peak] = edge;
        } else {
            right_slot_b[right_peak] = edge;
        }
    }

    for (unsigned peak = 0; peak < MAX_PEAKS; ++peak) {
        unsigned left_a = left_slot_a[peak];
        unsigned left_b = left_slot_b[peak];
        if (left_a != INVALID && left_b != INVALID) {
            insert_modified_neighbor(neighbor_a, neighbor_b, left_a, left_b);
            insert_modified_neighbor(neighbor_a, neighbor_b, left_b, left_a);
        }

        unsigned right_a = right_slot_a[peak];
        unsigned right_b = right_slot_b[peak];
        if (right_a != INVALID && right_b != INVALID) {
            insert_modified_neighbor(neighbor_a, neighbor_b, right_a, right_b);
            insert_modified_neighbor(neighbor_a, neighbor_b, right_b, right_a);
        }
    }
}

__device__ float modified_linear_cosine_score_rows(
    const float *mz, const float *intensity, const float *precursor, unsigned left_row,
    unsigned right_row, float mz_p, float intensity_p, float tolerance, float eps,
    unsigned peaks) {
    unsigned left_peaks = peaks;
    unsigned right_peaks = peaks;
    float zero = 0.0f;
    float one = 1.0f;

    float similarity = zero;

    if (!(left_peaks <= MAX_PEAKS && right_peaks <= MAX_PEAKS)) {
        return similarity;
    }

    float left_products[MAX_PEAKS];
    float right_products[MAX_PEAKS];

    float left_mz_max = zero;
    float left_intensity_max = zero;
    for (unsigned peak = 0; peak < left_peaks; ++peak) {
        left_products[peak] = zero;
        float i = intensity[left_row * MAX_PEAKS + peak];
        if (i > zero) {
            float m = mz[left_row * MAX_PEAKS + peak];
            left_mz_max = fmaxf(left_mz_max, powf(m, mz_p));
            left_intensity_max = fmaxf(left_intensity_max, powf(i, intensity_p));
        }
    }

    float left_product_max = zero;
    for (unsigned peak = 0; peak < left_peaks; ++peak) {
        float i = intensity[left_row * MAX_PEAKS + peak];
        if (i > zero) {
            float m = mz[left_row * MAX_PEAKS + peak];
            float mz_component = powf(m, mz_p);
            if (left_mz_max > zero) {
                mz_component /= left_mz_max;
            }
            float intensity_component = powf(i, intensity_p);
            if (left_intensity_max > zero) {
                intensity_component /= left_intensity_max;
            }
            float product = mz_component * intensity_component;
            left_products[peak] = product;
            left_product_max = fmaxf(left_product_max, product);
        }
    }

    float left_norm_square = zero;
    for (unsigned peak = 0; peak < left_peaks; ++peak) {
        if (left_product_max > zero) {
            left_products[peak] /= left_product_max;
        }
        left_norm_square += left_products[peak] * left_products[peak];
    }

    float right_mz_max = zero;
    float right_intensity_max = zero;
    for (unsigned peak = 0; peak < right_peaks; ++peak) {
        right_products[peak] = zero;
        float i = intensity[right_row * MAX_PEAKS + peak];
        if (i > zero) {
            float m = mz[right_row * MAX_PEAKS + peak];
            right_mz_max = fmaxf(right_mz_max, powf(m, mz_p));
            right_intensity_max = fmaxf(right_intensity_max, powf(i, intensity_p));
        }
    }

    float right_product_max = zero;
    for (unsigned peak = 0; peak < right_peaks; ++peak) {
        float i = intensity[right_row * MAX_PEAKS + peak];
        if (i > zero) {
            float m = mz[right_row * MAX_PEAKS + peak];
            float mz_component = powf(m, mz_p);
            if (right_mz_max > zero) {
                mz_component /= right_mz_max;
            }
            float intensity_component = powf(i, intensity_p);
            if (right_intensity_max > zero) {
                intensity_component /= right_intensity_max;
            }
            float product = mz_component * intensity_component;
            right_products[peak] = product;
            right_product_max = fmaxf(right_product_max, product);
        }
    }

    float right_norm_square = zero;
    for (unsigned peak = 0; peak < right_peaks; ++peak) {
        if (right_product_max > zero) {
            right_products[peak] /= right_product_max;
        }
        right_norm_square += right_products[peak] * right_products[peak];
    }

    if (left_norm_square > zero && right_norm_square > zero) {
        unsigned candidate_left[CANDIDATE_CAPACITY];
        unsigned candidate_right[CANDIDATE_CAPACITY];
        float left_precursor_value = precursor[left_row];
        float right_precursor_value = precursor[right_row];

        unsigned candidate_count = collect_modified_candidates(
            mz, left_products, left_row, left_peaks, left_precursor_value, right_products,
            right_row, right_peaks, right_precursor_value, tolerance, candidate_left,
            candidate_right);

        if (candidate_count > 0u) {
            candidate_count = sort_and_dedupe_modified_candidates(candidate_left, candidate_right,
                                                                  candidate_count);

            unsigned left_slot_a[MAX_PEAKS];
            unsigned left_slot_b[MAX_PEAKS];
            unsigned right_slot_a[MAX_PEAKS];
            unsigned right_slot_b[MAX_PEAKS];
            unsigned neighbor_a[CANDIDATE_CAPACITY];
            unsigned neighbor_b[CANDIDATE_CAPACITY];
            unsigned visited[CANDIDATE_CAPACITY];

            build_modified_conflict_graph(candidate_left, candidate_right, candidate_count,
                                          left_slot_a, left_slot_b, right_slot_a, right_slot_b,
                                          neighbor_a, neighbor_b, visited);

            unsigned path[CANDIDATE_CAPACITY];
            float benefits[CANDIDATE_CAPACITY];
            float dp[DP_CAPACITY];
            float score = zero;

            for (unsigned start = 0; start < candidate_count; ++start) {
                if (visited[start] == 0u) {
                    unsigned end = start;
                    unsigned from = INVALID;
                    while (true) {
                        unsigned next =
                            first_modified_neighbor_not_from(neighbor_a[end], neighbor_b[end], from);
                        if (next == INVALID) {
                            break;
                        }
                        from = end;
                        end = next;
                    }

                    unsigned path_len = 0u;
                    unsigned current = end;
                    unsigned previous = INVALID;
                    while (true) {
                        visited[current] = 1u;
                        path[path_len] = current;
                        path_len += 1u;

                        unsigned next = first_modified_neighbor_not_from(neighbor_a[current],
                                                                         neighbor_b[current],
                                                                         previous);
                        if (next == INVALID) {
                            break;
                        }
                        previous = current;
                        current = next;
                    }

                    if (path_len == 1u) {
                        unsigned edge = path[0];
                        unsigned lp = candidate_left[edge];
                        unsigned rp = candidate_right[edge];
                        score += left_products[lp] * right_products[rp];
                    } else {
                        for (unsigned path_index = 0; path_index < path_len; ++path_index) {
                            unsigned edge = path[path_index];
                            unsigned lp = candidate_left[edge];
                            unsigned rp = candidate_right[edge];
                            benefits[path_index] = fmaxf(left_products[lp] * right_products[rp], eps);
                        }

                        dp[0] = zero;
                        dp[1] = benefits[0];
                        for (unsigned index = 2; index < path_len + 1u; ++index) {
                            float take = dp[index - 2u] + benefits[index - 1u];
                            float skip = dp[index - 1u];
                            dp[index] = take >= skip ? take : skip;
                        }

                        unsigned index = path_len;
                        while (index > 0u) {
                            if (index == 1u) {
                                unsigned edge = path[0];
                                unsigned lp = candidate_left[edge];
                                unsigned rp = candidate_right[edge];
                                score += left_products[lp] * right_products[rp];
                                break;
                            }
                            float take = dp[index - 2u] + benefits[index - 1u];
                            if (take >= dp[index - 1u]) {
                                unsigned edge = path[index - 1u];
                                unsigned lp = candidate_left[edge];
                                unsigned rp = candidate_right[edge];
                                score += left_products[lp] * right_products[rp];
                                index -= 2u;
                            } else {
                                index -= 1u;
                            }
                        }
                    }
                }
            }

            similarity = fminf(fmaxf(score / (sqrtf(left_norm_square) * sqrtf(right_norm_square) + eps), zero), one);
        }
    }
    return similarity;
}

// ---------------------------------------------------------------------------
// ranking_forward kernel (one thread per anchor).
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(256) ranking_forward(
    const float *__restrict__ mz, const float *__restrict__ intensity,
    const float *__restrict__ precursor, int *__restrict__ best_position_out, unsigned batch_start,
    unsigned batch_items, unsigned candidates_per_anchor, float mz_power, float intensity_power,
    float mz_tolerance, unsigned seed, float epsilon, unsigned peaks) {
    unsigned anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= batch_items || batch_items < 3u) {
        return;
    }

    unsigned candidate_count = min(max(candidates_per_anchor, 2u), batch_items - 1u);
    unsigned teacher_anchor = anchor;
    float mz_p = mz_power;
    float intensity_p = intensity_power;
    float tolerance = mz_tolerance;
    float eps = epsilon;

    unsigned state = seed ^ ((anchor + 1u) * 40503u) ^ (batch_start >> 16);
    if (state == 0u) {
        state = 0x6d2b79f5u;
    }
    unsigned partner_slots = batch_items - 1u;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    unsigned offset = state % partner_slots;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    unsigned stride = (state % partner_slots) + 1u;
    bool coprime = false;
    while (!coprime) {
        unsigned left = stride;
        unsigned right = partner_slots;
        while (right != 0u) {
            unsigned remainder = left % right;
            left = right;
            right = remainder;
        }
        coprime = left == 1u;
        if (!coprime) {
            stride += 1u;
            if (stride > partner_slots) {
                stride = 1u;
            }
        }
    }

    float best_score = -1.0f;
    unsigned best_position = 0u;

    for (unsigned candidate_position = 0; candidate_position < candidate_count;
         ++candidate_position) {
        unsigned local_partner = (offset + candidate_position * stride) % partner_slots;
        if (local_partner >= anchor) {
            local_partner += 1u;
        }
        unsigned partner_row = local_partner;
        float score = modified_linear_cosine_score_rows(mz, intensity, precursor, teacher_anchor,
                                                         partner_row, mz_p, intensity_p, tolerance,
                                                         eps, peaks);
        if (score > best_score) {
            best_score = score;
            best_position = candidate_position;
        }
    }

    best_position_out[anchor] = (int)best_position;
}

// ---------------------------------------------------------------------------
// Host harness: load the fixture, launch, check anchor 13673.
// ---------------------------------------------------------------------------

#define TEACHER_PEAKS 128
#define BATCH_ITEMS 32768
#define BATCH_START 65536u
#define CANDIDATES_PER_ANCHOR 4u
#define MZ_POWER 0.0f
#define INTENSITY_POWER 0.5f
#define MZ_TOLERANCE 0.02f
#define EPSILON 1.0e-8f
#define SEED_U64 3642086346266552404ULL
#define OFFENDER 13673

// Reconstruct the full [BATCH_ITEMS, TEACHER_PEAKS] arrays from the compact
// fixture `spectra.bin`. All rows start at zero (empty spectra); only the 159
// rows that the offender's warp actually reads carry real data. Layout:
//   u32 num_rows, u32 num_peaks, u32 batch_items,
//   then num_rows records of: u32 row_index, f32 mz[num_peaks],
//                             f32 intensity[num_peaks], f32 precursor.
static void load_spectra(const char *path, float *mz, float *intensity, float *precursor) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    unsigned header[3];
    if (fread(header, sizeof(unsigned), 3, f) != 3) {
        fprintf(stderr, "bad header in %s\n", path);
        exit(2);
    }
    unsigned num_rows = header[0], num_peaks = header[1], batch_items = header[2];
    if (num_peaks != TEACHER_PEAKS || batch_items != BATCH_ITEMS) {
        fprintf(stderr, "fixture shape mismatch: rows-peaks=%u batch=%u\n", num_peaks, batch_items);
        exit(2);
    }
    for (unsigned k = 0; k < num_rows; ++k) {
        unsigned row;
        if (fread(&row, sizeof(unsigned), 1, f) != 1 ||
            fread(&mz[(size_t)row * TEACHER_PEAKS], sizeof(float), TEACHER_PEAKS, f) != TEACHER_PEAKS ||
            fread(&intensity[(size_t)row * TEACHER_PEAKS], sizeof(float), TEACHER_PEAKS, f) != TEACHER_PEAKS ||
            fread(&precursor[row], sizeof(float), 1, f) != 1) {
            fprintf(stderr, "short read on record %u in %s\n", k, path);
            exit(2);
        }
    }
    fclose(f);
}

static void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        exit(3);
    }
}

int main(int argc, char **argv) {
    const char *fixture = (argc > 1) ? argv[1] : "spectra.bin";

    size_t mat = (size_t)BATCH_ITEMS * TEACHER_PEAKS;
    float *mz = (float *)calloc(mat, sizeof(float));
    float *intensity = (float *)calloc(mat, sizeof(float));
    float *precursor = (float *)calloc(BATCH_ITEMS, sizeof(float));
    load_spectra(fixture, mz, intensity, precursor);

    float *d_mz, *d_intensity, *d_precursor;
    int *d_best;
    check(cudaMalloc(&d_mz, mat * sizeof(float)), "malloc mz");
    check(cudaMalloc(&d_intensity, mat * sizeof(float)), "malloc intensity");
    check(cudaMalloc(&d_precursor, BATCH_ITEMS * sizeof(float)), "malloc precursor");
    check(cudaMalloc(&d_best, BATCH_ITEMS * sizeof(int)), "malloc best");

    check(cudaMemcpy(d_mz, mz, mat * sizeof(float), cudaMemcpyHostToDevice), "copy mz");
    check(cudaMemcpy(d_intensity, intensity, mat * sizeof(float), cudaMemcpyHostToDevice),
          "copy intensity");
    check(cudaMemcpy(d_precursor, precursor, BATCH_ITEMS * sizeof(float), cudaMemcpyHostToDevice),
          "copy precursor");

    dim3 block(256);
    dim3 grid((BATCH_ITEMS + 255) / 256);
    ranking_forward<<<grid, block>>>(d_mz, d_intensity, d_precursor, d_best, BATCH_START,
                                     BATCH_ITEMS, CANDIDATES_PER_ANCHOR, MZ_POWER, INTENSITY_POWER,
                                     MZ_TOLERANCE, (unsigned)SEED_U64, EPSILON, TEACHER_PEAKS);
    check(cudaGetLastError(), "launch");
    check(cudaDeviceSynchronize(), "sync");

    int *best = (int *)malloc(BATCH_ITEMS * sizeof(int));
    check(cudaMemcpy(best, d_best, BATCH_ITEMS * sizeof(int), cudaMemcpyDeviceToHost), "copy back");

    int oob = 0;
    int first_oob[8];
    int n_first = 0;
    for (int i = 0; i < BATCH_ITEMS; ++i) {
        if (best[i] < 0 || best[i] >= (int)CANDIDATES_PER_ANCHOR) {
            if (n_first < 8) first_oob[n_first++] = i;
            ++oob;
        }
    }

    printf("offender anchor %d: best_position = %d (valid range [0,%u))\n", OFFENDER, best[OFFENDER],
           CANDIDATES_PER_ANCHOR);
    printf("total out-of-range best_position values: %d\n", oob);
    for (int i = 0; i < n_first; ++i) {
        printf("  out-of-range at anchor %d: best_position = %d\n", first_oob[i], best[first_oob[i]]);
    }

    if (oob == 0) {
        printf("PASS: all best_position values in range. ptxas miscompile NOT present.\n");
        return 0;
    } else {
        printf("FAIL: %d out-of-range best_position values. ptxas miscompile present.\n", oob);
        return 1;
    }
}
