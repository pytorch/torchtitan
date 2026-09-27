// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#include <stdexcept>

namespace py = pybind11;
using namespace std;

void build_blending_indices(py::array_t<int16_t> &dataset_index,
                            py::array_t<int64_t> &dataset_sample_index,
                            const py::array_t<double> &weights,
                            const int32_t num_datasets,
                            const int64_t size) {
    /* Given multiple datasets and a weighting array, build samples
     such that it follows those weights.*/

    // Get the pointer access without the checks.
    auto dataset_index_ptr = dataset_index.mutable_unchecked<1>();
    auto dataset_sample_index_ptr = dataset_sample_index.mutable_unchecked<1>();
    auto weights_ptr = weights.unchecked<1>();

    // Initialize buffer for number of samples used for each dataset.
    int64_t current_samples[num_datasets];
    for (int64_t i = 0; i < num_datasets; ++i) {
        current_samples[i] = 0;
    }

    // For each sample:
    for (int64_t sample_idx = 0; sample_idx < size; ++sample_idx) {
        // Determine where the max error in sampling is happening.
        auto sample_idx_double = std::max(static_cast<double>(sample_idx), 1.0);
        int64_t max_error_index = 0;
        double max_error = weights_ptr[0] * sample_idx_double - static_cast<double>(current_samples[0]);
        for (int64_t dataset_idx = 1; dataset_idx < num_datasets; ++dataset_idx) {
            double error =
                weights_ptr[dataset_idx] * sample_idx_double - static_cast<double>(current_samples[dataset_idx]);
            if (error > max_error) {
                max_error = error;
                max_error_index = dataset_idx;
            }
        }

        // Populate the indices.
        dataset_index_ptr[sample_idx] = static_cast<int16_t>(max_error_index);
        dataset_sample_index_ptr[sample_idx] = current_samples[max_error_index];

        // Update the total samples.
        current_samples[max_error_index] += 1;
    }
}

template <typename T>
py::array build_sample_idx(const py::array_t<int32_t> &sizes_,
                           const py::array_t<T> &doc_idx_,
                           const int32_t seq_length,
                           const int32_t num_epochs,
                           const int64_t tokens_per_epoch) {
    /* Sample index (sample_idx) is used for gpt2 like dataset for which
       the documents are flattened and the samples are built based on this
       1-D flatten array. It is a 2D array with sizes [number-of-samples + 1, 2]
       where [..., 0] contains the index into `doc_idx` and [..., 1] is the
       starting offset in that document.*/

    // Consistency checks.
    assert(seq_length > 1);
    assert(num_epochs > 0);
    assert(tokens_per_epoch > 1);

    // Remove bound checks.
    auto sizes = sizes_.template unchecked<1>();
    auto doc_idx = doc_idx_.template unchecked<1>();

    // Mapping and it's length (1D).
    int64_t num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length;
    T *sample_idx = new T[2 * (num_samples + 1)];

    // Index into sample_idx.
    int64_t sample_index = 0;
    // Index into doc_idx.
    int64_t doc_idx_index = 0;
    // Beginning offset for each document.
    T doc_offset = 0;
    // Start with first document and no offset.
    sample_idx[2 * sample_index] = doc_idx_index;
    sample_idx[2 * sample_index + 1] = doc_offset;
    ++sample_index;

    while (sample_index <= num_samples) {
        // Start with a fresh sequence.
        T remaining_seq_length = seq_length + 1;
        while (remaining_seq_length != 0) {
            // Get the document length.
            auto doc_id = doc_idx[doc_idx_index];
            auto doc_length = static_cast<T>(sizes[doc_id]) - doc_offset;
            // And add it to the current sequence.
            remaining_seq_length -= doc_length;
            // If we have more than a full sequence, adjust offset and set
            // remaining length to zero so we return from the while loop.
            // Note that -1 here is for the same reason we have -1 in
            // `_num_epochs` calculations.
            if (remaining_seq_length <= 0) {
                doc_offset += (remaining_seq_length + doc_length - 1);
                remaining_seq_length = 0;
            } else {
                // Otherwise, start from the beginning of the next document.
                ++doc_idx_index;
                doc_offset = 0;
            }
        }
        // Record the sequence.
        sample_idx[2 * sample_index] = doc_idx_index;
        sample_idx[2 * sample_index + 1] = doc_offset;
        ++sample_index;
    }

    // Method to deallocate memory.
    py::capsule free_when_done(sample_idx, [](void *mem_) {
        int32_t *mem = reinterpret_cast<int32_t *>(mem_);
        delete[] mem;
    });

    // Return the numpy array.
    const auto byte_size = sizeof(T);
    return py::array(std::vector<int64_t>{num_samples + 1, 2}, // shape
                     {2 * byte_size, byte_size},               // C-style contiguous strides
                     sample_idx,                               // the data pointer
                     free_when_done);                          // numpy array references
}

PYBIND11_MODULE(helpers, m) {
    m.def("build_sample_idx_int32", &build_sample_idx<int32_t>);
    m.def("build_sample_idx_int64", &build_sample_idx<int64_t>);
    m.def("build_blending_indices", &build_blending_indices);
}
