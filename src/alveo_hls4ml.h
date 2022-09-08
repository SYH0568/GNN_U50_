
#ifndef ALVEO_HLS4ML_H_
#define ALVEO_HLS4ML_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void alveo_hls4ml(
	const input_t *node_attr_in, // Read-Only Vector
	const input3_t *edge_attr_in, // Read-Only Vector
	const input4_t *edge_index_in, // Read-Only Vector

	layer11_t *out       // Output Result
	);
#endif