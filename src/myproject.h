
#ifndef MYPROJECT_H_
#define MYPROJECT_H_
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

#include "defines.h"
void myproject(
    input_t node_attr[N_NODE_GROUP][N_NODE_LAYER*NODE_DIM], input3_t edge_attr[N_EDGE_GROUP][N_EDGE_LAYER*EDGE_DIM], input4_t edge_index[N_EDGE_GROUP][N_EDGE_LAYER*TWO],
    layer11_t layer11_out[N_EDGE_GROUP][N_EDGE_LAYER*LAYER11_OUT_DIM]
);

#endif