/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is a wrapper to be used with an hls4ml project to enable proper handling by SDAccel
*******************************************************************************/
#include <iostream>
#include "myproject.h"


extern "C" {
void read_node_attr(const NODE_GROUP *node_attr_in,input_t node_attr_in_bigbuf[N_NODE_GROUP][N_NODE_LAYER*NODE_DIM],int g){
	for (int i = 0; i < N_NODE_LAYER*NODE_DIM; i++) {
		#pragma HLS PIPELINE
		for(int j=0;j<N_NODE_GROUP;j++){
			#pragma HLS UNROLL
			node_attr_in_bigbuf[j][i] = node_attr_in[g*N_NODE_LAYER*NODE_DIM+i].layer[j];
		}
	}
}
void read_edge_attr(const EDGE_GROUP *edge_attr_in,input3_t edge_attr_in_bigbuf[N_EDGE_GROUP][N_EDGE_LAYER*EDGE_DIM],int g){
	
	for (int i = 0; i < N_EDGE_LAYER*EDGE_DIM; i++) {
		#pragma HLS PIPELINE
		for(int j=0;j<N_EDGE_GROUP;j++){
			#pragma HLS UNROLL
			edge_attr_in_bigbuf[j][i] = edge_attr_in[g* N_EDGE_LAYER*EDGE_DIM+i].layer[j];
		}
	}
}
void read_edge_index(const INDEX_GROUP *edge_index_in,input4_t edge_index_in_bigbuf[N_EDGE_GROUP][N_EDGE_LAYER*TWO],int g){
	for (int i = 0; i < N_EDGE_LAYER*TWO; i++) {
		#pragma HLS PIPELINE
		for(int j=0;j<N_EDGE_GROUP;j++){
			#pragma HLS UNROLL
			edge_index_in_bigbuf[j][i] = edge_index_in[g*N_EDGE_LAYER*TWO+i].layer[j];
		}
	}
}
void write_output(OUT_GROUP *out,layer11_t out_bigbuf[N_EDGE_GROUP][N_EDGE_LAYER*LAYER11_OUT_DIM],int g){
	for (int i = 0; i < N_EDGE_LAYER*LAYER11_OUT_DIM; i++) {
		#pragma HLS PIPELINE
		for(int j=0;j<N_EDGE_GROUP;j++){
			#pragma HLS UNROLL
			out[g*N_EDGE_LAYER*LAYER11_OUT_DIM+i].layer[j] = out_bigbuf[j][i];
		}
	}
}
void alveo_hls4ml(
	const NODE_GROUP *node_attr_in, // Read-Only Vector
	const EDGE_GROUP *edge_attr_in, // Read-Only Vector
	const INDEX_GROUP *edge_index_in, // Read-Only Vector

	OUT_GROUP *out       // Output Result
	)
{
    #pragma HLS INTERFACE m_axi port=node_attr_in bundle=gmem0
    #pragma HLS INTERFACE m_axi port=edge_attr_in bundle=gmem1
    #pragma HLS INTERFACE m_axi port=edge_index_in bundle=gmem2
    #pragma HLS INTERFACE m_axi port=out bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=node_attr_in bundle=control
    #pragma HLS INTERFACE s_axilite port=edge_attr_in bundle=control
    #pragma HLS INTERFACE s_axilite port=edge_index_in bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS data_pack variable=node_attr_in
	#pragma HLS data_pack variable=edge_attr_in
	#pragma HLS data_pack variable=edge_index_in
	#pragma HLS data_pack variable=out
	//necessary for hls4ml kernel, not used
	#pragma HLS DATAFLOW

	input_t node_attr_in_bigbuf[N_NODE_GROUP][N_NODE_LAYER*NODE_DIM];
	input3_t edge_attr_in_bigbuf[N_EDGE_GROUP][N_EDGE_LAYER*EDGE_DIM];
	input4_t edge_index_in_bigbuf[N_EDGE_GROUP][N_EDGE_LAYER*TWO];
	layer11_t out_bigbuf[N_EDGE_GROUP][N_EDGE_LAYER*LAYER11_OUT_DIM];
	
	//getting data from DRAM
	// for (int i = 0; i < N_NODE_LAYER*NODE_DIM; i++) {
	// 	for(int j=0;j<N_NODE_GROUP;j++){
	// 		#pragma HLS UNROLL
	// 		node_atrt_in_bigbuf[j][i] = node_attr_in[i*N_NODE_GROUP+j];
	// 	}
	// }
	// for (int i = 0; i < N_EDGE_LAYER*EDGE_DIM; i++) {
	// 	for(int j=0;j<N_EDGE_GROUP;j++){
	// 		#pragma HLS UNROLL
	// 		edge_attr_in_bigbuf[j][i] = edge_attr_in[i*N_EDGE_GROUP+j];
	// 	}
	// }
	// for (int i = 0; i < N_EDGE_LAYER*TWO; i++) {
	// 	for(int j=0;j<N_EDGE_GROUP;j++){
	// 		#pragma HLS UNROLL
	// 		edge_index_in_bigbuf[j][i] = edge_index_in[i*N_EDGE_GROUP+j];
	// 	}
	// }
	for(int i=0;i<N_GRAPH;i++){
		#pragma HLS DATAFLOW
		#pragma HLS INLINE
	read_node_attr(node_attr_in,node_attr_in_bigbuf,i);
	read_edge_attr(edge_attr_in,edge_attr_in_bigbuf,i);
	read_edge_index(edge_index_in,edge_index_in_bigbuf,i);


	std::cout<<"------------------"<<std::endl;
	//=============================================
	//input
	//=============================================
	std::cout<<"inf start"<<std::endl;
	myproject(node_attr_in_bigbuf,edge_attr_in_bigbuf,edge_index_in_bigbuf,out_bigbuf);
	std::cout<<"inf end"<<std::endl;
	write_output(out,out_bigbuf,i);
	}

	//=============================================
	//output
	//=============================================
	// for(int i1 = 0; i1 < DATA_SIZE_OUT*OUT_STREAM_LEN; i1++) {
	// 	#pragma HLS UNROLL
	// 	//std::cout<<"reading from ["<<i1<<"]"<<std::endl;
	// 	result_t tmp_small = out_buf.read();
	// 	out_bigbuf = tmp_small;
	// }
	// out[0] = out_bigbuf;
	// out[1] = 0;
	//std::cout <<(double)out_bigbuf<<std::endl;
}
}
