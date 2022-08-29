
#include "hls_stream.h"

#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include "defines.h"
#include "weights/R1_w0.h"
#include "weights/R1_w1.h"
#include "weights/R1_w2.h"
#include "weights/R1_w3.h"
#include "weights/R1_b0.h"
#include "weights/R1_b1.h"
#include "weights/R1_b2.h"
#include "weights/R1_b3.h"
#include "weights/O_w0.h"
#include "weights/O_w1.h"
#include "weights/O_w2.h"
#include "weights/O_w3.h"
#include "weights/O_b0.h"
#include "weights/O_b1.h"
#include "weights/O_b2.h"
#include "weights/O_b3.h"
#include "weights/R2_w0.h"
#include "weights/R2_w1.h"
#include "weights/R2_w2.h"
#include "weights/R2_w3.h"
#include "weights/R2_b0.h"
#include "weights/R2_b1.h"
#include "weights/R2_b2.h"
#include "weights/R2_b3.h"

#ifndef __SYNTHESIS__
#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "/home/ShiYuHuang/manual_GNN_conversion/hls_output_current/add/source_to_target/memory/dataflow_stable_content/weights"
#endif

template<class T, size_t SIZE>
void load_weights_from_txt(T *w, const char* fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        size_t i = 0;
        while(std::getline(iss, token, ',')) {
            std::istringstream(token) >> w[i];
            i++;
        }

        if (SIZE != i) {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}
#endif
/*
template<class data_t,class weight_t,class output_t>
void dense(
    data_t data,weight_t weight_0,weight_t weight_1,output_t Y_0,output_t Y_1
) {

    #pragma HLS INTERFACE ap_none port=data,weight,Y
    #pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=Y dim=1 complete

    a_t A,D;
    b_t B;
    p_t P;
    p_t C;
        //#pragma HLS PIPELINE
        A=weight_0;
        //A.range(8,8)=weight[0].range(7,7);
        A.range(7,0)=weight_0.range(7,0);
        D=0;
        D.range(25,18)=weight_1.range(7,0);
        //B=0;
        B=data;
        B.range(7,0)=data.range(7,0);
        P=0;
        C=0;
        P=(A+D)*B+C;
        Y_0.range(7,0)=P.range(11,4);
        Y_1.range(7,0)=P.range(29,22);     
}
*/
template<class data_T,int par_factor, int row, int col,int dim>
    void clone_vector(
    data_T     IN  [row][col*dim],
    data_T     OUT1[row][col*dim],
    data_T     OUT2[row][col*dim]
  )
  {
    for(int j=0; j<col; j++){
      #pragma HLS UNROLL factor=par_factor*2
      #pragma HLS PIPELINE
      for(int i=0; i<row; i++){
      #pragma HLS UNROLL
        for(int c=0; c<dim; c++){
        #pragma HLS UNROLL
        OUT1[i][j*dim+c] =  IN[i][j*dim+c];
        OUT2[i][j*dim+c] =  IN[i][j*dim+c];
      }
    }
  }
  }
template<class data_T,class weight_T,class res_T>
res_T product_lut(
  data_T data,
  weight_T weight
){
  #pragma HLS INLINE
  res_T product;
  #pragma HLS RESOURCE variable=product core=Mul_LUT
  //#pragma HLS BIND_OP variable=product op=mul impl=fabric latency=1
  product=data*weight;
  return product;
}
template<class data_T,class weight_T,class res_T>
res_T product_dsp(
  data_T data,
  weight_T weight
){
  #pragma HLS INLINE
  res_T product;
  #pragma HLS RESOURCE variable=product core=DSP48
  product=data*weight;
  return product;
    
}

template<class data_T, class res_T,class weight_T>
void dsp_packing(
    data_T data,weight_T weight[2],res_T Y[2], res_T acc[2]
) {
    ap_fixed<26,26> A,D;
    ap_fixed<18,18> B;
    ap_fixed<48,48> P;
    ap_fixed<48,48> C;
        #pragma HLS INLINE 
        A=weight[0];
        //A.range(8,8)=weight[0].range(7,7);
        A.range(7,0)=weight[0].range(7,0);
        D=0;
        D.range(25,18)=weight[1].range(7,0);
        //B=0;
        B=data;
        B.range(13,0)=data.range(13,0);
        P=0;
        C=0;
        C.range(16,3)=acc[0].range(13,0);
        C.range(34,21)=acc[1].range(13,0);        
        P=(A+D)*B+C;
        //std::cout<<acc[0].range(13,0)<<std::endl;
        Y[0].range(13,0)=P.range(16,3);
        Y[1].range(13,0)=P.range(34,21);     
}

/*

template<class data_T, class res_T,class weight_T,size_t n_in,size_t n_out,bool use_dsp>
void dense_resource(
  data_T data[n_in],
  res_T  res[n_out],
  weight_T weights[n_in*n_out],
  weight_T bias[n_out]
)
{
  #pragma HLS function_instantiate variable=weights,biases
  res_T acc[n_out];
  #pragma HLS ARRAY_RESHAPE   variable=weights complete
  #pragma HLS ARRAY_PARTITION variable=acc complete
  InitAccum:
  for (int iacc = 0; iacc < n_out; iacc++) {
    #pragma HLS UNROLL
    acc[iacc] = bias[iacc];
    //std::cout<<acc[iacc]<<" ";
  }
  //std::cout<<std::endl;
    //#pragma HLS ALLOCATION function instances=product<data_T,res_T,res_T> limit=1

  MultLoop:
  for (int io = 0; io < n_out; io+=2) {
    #pragma HLS UNROLL

    if(io+1<n_out){
        for (int ii = 0; ii < n_in; ii++) {
            #pragma HLS UNROLL
            weight_T weight_p[2];
            #pragma HLS ARRAY_PARTITION variable=weight_p complete
            res_T res_p[2];
            #pragma HLS ARRAY_PARTITION variable=res_p complete
            res_T acc_p[2];
            #pragma HLS ARRAY_PARTITION variable=acc_p complete
            acc_p[0]= acc[io];
            acc_p[1]= acc[io+1];
            weight_p[0]=weights[(io+0)*n_in+ii];
            weight_p[1]=weights[(io+1)*n_in+ii];
            dsp_packing<data_T,res_T,weight_T>(data[ii],weight_p,res_p,acc_p);
            acc[io]=res_p[0];
            acc[io+1]=res_p[1];
        }
    }
    else{
        for (int ii = 0; ii < n_in; ii++) {
        #pragma HLS UNROLL
        if(use_dsp)
        acc[io]+=product_dsp<data_T,weight_T,res_T>(data[ii],weights[io*n_in+ii]);
        else
        acc[io]+=product_lut<data_T,weight_T,res_T>(data[ii],weights[io*n_in+ii]);
        //acc[io]+=data[ii]*weights[w_index];
    }
    }
  }
  Result:
    for (int ires = 0; ires < n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = acc[ires];
    }
}

*/
template<class data_T, class res_T,class weight_T,size_t n_in,size_t n_out,bool use_dsp>
void dense_resource(
  data_T data[n_in],
  res_T  res[n_out],
  weight_T weights[n_in*n_out],
  weight_T bias[n_out]
)
{
  #pragma HLS function_instantiate variable=weights,biases
  res_T acc[n_out];
  #pragma HLS ARRAY_RESHAPE   variable=weights complete
  #pragma HLS ARRAY_PARTITION variable=acc complete
  InitAccum:
  for (int iacc = 0; iacc < n_out; iacc++) {
    #pragma HLS UNROLL
    acc[iacc] = bias[iacc];
    //std::cout<<acc[iacc]<<" ";
  }
  //std::cout<<std::endl;
  int w_index=0;
  int in_index=0;
  int out_index=0;
  int acc_step=0;
    //#pragma HLS ALLOCATION function instances=product<data_T,res_T,res_T> limit=1

  MultLoop:
  for (int im = 0; im < n_in*n_out; im++) {
    #pragma HLS UNROLL
    if(use_dsp)
      acc[out_index]+=product_dsp<data_T,weight_T,res_T>(data[in_index],weights[w_index]);
    else
      acc[out_index]+=product_lut<data_T,weight_T,res_T>(data[in_index],weights[w_index]);
    //acc[out_index]+=data[in_index]*weights[w_index];
    in_index+=1;
    w_index+=1;
    acc_step+=1;
    if(in_index==n_in){
      in_index=0;
    }
    if(acc_step==n_in){
      acc_step=0;
      out_index+=1;
    }
  }
  Result:
    for (int ires = 0; ires < n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = acc[ires];
    }
}

template<class data_T, class res_T,size_t n_in>
void  relu(data_T data[n_in], res_T res[n_in])
{
  #pragma HLS PIPELINE
  data_T datareg;
  for (int ii=0; ii<n_in; ii++) {
      datareg = data[ii];
      if (datareg > 0) res[ii] = datareg;
      else res[ii] = 0;
  }
}
inline float sigmoid_fcn_float(float input) {
    return 1.0 / (1 + std::exp(-input));
}

template<class table_t, int N_TABLE>
void init_sigmoid_table(table_t table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        table_t real_val = sigmoid_fcn_float(in_val);
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T,class table_t, size_t n_in,int table_size>
void  sigmoid(data_T data[n_in], res_T res[n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    table_t sigmoid_table[table_size];
#else
    static bool initialized = false;
    static table_t sigmoid_table[table_size];
#endif
    if (!initialized) {
        init_sigmoid_table<table_t, table_size>(sigmoid_table);
        initialized = true;
    }
        #pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<n_in; ii++) {
        data_round = data[ii]*table_size/16;
        index = data_round + 8*table_size/16;
        if (index < 0)   index = 0;
        if (index > table_size-1) index = table_size-1;
        res[ii] = (res_T) sigmoid_table[index];
    }
}
template<class data_T, class res_T,class weight_T,size_t n_in, size_t n_out,bool use_dsp>
void dense_mult_3lyr(
	 data_T data[n_in],
	 res_T res[n_out],
	 weight_T weights0[n_in*8],
	 weight_T biases0[8],
	 weight_T weights1[8*8],
	 weight_T biases1[8],
	 weight_T weights2[8*n_out],
	 weight_T biases2[n_out])
{
  data_T data0_logits[8];
  #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
  dense_resource<data_T, data_T,weight_T, n_in,8,use_dsp>(data, data0_logits, weights0, biases0);
  data_T data0[8];
  #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
  relu<data_T, data_T,8>(data0_logits, data0);

  
  
  data_T data1_logits[8];
  #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
  dense_resource<data_T, data_T,weight_T,8,8,use_dsp>(data0, data1_logits, weights1, biases1); 
  data_T data1[8];
  #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
  relu<data_T, data_T,8>(data1_logits, data1); 
  dense_resource<data_T, res_T,weight_T,8,n_out,use_dsp>(data1, res, weights2, biases2);

}
  
template<class data_1_T,class data_2_T, class index_T, class res_T,class weight_T,int par_factor,size_t n_node_group, size_t n_edge_group,size_t n_node_layer, size_t n_edge_layer, size_t node_dim,size_t edge_dim, size_t out_dim, bool activate_final,bool use_dsp >
  void edgeblock(
    data_1_T    node_attr_1D[n_node_group][n_node_layer*node_dim],
    data_2_T    edge_attr_1D[n_edge_group][n_edge_layer*edge_dim],
    index_T   edge_index_1D[n_edge_group][n_edge_layer*2],
    res_T     edge_update_1D[n_edge_group][n_edge_layer*out_dim],
    weight_T  core_edge_w0[(node_dim*2 + edge_dim) *8],
    weight_T  core_edge_b0[8],
    weight_T  core_edge_w1[8*8],
    weight_T  core_edge_b1[8],
    weight_T  core_edge_w2[8*out_dim],
    weight_T  core_edge_b2[out_dim],
    weight_T  core_edge_w3[8*out_dim],
    weight_T  core_edge_b3[out_dim])
{
  #pragma HLS INLINE
  int sender_col;
  int receiver_col;
  sender_col = 0;
  receiver_col = 1;
  int s_index_offset[13]={0,1,2,0,1,2,3,4,5,6,7,8,9};
  int r_index_offset[13]={1,2,3,4,4,4,4,5,6,7,8,9,10};

  data_1_T node_attr_1D_s_mat[n_edge_group][n_node_layer][node_dim][par_factor];
  #pragma HLS ARRAY_PARTITION variable=node_attr_1D_s_mat complete dim=1
//  #pragma HLS ARRAY_PARTITION variable=node_attr_1D_s_mat cyclic factor=par_factor dim=2
  #pragma HLS ARRAY_PARTITION variable=node_attr_1D_s_mat complete dim=3
  #pragma HLS ARRAY_PARTITION variable=node_attr_1D_s_mat complete dim=4
  data_1_T node_attr_1D_r_mat[n_edge_group][n_node_layer][node_dim][par_factor];
  #pragma HLS ARRAY_PARTITION variable=node_attr_1D_r_mat complete dim=1
//  #pragma HLS ARRAY_PARTITION variable=node_attr_1D_r_mat cyclic factor=par_factor dim=2
  #pragma HLS ARRAY_PARTITION variable=node_attr_1D_r_mat complete dim=3
  #pragma HLS ARRAY_PARTITION variable=node_attr_1D_r_mat complete dim=4

  /*
  for(int i=0;i<40;i++)
    std::cout<<core_edge_w0[i]<<" ";
  std::cout<<std::endl;
  std::cout<<std::endl;
  */

  
  edge_choose_vertex_loop:for(int j=0;j<n_node_layer;j=j+1){
    #pragma HLS UNROLL factor=2*par_factor
    #pragma HLS PIPELINE II=1
    for (int c=0; c < node_dim; c++){
      #pragma HLS UNROLL
      for (int p=0; p < par_factor; p++){
      #pragma HLS UNROLL
      node_attr_1D_s_mat[0][j][c][p] = node_attr_1D[0][j*node_dim+c];
      node_attr_1D_s_mat[1][j][c][p] = node_attr_1D[1][j*node_dim+c];
      node_attr_1D_s_mat[2][j][c][p] = node_attr_1D[2][j*node_dim+c];
      node_attr_1D_s_mat[3][j][c][p] = node_attr_1D[0][j*node_dim+c];
      node_attr_1D_s_mat[4][j][c][p] = node_attr_1D[1][j*node_dim+c];
      node_attr_1D_s_mat[5][j][c][p] = node_attr_1D[2][j*node_dim+c];
      node_attr_1D_s_mat[6][j][c][p] = node_attr_1D[3][j*node_dim+c];
      node_attr_1D_s_mat[7][j][c][p] = node_attr_1D[4][j*node_dim+c];
      node_attr_1D_s_mat[8][j][c][p] = node_attr_1D[5][j*node_dim+c];
      node_attr_1D_s_mat[9][j][c][p] = node_attr_1D[6][j*node_dim+c];
      node_attr_1D_s_mat[10][j][c][p] = node_attr_1D[7][j*node_dim+c];
      node_attr_1D_s_mat[11][j][c][p] = node_attr_1D[8][j*node_dim+c];
      node_attr_1D_s_mat[12][j][c][p] = node_attr_1D[9][j*node_dim+c];
      node_attr_1D_r_mat[0][j][c][p] = node_attr_1D[1][j*node_dim+c];
      node_attr_1D_r_mat[1][j][c][p] = node_attr_1D[2][j*node_dim+c];
      node_attr_1D_r_mat[2][j][c][p] = node_attr_1D[3][j*node_dim+c];
      node_attr_1D_r_mat[3][j][c][p] = node_attr_1D[4][j*node_dim+c];
      node_attr_1D_r_mat[4][j][c][p] = node_attr_1D[4][j*node_dim+c];
      node_attr_1D_r_mat[5][j][c][p] = node_attr_1D[4][j*node_dim+c];
      node_attr_1D_r_mat[6][j][c][p] = node_attr_1D[4][j*node_dim+c];
      node_attr_1D_r_mat[7][j][c][p] = node_attr_1D[5][j*node_dim+c];
      node_attr_1D_r_mat[8][j][c][p] = node_attr_1D[6][j*node_dim+c];
      node_attr_1D_r_mat[9][j][c][p] = node_attr_1D[7][j*node_dim+c];
      node_attr_1D_r_mat[10][j][c][p] = node_attr_1D[8][j*node_dim+c];
      node_attr_1D_r_mat[11][j][c][p] = node_attr_1D[9][j*node_dim+c];
      node_attr_1D_r_mat[12][j][c][p] = node_attr_1D[10][j*node_dim+c];
      }
    }
  }
  edge_compute_loop:for(int i = 0; i < n_edge_layer; i+=1) { //for each edge
    #pragma HLS UNROLL factor=par_factor*2
    #pragma HLS PIPELINE II=1
    edge_loop_1: for(int g=0;g<n_edge_group;g=g+1){
    #pragma HLS UNROLL
    data_2_T edge_attr[edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
    trans_loop_1: for (int c=0; c < edge_dim; c++){
      #pragma HLS UNROLL
      edge_attr[c] = edge_attr_1D[g][i*edge_dim+c];
    }
    index_T s = edge_index_1D[g][i*2+sender_col] - s_index_offset[g]*n_node_layer;
    index_T r = edge_index_1D[g][i*2+receiver_col] - r_index_offset[g]*n_node_layer;
    data_1_T node_attr_r[node_dim];
    #pragma HLS ARRAY_PARTITION variable=node_attr_r complete dim=0
    data_1_T node_attr_s[node_dim];
    #pragma HLS ARRAY_PARTITION variable=node_attr_s complete dim=0
    trans_loop_3: for (int c=0; c < node_dim; c++){
      #pragma HLS UNROLL
      node_attr_s[c] = node_attr_1D_s_mat[g][s][c][i%par_factor];
      node_attr_r[c] = node_attr_1D_r_mat[g][r][c][i%par_factor];
    }
    data_1_T phi_input[edge_dim + 2*node_dim];
    #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
    for(int j=0; j<node_dim; j++){
      #pragma HLS UNROLL
      phi_input[j] = node_attr_r[j];
      phi_input[node_dim+j] = node_attr_s[j];
    }
    for(int k=0; k<edge_dim; k++){
      #pragma HLS UNROLL
      phi_input[2*node_dim+k] = edge_attr[k];
    }
    res_T edge_update[out_dim];
     //#pragma HLS RESOURCE variable=edge_update core=Mul_LUT   
    #pragma HLS ARRAY_PARTITION variable=edge_update complete dim=0


    // send it through NN

    
    dense_mult_3lyr<data_1_T, res_T,weight_T,node_dim*2+edge_dim,out_dim,use_dsp>(phi_input, edge_update, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
    // for(int k=0;k<node_dim*2+edge_dim;k++)
    //   std::cout<<phi_input[k]<<" ";
    // std::cout<<"\n";
    res_T edge_update_act [out_dim];
    if(activate_final){
      #pragma HLS ARRAY_PARTITION variable=edge_update_act dim=0
      sigmoid<res_T, res_T,ap_fixed<18,8>,out_dim,1024>(edge_update, edge_update_act);
    }
    trans_loop_5: for (int c=0; c < out_dim; c++){
      #pragma HLS UNROLL
      if(activate_final){
        edge_update_1D[g][i*out_dim+c] = edge_update_act[c];
      }
      else{
        edge_update_1D[g][i*out_dim+c] = edge_update[c];
      }
    }
  }
  // std::cout<<"\n";
  }
  
}

// template<class data_T, class index_T, class res_T,int par_factor,size_t n_node_group, size_t n_edge_group,size_t n_node_layer, size_t n_edge_layer, size_t node_dim,size_t edge_dim>
//     void edge_aggregate(
//             data_T    edge_attr_1D[n_edge_group][n_edge_layer*edge_dim],
//             index_T   edge_index_1D[n_edge_group][n_edge_layer*2],
//             res_T     edge_attr_aggr_1D[n_node_group][n_node_layer*edge_dim])
//   {
//   #pragma HLS INLINE 
//     res_T edge_attr_aggr[13][par_factor*8][n_node_layer][edge_dim];
//     #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=1 
//     #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=2
//     #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr cyclic factor=par_factor dim=3
//     #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=4  
//     int r_index_offset_1[13]={1,2,3,4,4,4,4,5,6,7,8,9,10};
//     #pragma HLS function_instantiate variable=r_index_offset_1
//     int receiver_col;
//     receiver_col = 1;
//     int loop_count;
//     fetch_loop:for (int i=0; i < n_node_layer; i++){
//       #pragma HLS UNROLL factor=par_factor*2
//       #pragma HLS PIPELINE II=1 

//       for (int g=0; g < n_edge_group; g++){
//         #pragma HLS UNROLL
//         for (int p=0; p < par_factor*8 ; p++){
//         #pragma HLS UNROLL
//         for (int c=0; c < edge_dim; c++){
//           #pragma HLS UNROLL           
//           edge_attr_aggr[g][p][i][c]=0;
//          }
//         }  
//       }
//     }
                     
//     agg_loop:for(int i=0; i < n_edge_layer; i++){
//       #pragma HLS UNROLL factor=par_factor*8
//       #pragma HLS PIPELINE II=1 
//       for (int g=0; g < n_edge_group; g++){
//         #pragma HLS UNROLL 
//         index_T r = edge_index_1D[g][i*2+receiver_col]-r_index_offset_1[g]*n_node_layer; 
//         for(int j=0; j<edge_dim; j++){
//           #pragma HLS UNROLL
//           edge_attr_aggr[g][i%(par_factor*8)][r][j]=edge_attr_aggr[g][i%(par_factor*8)][r][j]+edge_attr_1D[g][i*edge_dim+j];
//         }
//       }    
//     }                                                      
//     //output array --> output vec
//     out_loop:for (int r=0; r < n_node_layer; r++){
//       #pragma HLS UNROLL factor=2
//       #pragma HLS PIPELINE II=1 
//         for (int c=0; c<edge_dim; c++){
//         #pragma HLS UNROLL
//         res_T edge_attr_aggr_total[13];
//         for (int g=0; g<13; g++){
//           #pragma HLS UNROLL
//           for(int p=0;p<(par_factor*8);p++){
//             if(p==0)edge_attr_aggr_total[g]=edge_attr_aggr[g][p][r][c];
//             else edge_attr_aggr_total[g]+=edge_attr_aggr[g][p][r][c];
//           }
          
//         }
//           edge_attr_aggr_1D[1][r*edge_dim+c]=edge_attr_aggr_total[0];
//           edge_attr_aggr_1D[2][r*edge_dim+c]=edge_attr_aggr_total[1];
//           edge_attr_aggr_1D[3][r*edge_dim+c]=edge_attr_aggr_total[2];
//           edge_attr_aggr_1D[4][r*edge_dim+c]=edge_attr_aggr_total[3]+edge_attr_aggr_total[4]+edge_attr_aggr_total[5]+edge_attr_aggr_total[6];
//           edge_attr_aggr_1D[5][r*edge_dim+c]=edge_attr_aggr_total[7];
//           edge_attr_aggr_1D[6][r*edge_dim+c]=edge_attr_aggr_total[8];
//           edge_attr_aggr_1D[7][r*edge_dim+c]=edge_attr_aggr_total[9];
//           edge_attr_aggr_1D[8][r*edge_dim+c]=edge_attr_aggr_total[10];
//           edge_attr_aggr_1D[9][r*edge_dim+c]=edge_attr_aggr_total[11];
//           edge_attr_aggr_1D[10][r*edge_dim+c]=edge_attr_aggr_total[12];
//         }
//     }   
//   //std::cout<<"\n";  
//   }
template<class data_T, class index_T, class res_T,int par_factor,size_t n_node_group, size_t n_edge_group,size_t n_node_layer, size_t n_edge_layer, size_t node_dim,size_t edge_dim>
    void edge_aggregate(
            data_T    edge_attr_1D[n_edge_group][n_edge_layer*edge_dim],
            index_T   edge_index_1D[n_edge_group][n_edge_layer*2],
            res_T     edge_attr_aggr_1D[n_node_group][n_node_layer*edge_dim])
  {
    #pragma HLS INLINE 
    res_T edge_attr_aggr[n_edge_group][par_factor*4][n_node_layer][edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=1 
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=2
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr cyclic factor=par_factor dim=3
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=4  
    int r_index_offset_1[13]={1,2,3,4,4,4,4,5,6,7,8,9,10};
    //#pragma HLS function_instantiate variable=r_index_offset_1
    int receiver_col;
    receiver_col = 1;
    int loop_count;
    fetch_loop:for (int i=0; i < n_node_layer; i++){
      #pragma HLS UNROLL factor=par_factor*2
      #pragma HLS PIPELINE II=1 

      for (int g=0; g < n_edge_group; g++){
        #pragma HLS UNROLL
        for (int p=0; p < par_factor*4 ; p++){
        #pragma HLS UNROLL
        for (int c=0; c < edge_dim; c++){
          #pragma HLS UNROLL           
          edge_attr_aggr[g][p][i][c]=0;
         }
        }  
      }
    }
    index_T old[n_edge_group][par_factor*4];
    #pragma HLS ARRAY_PARTITION variable=old complete dim=0
    res_T agg[n_edge_group][edge_dim][par_factor*4];
    #pragma HLS  ARRAY_PARTITION variable=agg complete dim=0
    index_T old_p[n_edge_group][par_factor*4];
    #pragma HLS  ARRAY_PARTITION variable=old_p complete dim=0
    res_T agg_p[n_edge_group][edge_dim][par_factor*4];
    #pragma HLS  ARRAY_PARTITION variable=agg_p complete dim=0    
    res_T agg_temp[n_edge_group][edge_dim][par_factor*4];
    #pragma HLS  ARRAY_PARTITION variable=agg_temp complete dim=0        
    for (int g=0; g < n_edge_group; g++){
        #pragma HLS UNROLL 
        for(int p=0; p<par_factor*4; p++){
          #pragma HLS UNROLL
          old[g][p] = edge_index_1D[g][p*2+receiver_col]-r_index_offset_1[g]*n_node_layer; 
          old_p[g][p] = edge_index_1D[g][p*2+receiver_col]-r_index_offset_1[g]*n_node_layer; 
        }
        for(int j=0; j<edge_dim; j++){
            #pragma HLS UNROLL
            for(int p=0; p<par_factor*4; p++){
              #pragma HLS UNROLL
              agg[g][j][p] = 0;
              agg_p[g][j][p] = 0;
            }
        }
    }              
    agg_loop:for(int i=0; i < n_edge_layer; i++){
      #pragma HLS UNROLL factor=par_factor*4
      #pragma HLS PIPELINE II=1 
      #pragma HLS DEPENDENCE variable=edge_attr_aggr intra RAW false
      #pragma HLS DEPENDENCE variable=edge_attr_aggr inter RAW false
      for (int g=0; g < n_edge_group; g++){
        #pragma HLS UNROLL 
        index_T r = edge_index_1D[g][i*2+receiver_col]-r_index_offset_1[g]*n_node_layer; 
        if(r==old[g][i%(par_factor*4)]){
          for(int j=0; j<edge_dim; j++){
            #pragma HLS UNROLL
           agg[g][j][i%(par_factor*4)]=agg[g][j][i%(par_factor*4)]+edge_attr_1D[g][i*edge_dim+j];
          }
        }
        else{
          for(int j=0; j<edge_dim; j++){
            #pragma HLS UNROLL
            edge_attr_aggr[g][i%(par_factor*4)][old[g][i%(par_factor*4)]][j]=agg[g][j][i%(par_factor*4)];
            agg_temp[g][j][i%(par_factor*4)]=agg_p[g][j][i%(par_factor*4)];
            agg_p[g][j][i%(par_factor*4)]=agg[g][j][i%(par_factor*4)];
            if(r==old_p[g][i%(par_factor*4)]){
              agg[g][j][i%(par_factor*4)]= agg_temp[g][j][i%(par_factor*4)]+edge_attr_1D[g][i*edge_dim+j];
              //std::cout<<i<<" ";
            }
            else
              agg[g][j][i%(par_factor*4)]= edge_attr_aggr[g][i%(par_factor*4)][r][j]+edge_attr_1D[g][i*edge_dim+j];
          }
          old_p[g][i%(par_factor*4)]=old[g][i%(par_factor*4)];
        }
        old[g][i%(par_factor*4)]=r;
      }    

    }
    end_loop:for(int i=n_edge_layer-par_factor*4; i < n_edge_layer; i++){
      #pragma HLS UNROLL
      for (int g=0; g < n_edge_group; g++){
        #pragma HLS UNROLL
        for(int j=0; j<edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[g][i%(par_factor*4)][old[g][i%(par_factor*4)]][j]=agg[g][j][i%(par_factor*4)];
        }
      } 
    }                                                 
    //output array --> output vec
    out_loop:for (int r=0; r < n_node_layer; r++){
      #pragma HLS UNROLL factor=par_factor*2
      #pragma HLS PIPELINE II=1 
        for (int c=0; c<edge_dim; c++){
        #pragma HLS UNROLL
        res_T edge_attr_aggr_total[n_edge_group];
        for (int g=0; g<n_edge_group; g++){
          #pragma HLS UNROLL
          for(int p=0;p<(par_factor*4);p++){
            if(p==0)edge_attr_aggr_total[g]=edge_attr_aggr[g][p][r][c];
            else edge_attr_aggr_total[g]+=edge_attr_aggr[g][p][r][c];
          }
          
        }
          edge_attr_aggr_1D[1][r*edge_dim+c]=edge_attr_aggr_total[0];
          edge_attr_aggr_1D[2][r*edge_dim+c]=edge_attr_aggr_total[1];
          edge_attr_aggr_1D[3][r*edge_dim+c]=edge_attr_aggr_total[2];
          edge_attr_aggr_1D[4][r*edge_dim+c]=edge_attr_aggr_total[3]+edge_attr_aggr_total[4]+edge_attr_aggr_total[5]+edge_attr_aggr_total[6];
          edge_attr_aggr_1D[5][r*edge_dim+c]=edge_attr_aggr_total[7];
          edge_attr_aggr_1D[6][r*edge_dim+c]=edge_attr_aggr_total[8];
          edge_attr_aggr_1D[7][r*edge_dim+c]=edge_attr_aggr_total[9];
          edge_attr_aggr_1D[8][r*edge_dim+c]=edge_attr_aggr_total[10];
          edge_attr_aggr_1D[9][r*edge_dim+c]=edge_attr_aggr_total[11];
          edge_attr_aggr_1D[10][r*edge_dim+c]=edge_attr_aggr_total[12];
        }
    }   
  }
template<class data_1_T,class data_2_T, class res_T,class weight_T,int par_factor,size_t n_node_group, size_t n_edge_group,size_t n_node_layer, size_t n_edge_layer, size_t node_dim,size_t edge_dim, size_t out_dim,bool activate_final,bool use_dsp> 
void nodeblock(
  data_1_T    node_attr_1D[n_node_group][n_node_layer*node_dim],
  data_2_T    edge_attr_aggr_1D[n_node_group][n_node_layer*edge_dim],
  res_T     node_update_1D[n_node_group][n_node_layer*out_dim],
  weight_T core_node_w0[(node_dim+edge_dim)*8],
  weight_T core_node_b0[8],
  weight_T core_node_w1[8*8],
  weight_T core_node_b1[8],
  weight_T core_node_w2[8*out_dim],
  weight_T core_node_b2[out_dim],
  weight_T core_node_w3[8*out_dim],
  weight_T core_node_b3[out_dim])
{
  #pragma HLS INLINE
  node_compute_loop: for(int i = 0; i < n_node_layer; i++){ //for each node
    #pragma HLS UNROLL factor=par_factor
    #pragma HLS PIPELINE II=1
    for(int g=0;g<n_node_group;g++){
    #pragma HLS UNROLL
    data_2_T phi_input[edge_dim + node_dim];
    for(int c=0; c<node_dim; c++){
      #pragma HLS UNROLL
      phi_input[c] =node_attr_1D[g][i*node_dim+c];
    }
    for(int c=0; c<edge_dim; c++){
      #pragma HLS UNROLL
      if(g==0)
        phi_input[node_dim+c] = 0;
      else phi_input[node_dim+c] = edge_attr_aggr_1D[g][i*edge_dim+c];
    }
    #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
    res_T node_update[out_dim];
    #pragma HLS ARRAY_PARTITION variable=node_update complete dim=0
    dense_mult_3lyr<data_2_T, res_T,weight_T,node_dim+edge_dim,out_dim,use_dsp>(phi_input, node_update, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
    
    res_T node_update_act [out_dim];
      if (activate_final){
        #pragma HLS ARRAY_PARTITION variable=node_update_act dim=0
        sigmoid<res_T, res_T,ap_fixed<18,8>,out_dim,1024>(node_update, node_update_act);
      }
      trans_loop_3: for (int c=0; c < out_dim; c++){
      #pragma HLS UNROLL
      if (activate_final){
        node_update_1D[g][i*out_dim+c] = node_update_act[c];
      }
      else{
        node_update_1D[g][i*out_dim+c] = node_update[c];
      }
    }
    
  }
  }
  // for(int i=0;i<11;i++){
  //   for(int j=0;j<n_node_layer;j++){
  //     for(int k=0;k<out_dim;k++){
  //       std::cout<<node_update_1D[i][j*out_dim+k]<<" ";
  //     }
  //     std::cout<<"\n";
  //   }
  // }
}
