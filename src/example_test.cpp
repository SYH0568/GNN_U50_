  // 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
#include "hls_stream.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "example_test.h"
#include "defines.h"

//hls-fpga-machine-learning insert numbers

//hls-fpga-machine-learning insert layer-precision

void example(
    hls::stream<input_t> node_attr_mat_s[N_NODE_GROUP*NODE_DIM],hls::stream<input3_t> edge_attr_mat_s[N_EDGE_GROUP*EDGE_DIM], hls::stream<input4_t> edge_index_mat_s[N_EDGE_GROUP*TWO],
    hls::stream<layer11_t> layer11_out_s[N_EDGE_GROUP*LAYER11_OUT_DIM],
    unsigned short &const_size_in_1, unsigned short &const_size_in_2, unsigned short &const_size_in_3,
    unsigned short &const_size_out_1
);
int main()
{
  std::string in_data;  
  in_data = "/home/ShiYuHuang/manual_GNN_conversion/hls_output_current/add/source_to_target/memory/dataflow_stable_content/test_graph/graph1/input_data.txt";
  std::ifstream fin(in_data);
  std::string out_data;  
  out_data = "/home/ShiYuHuang/manual_GNN_conversion/hls_output_current/add/source_to_target/memory/dataflow_stable_content/test_graph/graph1/target_data.txt";
  std::ifstream fout(out_data);  
  input_t node_attr[N_NODE*NODE_DIM];
  input3_t edge_attr[N_EDGE*EDGE_DIM];
  input4_t edge_index[N_EDGE*TWO];
  layer11_t golden_target[N_EDGE*LAYER11_OUT_DIM];
  std::string iline;
  if(fin.is_open()){
    std::cout<<"load with "<<in_data<<" \n";
    std::getline(fin,iline);
    char* cstr=const_cast<char*>(iline.c_str());
    char* current;
    std::vector<float> in;
    current=strtok(cstr," ");
    while(current!=NULL) {
      in.push_back(atof(current));
      current=strtok(NULL," ");
    }
    copy_data<float, input_t, 0, N_NODE*NODE_DIM>(in, node_attr);
    copy_data<float, input3_t, N_NODE*NODE_DIM, N_EDGE*EDGE_DIM>(in, edge_attr);
    copy_data<float, input4_t, N_NODE*NODE_DIM + N_EDGE*EDGE_DIM, N_EDGE*TWO>(in, edge_index);
    fin.close();
  }
  // Declare streams
  else{
    std::cout<<"ERROR: cannot load intput file\n";
  }
  if(fout.is_open()){
    std::cout<<"load with "<<out_data<<" \n";
    std::getline(fout,iline);
    char* cstr=const_cast<char*>(iline.c_str());
    char* current;
    std::vector<float> in;
    current=strtok(cstr," ");
    while(current!=NULL) {
      in.push_back(atof(current));
      current=strtok(NULL," ");
    }
    copy_data<float, layer11_t, 0, N_EDGE*LAYER11_OUT_DIM>(in, golden_target);
    fout.close();
  }
  // Declare streams
  else{
    std::cout<<"ERROR: cannot load output file\n";
  }
  layer11_t layer11_out[N_EDGE_GROUP][N_EDGE_LAYER*LAYER11_OUT_DIM];
  unsigned short size_in1,size_in2,size_in3,size_out1;
  input_t node_attr_mat[N_NODE_GROUP][N_NODE_LAYER*NODE_DIM];
  input3_t edge_attr_mat[N_EDGE_GROUP][N_EDGE_LAYER*EDGE_DIM];
  input4_t edge_index_mat[N_EDGE_GROUP][N_EDGE_LAYER*TWO];
  for(int i=0;i<N_NODE_GROUP;i++){
    for(int j=0;j<N_NODE_LAYER*NODE_DIM;j++){
      node_attr_mat[i][j]=node_attr[i*N_NODE_LAYER*NODE_DIM+j];
            //std::cout<<node_attr_mat[i][j]<<" ";

    }
  }
  for(int i=0;i<N_EDGE_GROUP;i++){
    for(int j=0;j<N_EDGE_LAYER*EDGE_DIM;j++){
      edge_attr_mat[i][j]=edge_attr[i*N_EDGE_LAYER*EDGE_DIM+j];
      //std::cout<<edge_attr_mat[i][j]<<" ";
    }
  }
  for(int i=0;i<N_EDGE_GROUP;i++){
    for(int j=0;j<N_EDGE_LAYER*TWO;j++){
      edge_index_mat[i][j]=edge_index[i*N_EDGE_LAYER*TWO+j];
      //std::cout<<edge_index_mat[i][j]<<" ";
    }
  }
   //std::cout<<"fadsssssssssssssssssss";
   int count=0;
   hls::stream<input_t> node_attr_mat_s[N_NODE_GROUP*NODE_DIM];
   hls::stream<input3_t> edge_attr_mat_s[N_EDGE_GROUP*EDGE_DIM];
   hls::stream<input4_t> edge_index_mat_s[N_EDGE_GROUP*TWO];  
   hls::stream<layer11_t>layer11_out_s[N_EDGE_GROUP*LAYER11_OUT_DIM]; 
   for (unsigned i = 0; i < N_NODE_LAYER; i++) {
    for (unsigned j = 0; j < N_NODE_GROUP; j++) {
      for (unsigned k = 0; k < NODE_DIM; k++) {
        
            node_attr_mat_s[j*NODE_DIM+k].write(node_attr_mat[j][i*NODE_DIM+k]);
        }
      }
    }
    for (unsigned i = 0; i < N_EDGE_LAYER; i++) {
    for (unsigned j = 0; j < N_EDGE_GROUP; j++) {
      for (unsigned k = 0; k < EDGE_DIM; k++) {
        
            edge_attr_mat_s[j*EDGE_DIM+k].write(edge_attr_mat[j][i*EDGE_DIM+k]);
        }
      }
    }
     for (unsigned i = 0; i < N_EDGE_LAYER; i++) {
    for (unsigned j = 0; j < N_EDGE_GROUP; j++) {
      for (unsigned k = 0; k < TWO; k++) {
       
            edge_index_mat_s[j*TWO+k].write(edge_index_mat[j][i*TWO+k]);
        }
      }
    }    
    example(node_attr_mat_s,edge_attr_mat_s,edge_index_mat_s,layer11_out_s,size_in1,size_in2,size_in3,size_out1);
        for (unsigned i = 0; i < N_EDGE_LAYER; i++) {
    for (unsigned j = 0; j < N_EDGE_GROUP; j++) {
      for (unsigned k = 0; k < LAYER11_OUT_DIM; k++) {

            layer11_out[j][i*LAYER11_OUT_DIM+k]=layer11_out_s[j*LAYER11_OUT_DIM+k].read();
        }
      }
    }   
  
  for(int i=0;i<N_EDGE_GROUP;i++){
    for(int j=0;j<N_EDGE_LAYER*LAYER11_OUT_DIM;j++){
      if(layer11_out[i][j]-golden_target[i*N_EDGE_LAYER*LAYER11_OUT_DIM+j]>0.5||golden_target[i*N_EDGE_LAYER*LAYER11_OUT_DIM+j]-layer11_out[i][j]>0.5){
        std::cout<<"different answer"<<" \n";
        std::cout<<"golden answer:"<< golden_target[i*N_EDGE_LAYER*LAYER11_OUT_DIM+j]<<" \n";
        std::cout<<"my answer:"<< layer11_out[i][j]<<" \n\n";
        count++;
      }
    }
  } 
  
  std::cout<<"total different answer: "<<count<<" \n"; 
  
  // Write data into a and b

  return 0;
}
