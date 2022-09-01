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

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#include "xcl2.hpp"
#include <vector>
#include<defines.h>
#define STRINGIFY2(var) #var
#define STRINGIFY(var) STRINGIFY2(var)


int main(int argc, char** argv)
{

    int nevents = 1;
	cl_int err;
	cl::Kernel krnl_aws_hls4ml;
    std::string datadir = STRINGIFY(HLS4ML_DATA_DIR);
	std::string xclbinFilename = "";
	if (argc > 1) xclbinFilename = argv[1];
	if (argc > 2) nevents = atoi(argv[2]);
	if (argc > 3) datadir = argv[3];
    std::cout << "Will run " << nevents << " time(s), using " << datadir << " to get input features and output predictions (tb_input_features.dat and tb_output_predictions.dat)" << std::endl;

    size_t vector_size_node_attr_in_bytes = sizeof(NODE_GROUP) *N_NODE_LAYER*NODE_DIM*N_GRAPH;
    size_t vector_size_edge_attr_in_bytes = sizeof(EDGE_GROUP) *N_EDGE_LAYER*EDGE_DIM*N_GRAPH;
    size_t vector_size_edge_index_in_bytes = sizeof(INDEX_GROUP) *N_EDGE_LAYER*TWO*N_GRAPH;
    size_t vector_size_out_bytes = sizeof(OUT_GROUP) *N_EDGE_LAYER*LAYER11_OUT_DIM*N_GRAPH;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 
    std::vector<NODE_GROUP,aligned_allocator<NODE_GROUP>> source_hw_node_attr_in(N_NODE_LAYER*NODE_DIM*N_GRAPH);
    std::vector<EDGE_GROUP,aligned_allocator<EDGE_GROUP>> source_hw_edge_attr_in(N_EDGE_LAYER*EDGE_DIM*N_GRAPH);
    std::vector<INDEX_GROUP,aligned_allocator<INDEX_GROUP>> source_hw_edge_index_in(N_EDGE_LAYER*TWO*N_GRAPH);
    std::vector<OUT_GROUP,aligned_allocator<OUT_GROUP>> source_hw_results(N_EDGE_LAYER*LAYER11_OUT_DIM*N_GRAPH);

	//Reset the input data
	for(int i0 = 0; i0 < N_NODE_LAYER*NODE_DIM*N_GRAPH; i0++) { 
    for(int j=0;j<N_NODE_GROUP;j++){
		  source_hw_node_attr_in[i0].layer[j] = 0;
    }
		//std::cout<<(double)fpga.source_in[i0]<<std::endl;
	}
	for(int i0 = 0; i0 < N_EDGE_LAYER*EDGE_DIM*N_GRAPH; i0++) { 
    for(int j=0;j<N_EDGE_GROUP;j++){
      source_hw_edge_attr_in[i0].layer[j] = 0;
    }
		//std::cout<<(double)fpga.source_in[i0]<<std::endl;
	}
	for(int i0 = 0; i0 < N_EDGE_LAYER*TWO*N_GRAPH; i0++) { 
    for(int j=0;j<N_EDGE_GROUP;j++){
		  source_hw_edge_index_in[i0].layer[j] = 0;
    }
		//std::cout<<(double)fpga.source_in[i0]<<std::endl;
	}
	//Reset the output result
		for(int i0 = 0; i0 < N_EDGE_LAYER*LAYER11_OUT_DIM*N_GRAPH; i0++) { 
      for(int j=0;j<N_EDGE_GROUP;j++){
		    source_hw_results[i0].layer[j] = 0;
      }
		//std::cout<<(double)fpga.source_in[i0]<<std::endl;
	}

//---------------------------------------------------------------------------------------

// OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
    std::cout << "Found Device=" << device_name.c_str() << std::endl;
	
	cl::Program::Binaries bins;
	// Load xclbin
	std::cout << "Loading: '" << xclbinFilename << "'\n";
	std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
	bin_file.seekg (0, bin_file.end);
	unsigned nb = bin_file.tellg();
	bin_file.seekg (0, bin_file.beg);
	char *buf = new char [nb];
	bin_file.read(buf, nb);
	// Creating Program from Binary File
	bins.push_back({buf,nb});
	
	//program the device
	bool valid_device = false;
	cl::Program program(context, {device}, bins, nullptr, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Failed to program device with xclbin file!\n";
	}else {
		std::cout <<"program successful!\n";
		
		std::string cu_id = std::to_string(1);
		std::string krnl_name_full = "alveo_hls4ml";
		printf("Creating a kernel [%s] for CU(%d)\n", krnl_name_full.c_str(), 0);
		krnl_aws_hls4ml = cl::Kernel(program,"alveo_hls4ml");
		valid_device = true;
	}
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
	
    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    cl::Buffer buffer_node_attr_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_node_attr_in_bytes, source_hw_node_attr_in.data());
	cl::Buffer buffer_edge_attr_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_edge_attr_in_bytes, source_hw_edge_attr_in.data());
	cl::Buffer buffer_edge_index_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_edge_index_in_bytes, source_hw_edge_index_in.data());		
    cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            vector_size_out_bytes, source_hw_results.data());

    int narg = 0;
    krnl_aws_hls4ml.setArg(narg++, buffer_node_attr_in);
    krnl_aws_hls4ml.setArg(narg++, buffer_edge_attr_in);
    krnl_aws_hls4ml.setArg(narg++, buffer_edge_index_in);
    krnl_aws_hls4ml.setArg(narg++, buffer_output);

    auto t1 = Clock::now();
    auto t2 = Clock::now();
		
//=====================
//input
//=====================
for(int e=0;e<N_GRAPH;e++){
  std::string in_data;  
  in_data = datadir+"/test_graph/graph1/input_data.txt";
  std::ifstream fin(in_data);
  std::string out_data;  
  out_data = datadir+"/test_graph/graph1/target_data.txt";
  std::ifstream fout(out_data);  

  layer11_t golden_target[N_EDGE_GROUP*N_EDGE_LAYER*LAYER11_OUT_DIM];
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
	fin.close();

	for (int i = 0; i < N_NODE_LAYER*NODE_DIM; i++) {
    for(int j=0;j<N_NODE_GROUP;j++){
			source_hw_node_attr_in[e*N_NODE_LAYER*NODE_DIM+i].layer[j] = in[j*N_NODE_LAYER*NODE_DIM+i];
		}
	}
	
	for (int i = 0; i < N_EDGE_LAYER*EDGE_DIM; i++) {
    for(int j=0;j<N_EDGE_GROUP;j++){
			source_hw_edge_attr_in[e* N_EDGE_LAYER*EDGE_DIM+i].layer[j] = in[N_NODE*NODE_DIM+j*N_EDGE_LAYER*EDGE_DIM+i];
		}
	}
	
	for (int i = 0; i < N_EDGE_LAYER*TWO; i++) {
    for(int j=0;j<N_EDGE_GROUP;j++){
			source_hw_edge_index_in[e*N_EDGE_LAYER*TWO+i].layer[j] = in[N_NODE*NODE_DIM + N_EDGE*EDGE_DIM+j*N_EDGE_LAYER*TWO+i];
    }
	}
	for(int i = 0 ; i < N_EDGE_LAYER*LAYER11_OUT_DIM ; i++){
    for(int j=0;j<N_EDGE_GROUP;j++){
		source_hw_results[e*N_EDGE_LAYER*LAYER11_OUT_DIM+i].layer[j] = 0;
	}
  }
  }
  // Declare streams
  else{
    std::cout<<"ERROR: cannot load intput file\n";
  }
}
  // if(fout.is_open()){
  //   std::cout<<"load with "<<out_data<<" \n";
  //   std::getline(fout,iline);
  //   char* cstr=const_cast<char*>(iline.c_str());
  //   char* current;
  //   std::vector<float> in;
  //   current=strtok(cstr," ");
  //   while(current!=NULL) {
  //     in.push_back(atof(current));
  //     current=strtok(NULL," ");
  //   }
	// for(int j=0;j<N_EDGE_GROUP;j++){
	// 	for (int i = 0; i < N_EDGE_LAYER*LAYER11_OUT_DIM; i++) {
	// 		golden_target[j*N_EDGE_LAYER*LAYER11_OUT_DIM+i] = in[j*N_EDGE_LAYER*LAYER11_OUT_DIM+i];
	// 	}
	// }
  //   fout.close();
  // }
  // Declare streams
  // else{
  //   std::cout<<"ERROR: cannot load output file\n";
  // }
        t1 = Clock::now();
        // Copy input data to device global memory
        q.enqueueMigrateMemObjects({buffer_node_attr_in,buffer_edge_attr_in,buffer_edge_index_in},0/* 0 means from host*/);
        // Launch the Kernel
        // For HLS kernels global and local size is always (1,1,1). So, it is recommended
        // to always use enqueueTask() for invoking HLS kernel
        q.enqueueTask(krnl_aws_hls4ml);
        // Copy Result from Device Global Memory to Host Local Memory
        q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);
        // Check for any errors from the command queue
        q.finish();
        t2 = Clock::now();
        std::cout << "FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;



//=====================
//output result
//=====================
	// for(int i=0;i<N_EDGE_LAYER*LAYER11_OUT_DIM ;i++){
  //   for(int j=0;j<N_EDGE_GROUP;j++)
	// 	if( golden_target[j*N_EDGE_LAYER*LAYER11_OUT_DIM+i] - source_hw_results[i].layer[j]<-0.5 || golden_target[j*N_EDGE_LAYER*LAYER11_OUT_DIM+i] - source_hw_results[i].layer[j]>0.5)
	// 	std::cout << golden_target[j*N_EDGE_LAYER*LAYER11_OUT_DIM+i] << " "<<source_hw_results[i].layer[j];
	// }
	// for(int i=0;i<N_EDGE_GROUP*N_EDGE_LAYER*LAYER11_OUT_DIM ;i++){
	// 	std::cout << golden_target[i] << " ";
	// }
	for(int e=0;e<N_GRAPH;e++){
	std::cout<<"Quantized predictions: \n";
	for(int i=0;i<N_EDGE_LAYER*LAYER11_OUT_DIM ;i++){
    for(int j=0;j<N_EDGE_GROUP;j++){
      std::cout << source_hw_results[e*N_EDGE_LAYER*LAYER11_OUT_DIM+i].layer[j]<< " ";
    }
		
	}
  }
	// std::cout << std::endl;
	// std::cout<<"---- END EVENT "<<" ----"<<std::endl;

	return EXIT_SUCCESS;

}
