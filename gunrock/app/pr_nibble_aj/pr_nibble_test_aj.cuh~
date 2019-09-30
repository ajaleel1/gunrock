// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_nibble_test.cu
 *
 * @brief Test related functions for pr_nibble
 */

#pragma once

#include <iostream>

namespace gunrock {
namespace app {
namespace pr_nibble_aj {

/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference pr_nibble ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
 * @param[in]   ref_node      Source node
 * @param[in]   values        Array for output pagerank values
 * @param[in]   quiet         Whether to print out anything to stdout
 */

template <typename GraphT>
double CPU_Reference(const GraphT &graph, util::Parameters &parameters,
                     typename GraphT::VertexT ref_node,
                     typename GraphT::ValueT *values, bool quiet) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::VertexT VertexT;

  int num_ref_nodes = 1;  // HARDCODED

  // Graph statistics
  SizeT nodes = graph.nodes;
  ValueT num_edges = (ValueT)graph.edges / 2;
  ValueT log_num_edges = log2(num_edges);

  // Load parameters
  ValueT alpha   = 0.15;
  ValueT epsilon = 1e-9;
//   ValueT epsilon = 1e-6;

  ValueT weight1 = (2*alpha) / (1+alpha);
  ValueT weight2 = (1-alpha) / (1+alpha);

  // Init algorithm storage
  SizeT  frontier_size   = 0;
  ValueT *pageRank       = values;
  ValueT *frontier       = new ValueT[nodes];
  ValueT *residual       = new ValueT[nodes];
  ValueT *residual_prime = new ValueT[nodes];

  // Initialize The State
  for (SizeT i = 0; i < nodes; ++i) {
    pageRank[i] = (ValueT)0;
    frontier[i] = (ValueT)0;
    residual[i] = (ValueT)0;
    residual_prime[i] = (ValueT)0;
  }
  
  printf("pr_nibble::CPU_Reference_AJ: With Reference Node: %d\n", ref_node);

  // Set Up The Algorithm Start
  frontier[frontier_size]  = ref_node;
  residual[ref_node]       = 1;
  residual_prime[ref_node] = 1;
  
  frontier_size            = 1;
    
  int iter = 0;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();
  
  while( frontier_size != 0 ) {
     
      // Update The Page Rank
      for(int p=0; p<frontier_size; p++) {
              
          VertexT v = frontier[p];

          pageRank[v] += weight1 * residual[v];
          residual_prime[v] = 0;
      }


      // Propogate The Residuals To Neighbors
      for(int q=0; q<frontier_size; q++) {

          VertexT s = frontier[q];

          SizeT num_neighbors = graph.GetNeighborListLength(s);
          ValueT update       = weight2 * residual[s] / num_neighbors;

          for (int offset = 0; offset < num_neighbors; ++offset) {
              VertexT d = graph.GetEdgeDest(graph.GetNeighborListOffset(s) + offset);
              residual_prime[d] += update;
          }
      }

      // Generate The New Frontier
      frontier_size      = 0;

      for (int v = 0; v < nodes; v++) {
          
          // copy the update residuals
          residual[v] = residual_prime[v];

          SizeT num_neighbors = graph.GetNeighborListLength(v);
          
          if( num_neighbors && (residual[v] >= (num_neighbors * epsilon)))
          {
              
              frontier[frontier_size] = v;
              frontier_size++;
          }
      }

      printf("\tIteration: %d Frontier Size: %d\n", iter, frontier_size);


      iter++;
  }

  printf("\n\n\n");

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  return elapsed;
}


/**
 * @brief Validation of pr_nibble results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  h_values      GPU PR values
 * @param[in]  ref_values    CPU PR values
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(util::Parameters &parameters,
                                        GraphT &graph,
                                        typename GraphT::ValueT *h_values,
                                        typename GraphT::ValueT *ref_values,
                                        bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;

  bool quiet = parameters.Get<bool>("quiet");

  // Check agreement (within a small tolerance)
  SizeT num_errors = 0;
  ValueT tolerance = 0.00001;
  for (SizeT i = 0; i < graph.nodes; i++) {
    if (h_values[i] != ref_values[i]) {
      float err = abs(h_values[i] - ref_values[i]) / abs(ref_values[i]);
      if (err > tolerance) {
        num_errors++;
        // printf("FAIL: [%d]:\t%0.17g != %0.17g\n",
        //     i, h_values[i], ref_values[i]);
      }
    }
  }

  util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);

  return num_errors;
}

}  // namespace pr_nibble_aj
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
