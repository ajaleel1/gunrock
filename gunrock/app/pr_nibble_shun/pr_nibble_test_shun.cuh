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
#include <vector>
#include <set>

namespace gunrock {
namespace app {
namespace pr_nibble_shun {

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
double CPU_Reference_Dense(const GraphT &graph, util::Parameters &parameters,
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
  ValueT alpha    = 0.15;
  ValueT epsilon  = parameters.Get<double>("eps");
  int    max_iter = parameters.Get<double>("max-iter");

  ValueT weight1 = (2*alpha) / (1+alpha);
  ValueT weight2 = (1-alpha) / (1+alpha);

  // Init algorithm storage
  SizeT  frontier_size   = 0;
  ValueT *pageRank       = values;
  
  ValueT *frontier       = new ValueT[nodes];
  ValueT *residual       = new ValueT[nodes];
  ValueT *residual_prime = new ValueT[nodes];

  // Profiling unique neighbors visited each iteration
  std::set<VertexT> visited;

  // Initialize The State
  for (SizeT i = 0; i < nodes; ++i) {
    pageRank[i] = (ValueT)0;
    frontier[i] = (ValueT)0;
    residual[i] = (ValueT)0;
    residual_prime[i] = (ValueT)0;
  }
  
  printf("pr_nibble::CPU_Reference_AJ: With Reference Node: %d With Neighbors: %d\n", ref_node, graph.GetNeighborListLength(ref_node));

  // Set Up The Algorithm Start
  frontier[frontier_size]  = ref_node;
  residual[ref_node]       = 1;
  residual_prime[ref_node] = 1;
  
  frontier_size            = 1;
    
  int iter = 0;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();
  
  util::CpuTimer pr_update_timer;
  util::CpuTimer res_update_timer;
  util::CpuTimer frontier_gen_timer;

  float pr_update    = 0;
  float res_update   = 0;
  float frontier_gen = 0;

  while( iter<max_iter ) {
     
      pr_update_timer.Start();
      // Update The Page Rank

      #pragma omp parallel for
      for(int v=0; v<nodes; v++) {
          pageRank[v] += weight1 * residual[v];
          residual_prime[v] = 0;
      }
      pr_update_timer.Stop();


      res_update_timer.Start();
      long tot_neighbors = 0;
      // Propogate The Residuals To Neighbors
      #pragma omp parallel for
      for(int s=0; s<nodes; s++) {

//           visited.insert(s);
          
          SizeT num_neighbors = graph.GetNeighborListLength(s);
          ValueT update       = weight2 * residual[s] / num_neighbors;

          for (int offset = 0; offset < num_neighbors; ++offset) {
              VertexT d = graph.GetEdgeDest(graph.GetNeighborListOffset(s) + offset);
//               visited.insert(d);

              #pragma omp atomic
              residual_prime[d] += update;

          }

          #pragma omp atomic
          tot_neighbors += num_neighbors;
      }
      res_update_timer.Stop();

      frontier_gen_timer.Start();
      for (int v = 0; v < nodes; v++) {
          
          // copy the update residuals
          residual[v] = residual_prime[v];
      }
      frontier_gen_timer.Stop();

      printf("\tIteration: %d Visited Neighbors: %ld\n", frontier_size, tot_neighbors);
//       printf("\tIteration: %d Visited Neighbors: %ld Unique: %ld\n", frontier_size, tot_neighbors, visited.size());
//       visited.clear();
      
      frontier_gen += frontier_gen_timer.ElapsedMillis();
      res_update   += res_update_timer.ElapsedMillis();
      pr_update    += pr_update_timer.ElapsedMillis();

      iter++;
  }

  printf("\n\n\n");

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  printf("pr_update: %lf res_update: %lf frontier_gen: %lf total: %lf\n", pr_update, res_update, frontier_gen, elapsed);

  return elapsed;
}


template <typename GraphT>
double CPU_Reference(const GraphT &graph, util::Parameters &parameters,
                     typename GraphT::VertexT ref_node,
                     typename GraphT::ValueT *values, bool quiet) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::VertexT VertexT;

  int dense  = parameters.Get<double>("dense");
  if( dense ) return CPU_Reference_Dense( graph, parameters, ref_node, values, quiet );

  int num_ref_nodes = 1;  // HARDCODED

  // Graph statistics
  SizeT nodes = graph.nodes;
  ValueT num_edges = (ValueT)graph.edges / 2;
  ValueT log_num_edges = log2(num_edges);

  // Load parameters
  ValueT alpha    = 0.15;
  ValueT epsilon  = parameters.Get<double>("eps");
  int    max_iter = parameters.Get<double>("max-iter");

  ValueT weight1 = (2*alpha) / (1+alpha);
  ValueT weight2 = (1-alpha) / (1+alpha);

  // Init algorithm storage
  SizeT  frontier_size   = 0;
  ValueT *pageRank       = values;
  
  ValueT *frontier       = new ValueT[nodes];
  ValueT *residual       = new ValueT[nodes];
  ValueT *residual_prime = new ValueT[nodes];

  // Profiling unique neighbors visited each iteration
  std::set<VertexT> visited;

  // Initialize The State
  for (SizeT i = 0; i < nodes; ++i) {
    pageRank[i] = (ValueT)0;
    frontier[i] = (ValueT)0;
    residual[i] = (ValueT)0;
    residual_prime[i] = (ValueT)0;
  }
  
  printf("pr_nibble::CPU_Reference_AJ: With Reference Node: %d With Neighbors: %d\n", ref_node, graph.GetNeighborListLength(ref_node));

  // Set Up The Algorithm Start
  frontier[frontier_size]  = ref_node;
  residual[ref_node]       = 1;
  residual_prime[ref_node] = 1;
  
  frontier_size            = 1;
    
  int iter = 0;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();
  
  util::CpuTimer pr_update_timer;
  util::CpuTimer res_update_timer;
  util::CpuTimer frontier_gen_timer;

  float pr_update    = 0;
  float res_update   = 0;
  float frontier_gen = 0;

  while( frontier_size > 0 && (iter<max_iter)) {
     
      pr_update_timer.Start();
      // Update The Page Rank

      #pragma omp parallel for
      for(int p=0; p<frontier_size; p++) {
              
          VertexT v = frontier[p];

          pageRank[v] += weight1 * residual[v];
          residual_prime[v] = 0;
      }
      pr_update_timer.Stop();


      res_update_timer.Start();
//      long tot_neighbors = 0;
      // Propogate The Residuals To Neighbors
      #pragma omp parallel for
      for(int q=0; q<frontier_size; q++) {

          VertexT s = frontier[q];
//           visited.insert(s);
          
          SizeT num_neighbors = graph.GetNeighborListLength(s);
          ValueT update       = weight2 * residual[s] / num_neighbors;

          for (int offset = 0; offset < num_neighbors; ++offset) {
              VertexT d = graph.GetEdgeDest(graph.GetNeighborListOffset(s) + offset);
//               visited.insert(d);

              #pragma omp atomic
              residual_prime[d] += update;

          }

//          #pragma omp atomic
//          tot_neighbors += num_neighbors;
      }
      res_update_timer.Stop();

      frontier_gen_timer.Start();
      
      // Generate The New Frontier
      frontier_size      = 0;

// #pragma omp declare reduction (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

// #pragma omp parallel for // reduction(merge: vec_frontier)
      for (int v = 0; v < nodes; v++) {
          
          // copy the update residuals
          residual[v] = residual_prime[v];

          SizeT num_neighbors = graph.GetNeighborListLength(v);
          
          if( num_neighbors && (residual[v] >= (num_neighbors * epsilon)))
          {
// #pragma omp critical
              {
                  frontier[frontier_size] = v;
                  frontier_size++;

//                  if( iter < 2 ){
//                      printf("%d, ", v);
//                  }
              }
              
          }
      }
//      if( iter < 2 ){
//          printf("\n");
//      }
      
      frontier_gen_timer.Stop();      

//      printf("\tIteration: %d Frontier Size: %d Visited Neighbors: %ld\n", iter, frontier_size, tot_neighbors);
//       printf("\tIteration: %d Frontier Size: %d Visited Neighbors: %ld Unique: %ld\n", iter, frontier_size, tot_neighbors, visited.size());
//       visited.clear();
      
      frontier_gen += frontier_gen_timer.ElapsedMillis();
      res_update   += res_update_timer.ElapsedMillis();
      pr_update    += pr_update_timer.ElapsedMillis();

      iter++;
  }

  printf("\n\n\n");

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  printf("pr_update: %lf res_update: %lf frontier_gen: %lf total: %lf\n", pr_update, res_update, frontier_gen, elapsed);

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
	util::PrintMsg("FAIL: [" + std::to_string(i) + "] " + " " + 
			std::to_string(h_values[i]) + " != " + 
			std::to_string(ref_values[i]), !quiet);
      }
    }
  }

  util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);

  return num_errors;
}

}  // namespace pr_nibble_shun
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
