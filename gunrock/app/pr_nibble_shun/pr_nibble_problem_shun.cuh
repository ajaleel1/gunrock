// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_nibble_problem.cuh
 *
 * @brief GPU Storage management Structure for pr_nibble Problem Data
 */

#pragma once

#include <iostream>
#include <math.h>
#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace pr_nibble_shun {

/**
 * @brief Speciflying parameters for pr_nibble Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(gunrock::app::UseParameters_problem(parameters));
  return retval;
}

/**
 * @brief Template Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // ----------------------------------------------------------------
  // Dataslice structure

  /**
   * @brief Data structure containing problem specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    util::Array1D<SizeT, ValueT> pageRank;           // Output values

    util::Array1D<SizeT, ValueT> residual;           // residual
    util::Array1D<SizeT, ValueT> residual_prime;     // residual_prime

    VertexT src;        // Node to start local PR from
    int num_ref_nodes;  // Number of source nodes (hardcoded to 1 for now)

    ValueT eps;    // Tolerance for convergence
    ValueT alpha;  // Parameterizes conductance/size of output cluster
    ValueT weight1;  // Parameterizes conductance/size of output cluster
    ValueT weight2;  // Parameterizes conductance/size of output cluster
    bool   dense;    // run in dense mode

    int max_iter;  // Maximum number of iterations

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      pageRank.SetName("pageRank");
      residual.SetName("residual");
      residual_prime.SetName("residual_prime");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      GUARD_CU(pageRank.Release(target));
      GUARD_CU(residual.Release(target));
      GUARD_CU(residual_prime.Release(target));

      GUARD_CU(BaseDataSlice::Release(target));
      return retval;
    }

    /**
     * @brief initializing sssp-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * @param[in] _eps        Convergence criteria
     * @param[in] _alpha
     * @param[in] _rho
     * @param[in] _max_iter   Max number of iterations

     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx,
                     util::Location target, ProblemFlag flag, ValueT _eps, bool _dense,
                     ValueT _alpha, int _max_iter) {
      cudaError_t retval = cudaSuccess;
      
      dense = _dense;
      eps = _eps;
      alpha = _alpha;
      max_iter = _max_iter;

      weight1 = (2*alpha) / (1+alpha);
      weight2 = (1-alpha) / (1+alpha);

      printf("DataSlice Setting Weight1: %lf Weight2 %lf Epsilon: %lf\n", this->weight1, this->weight2, this->eps);

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      GUARD_CU(pageRank.Allocate(sub_graph.nodes, target));
      GUARD_CU(residual.Allocate(sub_graph.nodes, target));
      GUARD_CU(residual_prime.Allocate(sub_graph.nodes, target));

      if (target & util::DEVICE) {
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
      }
      return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * @param[in] _src             Source node
     * @param[in] _num_ref_nodes   Number of source nodes (HARDCODED to 1
     * elsewhere) \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(VertexT _src, 
                      int _num_ref_nodes,
                      util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;

      src = _src;
      num_ref_nodes = _num_ref_nodes;

      // Ensure data are allocated
      GUARD_CU(pageRank.EnsureSize_(nodes, target));
      GUARD_CU(residual.EnsureSize_(nodes, target));
      GUARD_CU(residual_prime.EnsureSize_(nodes, target));

      // Reset data
      GUARD_CU(pageRank.ForEach([] __host__ __device__(ValueT & x) { x = (ValueT)0; },
                                nodes, target, this->stream));

      GUARD_CU(residual.ForEach([] __host__ __device__(ValueT & x) { x = (ValueT)0; },
                                nodes, target, this->stream));

      GUARD_CU(residual_prime.ForEach([] __host__ __device__(ValueT & x) { x = (ValueT)0; },
                                      nodes, target, this->stream));

      GUARD_CU(residual.ForAll([_src] __host__ __device__(ValueT * x, const SizeT &pos) { 
			      x[_src] = (ValueT)1; }, 1, target, this->stream));
      
      GUARD_CU(residual_prime.ForAll([_src] __host__ __device__(ValueT * x, const SizeT &pos) { 
			      x[_src] = (ValueT)1; }, 1, target, this->stream));
      return retval;
    }
  };  // DataSlice

  // Problem attributes
  util::Array1D<SizeT, DataSlice> *data_slices;

  int max_iter;
  int dense;
  ValueT eps;
  ValueT alpha;
  ValueT weight1;  // Parameterizes conductance/size of output cluster
  ValueT weight2;  // Parameterizes conductance/size of output cluster

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief pr_nibble default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {
    // Load command line parameters
    max_iter = _parameters.Get<int>("max-iter");
    eps = _parameters.Get<ValueT>("eps");
    alpha = _parameters.Get<ValueT>("alpha");
    dense = _parameters.Get<int>("dense");
  }

  /**
   * @brief pr_nibble default destructor
   */
  virtual ~Problem() { Release(); }

  /*
   * @brief Releasing allocated memory space
   * @param[in] target      The location to release memory from
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    if (data_slices == NULL) return retval;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i].Release(target));

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * @brief Copy result distancess computed on GPUs back to host-side arrays.
   * @param[in] h_values      Host array for PR values
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(ValueT *h_values, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    ValueT *h_pageRank = new ValueT[nodes];
    ValueT *h_residual = new ValueT[nodes];
    ValueT *h_residual_prime = new ValueT[nodes];

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(data_slice.pageRank.SetPointer(h_values, nodes, util::HOST));
        GUARD_CU(data_slice.pageRank.Move(util::DEVICE, util::HOST));

        GUARD_CU(data_slice.residual.SetPointer(h_residual, nodes, util::HOST));
        GUARD_CU(data_slice.residual.Move(util::DEVICE, util::HOST));

        GUARD_CU(data_slice.residual_prime.SetPointer(h_residual_prime, nodes, util::HOST));
        GUARD_CU(data_slice.residual_prime.Move(util::DEVICE, util::HOST));
      } else if (target == util::HOST) {

        GUARD_CU(data_slice.pageRank.ForEach(
            h_pageRank,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));

        GUARD_CU(data_slice.residual.ForEach(
            h_residual,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));

        GUARD_CU(data_slice.residual_prime.ForEach(
            h_residual_prime,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));
      }
    } else {  // num_gpus != 1

      // ============ INCOMPLETE TEMPLATE - MULTIGPU ============

      // // TODO: extract the results from multiple GPUs, e.g.:
      // // util::Array1D<SizeT, ValueT *> th_distances;
      // // th_distances.SetName("bfs::Problem::Extract::th_distances");
      // // GUARD_CU(th_distances.Allocate(this->num_gpus, util::HOST));

      // for (int gpu = 0; gpu < this->num_gpus; gpu++)
      // {
      //     auto &data_slice = data_slices[gpu][0];
      //     if (target == util::DEVICE)
      //     {
      //         GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      //         // GUARD_CU(data_slice.distances.Move(util::DEVICE,
      //         util::HOST));
      //     }
      //     // th_distances[gpu] = data_slice.distances.GetPointer(util::HOST);
      // } //end for(gpu)

      // for (VertexT v = 0; v < nodes; v++)
      // {
      //     int gpu = this -> org_graph -> GpT::partition_table[v];
      //     VertexT v_ = v;
      //     if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
      //         v_ = this -> org_graph -> GpT::convertion_table[v];

      //     // h_distances[v] = th_distances[gpu][v_];
      // }

      // // GUARD_CU(th_distances.Release());
    }

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that SSSP processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    ValueT num_edges = (ValueT)graph.edges / 2.0;
    ValueT log_num_edges = log2(num_edges);

//     // alpha
//     this->alpha = 0.15;
//     this->eps   = (double)1e-6;

    this->weight1 = (2*this->alpha) / (1+this->alpha);
    this->weight2 = (1-this->alpha) / (1+this->alpha);

    printf("Init Problem Setting Weight1: %lf Weight2 %lf Epsilon: %lf\n", this->weight1, this->weight2, this->eps);

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(
          this->sub_graphs[gpu], this->num_gpus, this->gpu_idx[gpu], target,
          this->flag, this->eps, this->dense, this->alpha, this->max_iter));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src       Source vertex
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(VertexT src, 
                    util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    int num_ref_nodes = 1;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(src, num_ref_nodes, target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    int gpu;
    VertexT src_;
    if (this->num_gpus <= 1) {
      gpu = 0;
      src_ = src;
    } else {
      // TODO -- MULTIGPU
    }

    if (target & util::DEVICE) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }
};

}  // namespace pr_nibble_shun
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
