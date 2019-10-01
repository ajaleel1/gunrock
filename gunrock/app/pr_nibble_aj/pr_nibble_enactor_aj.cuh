// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * Template_enactor.cuh
 *
 * @brief pr_nibble Problem Enactor
 */

#pragma once

#include <iostream>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <gunrock/app/pr_nibble_aj/pr_nibble_problem_aj.cuh>

namespace gunrock {
namespace app {
namespace pr_nibble_aj {

/**
 * @brief Speciflying parameters for pr_nibble Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of pr_nibble iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct PRNibbleIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  PRNibbleIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of pr_nibble, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // --
    // Alias variables

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];

    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];

    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;

    // problem specific data alias
    auto &pageRank = data_slice.pageRank;
    auto &residual = data_slice.residual;
    auto &residual_prime = data_slice.residual_prime;

    auto &alpha = data_slice.alpha;
    auto &epsilon = data_slice.eps;
    auto &max_iter = data_slice.max_iter;

    auto &src_node = data_slice.src;
    auto &src_neib = data_slice.src_neib;
    auto &num_ref_nodes = data_slice.num_ref_nodes;

    auto weight1 = data_slice.weight1;
    auto weight2 = data_slice.weight2;
	
    util::Array1D<SizeT, VertexT> *null_frontier = NULL;
    auto complete_graph = null_frontier;

    // --
    // Define operations

    // compute operation
    auto compute_op = [graph, weight1, pageRank, residual, residual_prime ] __host__
                      __device__(VertexT * v, const SizeT &i) {

        auto v_idx = v[i];

        //ValueT update_val = weight1 * Load<cub::LOAD_CG>(residual+v_idx); 
        auto update_val = weight1 * residual[v_idx]; 
        atomicAdd(pageRank+v_idx, update_val);
        
        residual_prime[v_idx] = 0;

//         printf("-- Compute -- Vertex: %d Update Val: %lf\n", v_idx, update_val);
    };
    

    // printf("Doing Compute -- Frontier Size: %d\n", frontier.queue_length);
    GUARD_CU(frontier.V_Q()->ForAll(compute_op, frontier.queue_length,
                                    util::DEVICE, oprtr_parameters.stream));


    // advance operation
    auto advance_op =
        [graph, weight2, residual, residual_prime] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {

        auto num_neighbors = graph.CsrT::GetNeighborListLength(src);
        // auto update_val  = weight2 * Load<cub::LOAD_CG>(residual+src) / num_neighbors;
        auto update_val  = weight2 * residual[src] / num_neighbors;
        atomicAdd(residual_prime + dest, update_val);

        // printf("-- Advance -- Src: %d Dest: %d Num Neighbors: %d Update Val: %lf\n", src, dest, num_neighbors, update_val);
	return true;
    };

    // filter operation
    auto filter_op = [graph, epsilon, residual, residual_prime] __host__ __device__(
                         const VertexT &src, VertexT &dest,
                         const SizeT &edge_id, const VertexT &input_item,
                         const SizeT &input_pos,
                         SizeT &output_pos) -> bool { 

        auto num_neighbors = graph.CsrT::GetNeighborListLength(dest);
        residual[dest] = residual_prime[dest];
        double myepsilon = 1e-9;

        if(num_neighbors && ((double)residual[dest] >= (double)(num_neighbors * myepsilon))) {
            return true;
        }

        // printf("-- Filter -- Src: %d Dest: %d input_pos: %d output_pos: %d Num Neighbors: %d residual: %lf epsilon: %lf Keep: %d\n", src, dest, input_pos, output_pos, num_neighbors, residual[dest], myepsilon, 0);
        
        return false; 
    };


    // printf("Calling Advance/Filter (mode: %s) -- Frontier Size: %d\n", oprtr_parameters.advance_mode.c_str(), frontier.queue_length);

    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
        advance_op, filter_op));

    // printf("\tNew Frontier Size: %d\n", frontier.queue_length);

      // frontier.queue_reset = true;
      // frontier.queue_length = graph.nodes;

      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          filter_op));

      GUARD_CU(frontier.work_progress.GetQueueLength(
        frontier.queue_index, frontier.queue_length, false,
        oprtr_parameters.stream, true));
    return retval;
  }


  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each
   * transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each
   * transmition item, typed ValueT
   * @param  received_length The numver of transmition items received
   * @param[in] peer_ which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // TODO: add problem specific data alias here, e.g.:
    // auto         &distances          =   data_slice.distances;

    auto expand_op = [
                         // TODO: pass data used by the lambda, e.g.:
                         // distances
    ] __host__ __device__(VertexT & key, const SizeT &in_pos,
                          VertexT *vertex_associate_ins,
                          ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distances + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of PRNibbleIterationLoop

/**
 * @brief Template enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT, typename _Problem::GraphT::VertexT,
          typename _Problem::GraphT::ValueT, ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::GraphT GraphT;
  typedef typename GraphT::VertexT LabelT;
  typedef typename GraphT::ValueT ValueT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef PRNibbleIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief pr_nibble constructor
   */
  Enactor() : BaseEnactor("pr_nibble"), problem(NULL) {
    // <OPEN> change according to algorithmic needs
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
    // </OPEN>
  }

  /**
   * @brief pr_nibble destructor
   */
  virtual ~Enactor() { /*Release();*/
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Release(target));
    delete[] iterations;
    iterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
   * @brief Initialize the problem.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(
        problem, Enactor_None,
        // <OPEN> change to how many frontier queues, and their types
        2, NULL,
        // </OPEN>
        target, false));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));
    }

    iterations = new IterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of pr_nibble, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        // <OPEN> change to how many {VertexT, ValueT} data need to communicate
        //       per element in the inter-GPU sub-frontiers
        0, 1,
        // </OPEN>
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
...
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(
      // <DONE> problem specific data if necessary, eg
      VertexT src, VertexT src_neib,
      // </DONE>
      util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));


    // printf("Resetting Enactor: num_gpus = %d Adding Src: %d Src Neighb: %d\n", this->num_gpus, src, src_neib);

    // <DONE> Initialize frontiers according to the algorithm:
    // In this case, we add a `src` to the frontier
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForAll(
                [src] __host__ __device__(VertexT * v, const SizeT &i) {
                  v[i] = src;
                },
                1, target, 0));
          }
        }
      } else {
        this->thread_slices[gpu].init_size = 0;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          this->enactor_slices[gpu * this->num_gpus + peer_]
              .frontier.queue_length = 0;
        }
      }
    }
    // </DONE>

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a pr_nibble computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(
      // <TODO> problem specific data if necessary, eg
      // VertexT src = 0
      // </TODO>
  ) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU Template Done.", this->flag & Debug);
    return retval;
  }
};

}  // namespace pr_nibble_aj
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
