#include <queue>

#include "tensorflow/core/data/service/easl/cache_utils.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_put_op.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_get_op.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace cache_utils {

Status DoBFS(NodeDef* sink_node, GraphDef& graph_def, string prefix) {
  absl::flat_hash_set<std::string> visited;
  std::queue<NodeDef*> bfs_queue;
  bfs_queue.push(sink_node);

  VLOG(1) << "(" << prefix << ") BFS @ current_node: " 
          << "Root --> " << sink_node->op();
  
  while (!bfs_queue.empty()) {
    NodeDef* current_node = bfs_queue.front();
    bfs_queue.pop();
    visited.insert(current_node->name());

    // Iterate throught the neighbors
    for (int i = 0; i < current_node->input_size(); ++i) {
      if (!visited.contains(current_node->input(i))) {
        int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(
          current_node->input(i), graph_def);
        NodeDef* neighbor_node = graph_def.mutable_node(idx);
        bfs_queue.push(neighbor_node);

        VLOG(1) << "(" << prefix << ") BFS @ current_node: " 
                << SummarizeNodeDef(*current_node) << " --> " 
                << SummarizeNodeDef(*neighbor_node);
      }
    }
  }

  return Status::OK();
}

std::string DatasetPutKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_put");
}

std::string DatasetGetKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_get");
}

Status DatasetKey(const ::tensorflow::data::easl::CacheState& cache_state,
                  const int64 dataset_id,
                  const uint64 fingerprint,
                  const std::string& worker_address,
                  const int64 task_id,
                  std::string& dataset_key){
  if(cache_state.IsDatasetCached(fingerprint, worker_address)){
    dataset_key =
        absl::StrCat("id_", dataset_id, "_fp_", fingerprint, "_get");
    VLOG(0) << "Use get dataset for fingerprint " << fingerprint
                 << " at worker " << worker_address;
    return Status::OK();
  }

  int64 caching_task;
  TF_RETURN_IF_ERROR(cache_state.GetCachingTaskId(
      fingerprint, worker_address, caching_task));
  if(caching_task == task_id) {
    dataset_key =
        absl::StrCat("id_", dataset_id, "_fp_", fingerprint, "_put");
    VLOG(0) << "Use put dataset for fingerprint " << fingerprint
                 << " at worker " << worker_address;
    return Status::OK();
  }

  dataset_key =
      absl::StrCat("id_", dataset_id, "_fp_", fingerprint);
  VLOG(0) << "Use standard dataset for fingerprint " << fingerprint
               << " at worker " << worker_address;
  return Status::OK();
}

Status AddPutOperator(const DatasetDef& dataset, DatasetDef& updated_dataset) {
  updated_dataset = dataset;
  return Status::OK();
  
  VLOG(1) << "(AddPutOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset; 

  // Initialize the optimizer  
  tensorflow::grappler::easl::AddPutOp optimizer;
  optimizer.Init(nullptr);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  DoBFS(sink, *graph_def, "AddPutOperator");
  
  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  DoBFS(sink, *graph_def, "AfterAddPutOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(1) << "(AddPutOperator) At the end of the method";
  
  return Status::OK();
}

Status AddGetOperator(const DatasetDef& dataset, DatasetDef& updated_dataset){
  updated_dataset = dataset;
  return Status::OK();

  VLOG(1) << "(AddGetOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset; 

  // Initialize the optimizer  
  tensorflow::grappler::easl::AddGetOp optimizer;
  optimizer.Init(nullptr);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  DoBFS(sink, *graph_def, "AddGetOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  DoBFS(sink, *graph_def, "AfterAddGetOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(1) << "(AddGetOperator) At the end of the method";

  return Status::OK();
}



} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow
