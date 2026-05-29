#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "memtier_uvm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MemTier PyTorch UVM file/data prototype";

  m.def("allocate_managed", &allocate_managed,
        "Allocate CUDA managed memory and return its address as an integer",
        pybind11::arg("bytes"), pybind11::arg("device") = 0);
  m.def("free_managed", &free_managed, "Free CUDA managed memory",
        pybind11::arg("ptr"));
  m.def("advise_preferred_cpu", &advise_preferred_cpu,
        "Set CUDA UVM preferred location to CPU", pybind11::arg("ptr"),
        pybind11::arg("bytes"));
  m.def("advise_preferred_gpu", &advise_preferred_gpu,
        "Set CUDA UVM preferred location to a GPU", pybind11::arg("ptr"),
        pybind11::arg("bytes"), pybind11::arg("device") = 0);
  m.def("advise_accessed_by_gpu", &advise_accessed_by_gpu,
        "Mark CUDA UVM memory as accessed by a GPU", pybind11::arg("ptr"),
        pybind11::arg("bytes"), pybind11::arg("device") = 0);
  m.def("prefetch_to_gpu", &prefetch_to_gpu,
        "Prefetch CUDA UVM memory to a GPU on the current stream",
        pybind11::arg("ptr"), pybind11::arg("bytes"),
        pybind11::arg("device") = 0);
  m.def("prefetch_to_cpu", &prefetch_to_cpu,
        "Prefetch CUDA UVM memory to CPU on the current stream",
        pybind11::arg("ptr"), pybind11::arg("bytes"));
  m.def("load_file_to_managed", &load_file_to_managed,
        "pread a float32 file slice into CUDA managed memory and wrap it as a CPU tensor",
        pybind11::arg("path"), pybind11::arg("offset"),
        pybind11::arg("rows"), pybind11::arg("cols"),
        pybind11::arg("device") = 0);
  m.def("uvm_tensor_from_file", &uvm_tensor_from_file,
        "Create a UVM-backed tensor from a file slice and apply an optional UVM policy",
        pybind11::arg("path"), pybind11::arg("rows"), pybind11::arg("cols"),
        pybind11::arg("offset") = 0, pybind11::arg("device") = 0,
        pybind11::arg("prefetch_policy") = "none");
  m.def("touch_gpu", &touch_gpu,
        "Launch a CUDA kernel that reads the managed tensor to trigger UVM migration",
        pybind11::arg("tensor"));
  m.def("get_stats", &get_stats, "Return MemTier prototype counters");
  m.def("reset_stats", &reset_stats, "Reset MemTier prototype counters");
}
