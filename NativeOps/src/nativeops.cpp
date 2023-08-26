#include "utils.hpp"

thread_local char *torch_last_err = nullptr;

EXPORT_API(const char *)
llm_sharp_check_last_err()
{
  char *tmp = torch_last_err;
  torch_last_err = nullptr;
  return tmp;
}

EXPORT_API(Tensor)
llm_sharp_hello(const Tensor tensor)
{
  CATCH_TENSOR(tensor->add(1));
}
