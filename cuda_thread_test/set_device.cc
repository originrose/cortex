#include "cuda_errors.hpp"
#include <iostream>

using namespace think;

int main (int c, char** v)
{
  int retval = cuInit(0);
  cout << "Error from cuInit: " << error_name(retval) << endl;
  return 0;
}
