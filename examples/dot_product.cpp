#include <cstdlib>
#include <cstdio>

#include "../src/Vecky.hpp"

using namespace Vecky;

int main(int argc, char* argv[]){
  if (argc < 4){
    printf("Need at least 4 numbers\n");
  }
  VecN<float, 3> v1;
  for (int i = 0; i < 3; i++){
    v1[i] = atof(argv[i + 1]);
  }
  
  float v = 5.f;
  VecN<float, 3> v3 = dot(v1 * v * v1 * v1, v1);

  printf("%f\n",  v3[0]);
  return 0;
}
