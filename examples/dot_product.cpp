#include <cstdlib>
#include <cstdio>

#include "../src/Vecky.hpp"

using namespace Vecky;

int main(int argc, char* argv[]){
  if (argc < 4){
    printf("Need at least 3 numbers\n");
  }
  VecN<float, 3> v1;
  for (int i = 0; i < 3; i++){
    v1[i] = atof(argv[i + 1]);
  }

  VecN<float, 3> v2;
  for (int i = 0; i < 3; i++){
    v2[i] = atof(argv[i + 4]);
  }
  VecN<float, 3> v3 = v1 * -v2 +  sin(v1 + v2 * 5.0f + cos(v1 * 3.f));
  
  printf("%f\n",  v3[0]);
  return 0;
}
