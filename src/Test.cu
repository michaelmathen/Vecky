#include "Vecky.hpp"
using namespace Vecky;

int main(int argc, char* argv[]){
  VecN<float, 3> vec1;
  VecN<float, 3> vec2;
  VecN<float, 3> vec3;
  return (vec1 + 3.f * vec2 + vec3).get<0>();
}
