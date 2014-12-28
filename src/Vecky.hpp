#include <math.h>

#ifndef MM_VECKY
#define MM_VECKY

#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __global__ 
#endif

namespace Vecky {

#define OPERATION_STRUCT(OP_NAME, OPERATION) \
struct OP_NAME {  \
  template <typename T, unsigned int N, unsigned int I>   \
  __host__ __device__ inline static T apply(T const& v1, T const& v2) {	\
    return v1 OPERATION v2;	   \
  }  \
}; 

OPERATION_STRUCT(OprAdd, +)
OPERATION_STRUCT(OprMult, *)
OPERATION_STRUCT(OprDiv, /)
OPERATION_STRUCT(OprDiff, -)
OPERATION_STRUCT(OprLess, <)
OPERATION_STRUCT(OprGreater, >)
OPERATION_STRUCT(OprEqual, ==)
OPERATION_STRUCT(OprNEq, ==)
OPERATION_STRUCT(OprGEq, >=)
OPERATION_STRUCT(OprLEq, <=)

template <typename E1, typename OP, typename E2, typename T, unsigned int N>
struct ExprNode {
  E1 const& _e1;
  E2 const& _e2;
public:
  __host__ __device__ inline ExprNode(E1 const& e1, E2 const& e2)
    : _e1(e1), _e2(e2)
  {}
  
  template<unsigned int I>
  __host__ __device__ inline T get() const {
    return OP::template apply<T, N, I>(_e1.template get<I>(), _e2.template get<I>());
  }
};

template <typename T>
struct ScalarNode {
  T const& _v;
public:

  __host__ __device__ inline ScalarNode(T const& v)
    : _v(v)
  {}
  template<unsigned int I>
  __host__ __device__ inline T get() const {
    return _v;
  }
};

template<typename T, typename Opr, unsigned int N>
struct Binary_Map {
  void apply(T& dst, T const& src){
    dst[N] = Opr::template apply<T, 0, 0>(dst.template get<N>(), src.template get<N>());
    Binary_Map<T, Opr, N - 1>::apply(dst, src);
  }
};

template<typename T, typename Opr>
struct Binary_Map <T, Opr, 0> {
  void apply(T& dst, T const& src){
    dst[0] = Opr::template apply<T, 0, 0>(dst.template get<0>(), src.template get<0>());
  }
};

template<typename E1, typename OP, typename E2, typename T, unsigned int N, unsigned int i>
struct Eval {
  __host__ __device__ inline void
  apply(T* _data, ExprNode<E1, OP, E2, T, N> const& expr){
    _data[i] = expr.template get<i>();
    Eval<E1, OP, E2, T, N, i + 1>::apply(_data, expr);
  }
};

template<typename E1, typename OP, typename E2, typename T, unsigned int N>
struct Eval<E1, OP, E2, T, N, N> {
  __host__ __device__ inline void
  apply(T* _data, ExprNode<E1, OP, E2, T, N> const& expr){}
};

  
template <typename T, unsigned int N>
class VecN {
  T _data[N];
public:
  
  template<typename E1, typename OP, typename E2>
  __host__ __device__ inline
  VecN(ExprNode<E1, OP, E2, T, N> const& expr){
    Eval<E1, OP, E2, T, N, 0>::apply(&_data[0], expr);
  }

  __host__ __device__ inline
  VecN(){}

  __host__ __device__ inline T& operator[](unsigned int i){
    return _data[i];
  }

  __host__ __device__ inline T operator[](unsigned int i) const {
    return _data[i];
  }

  template<unsigned int i>
  __host__ __device__ inline T get() const {
    return _data[i];
  }

  __host__ __device__ inline VecN<T, N>&
  operator *= (T const& val){
    Binary_Map<T, OprMult, N>::apply(*this, ScalarNode<T>(val));
    return *this;
  }

  __host__ __device__ inline VecN<T, N>&
  operator *= (VecN<T, N> const& vec){
    Binary_Map<T, OprMult, N>::apply(*this, vec);
    return *this;
  }

  __host__ __device__ inline VecN<T, N>&
  operator /= (T const& val){
    Binary_Map<T, OprDiv, N>::apply(*this, ScalarNode<T>(val));
    return *this;
  }

  __host__ __device__ inline VecN<T, N>&
  operator /= (VecN<T, N> const& vec){
    Binary_Map<T, OprDiv, N>::apply(*this, vec);
    return *this;
  }

  
  __host__ __device__ inline VecN<T, N>&
  operator += (T const& val){
    Binary_Map<T, OprAdd, N>::apply(*this, ScalarNode<T>(val));
    return *this;
  }

  __host__ __device__ inline VecN<T, N>&
  operator -= (T const& val){
    Binary_Map<T, OprDiff, N>::apply(*this, ScalarNode<T>(val));
    return *this;
  }

  __host__ __device__ inline VecN<T, N>&
  operator += (VecN<T, N> const& vec){
    Binary_Map<T, OprAdd, N>::apply(*this, vec);
    return *this;
  }

  __host__ __device__ inline VecN<T, N>&
  operator -= (VecN<T, N> const& vec){
    Binary_Map<T, OprDiff, N>::apply(*this, vec);
    return *this;
  }

  
};


#define BINARY_FUNCTION_STRUCT(F_NAME, FUNCTION) \
struct F_NAME { \
  template <typename T, unsigned int N, unsigned int I> \
  __host__ __device__ inline static T \
  apply(T const& v1, T const& v2){		\
    return FUNCTION(v1, v2);	 		\
  } \
}; 

BINARY_FUNCTION_STRUCT(VecExponent, pow)

#define BINARY_VECTOR_OPERATION(OP_NAME, SYMBOL, RET_T)			\
template<typename E1, typename E2, typename E3, typename E4, typename PrevOp1, typename PrevOp2, typename T, unsigned int N> \
__host__ __device__ inline ExprNode<ExprNode<E1, PrevOp1, E2, T, N>, OP_NAME, ExprNode<E3, PrevOp2, E4, T, N>, RET_T, N> const \
operator SYMBOL (ExprNode<E1, PrevOp1, E2, T, N> const& expNode1, ExprNode<E2, PrevOp2, E3, T, N> const& expNode2){ \
  return ExprNode<ExprNode<E1, PrevOp1, E2, T, N>, OP_NAME, ExprNode<E3, PrevOp2, E4, T, N>, RET_T, N>(expNode1, expNode2); \
}   \
    \
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>  \
__host__ __device__ inline ExprNode<ExprNode<E1, PrevOp, E2, T, N>, OP_NAME, VecN<T, N>, RET_T, N> const   \
operator SYMBOL (ExprNode<E1, PrevOp, E2, T, N> const& expNode, VecN<T, N> const& vec){  \
  return ExprNode<ExprNode<E1, PrevOp, E2, T, N>, OP_NAME, VecN<T, N>, RET_T, N>(expNode, vec);  \
}  \
  \
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>  \
__host__ __device__ inline ExprNode<VecN<T, N>, OP_NAME, ExprNode<E1, PrevOp, E2, T, N>, RET_T, N> const  \
operator SYMBOL (VecN<T, N> const& vec, ExprNode<E1, PrevOp, E2, T, N> const& expNode){  \
  return ExprNode<VecN<T, N>, OP_NAME, ExprNode<E1, PrevOp, E2, T, N>, RET_T, N>(vec, expNode);  \
}  \
  \
template<typename T, unsigned int N>  \
__host__ __device__ inline ExprNode<VecN<T, N> , OP_NAME, VecN<T, N>, RET_T, N> const  \
operator SYMBOL (VecN<T, N> const& vec1, VecN<T, N> const& vec2){   \
  return ExprNode<VecN<T, N>, OP_NAME, VecN<T, N>, RET_T, N>(vec1, vec2);  \
}

BINARY_VECTOR_OPERATION(OprAdd, +, T)
BINARY_VECTOR_OPERATION(OprMult, *, T)
BINARY_VECTOR_OPERATION(OprDiv, /, T)
BINARY_VECTOR_OPERATION(OprDiff, -, T)

BINARY_VECTOR_OPERATION(OprLess, <, bool)
BINARY_VECTOR_OPERATION(OprGreater, >, bool)
BINARY_VECTOR_OPERATION(OprEqual, ==, bool)
BINARY_VECTOR_OPERATION(OprNEq, !=, bool)
BINARY_VECTOR_OPERATION(OprGEq, ==, bool)
BINARY_VECTOR_OPERATION(OprLEq, !=, bool)


#define SCALAR_VECTOR_OPERATION(OP_NAME, SYMBOL)   \
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>	\
__host__ __device__ inline  ExprNode<ExprNode<E1, PrevOp, E2, T, N>, OP_NAME, ScalarNode<T>, T, N> const \
operator SYMBOL (ExprNode<E1, PrevOp, E2, T, N> const& expNode, T const& val){	\
  return ExprNode<ExprNode<E1, PrevOp, E2, T, N>, OP_NAME, ScalarNode<T>, T, N>(expNode, ScalarNode<T>(val)); \
}    \
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>  \
__host__ __device__ inline ExprNode<ScalarNode<T>, OP_NAME, ExprNode<E1, PrevOp, E2, T, N>, T, N> const \
operator SYMBOL (T const& val, ExprNode<E1, PrevOp, E2, T, N> const& expNode){  \
  return ExprNode<ScalarNode<T>, OP_NAME, ExprNode<E1, PrevOp, E2, T, N>, T, N>(ScalarNode<T>(val), expNode); \
}  \
template<typename T, unsigned int N>			\
__host__ __device__ inline  ExprNode<VecN<T, N>, OP_NAME, ScalarNode<T>, T, N> const	\
operator SYMBOL (VecN<T, N> const& vec, T const& val){	\
  return ExprNode<VecN<T, N>, OP_NAME, ScalarNode<T>, T, N>(vec, ScalarNode<T>(val)); \
}    \
template<typename T, unsigned int N>			\
__host__ __device__ inline  ExprNode<ScalarNode<T>, OP_NAME, VecN<T, N>, T, N> const	\
operator SYMBOL (T const& val, VecN<T, N> const& vec){		\
  return ExprNode<ScalarNode<T>, OP_NAME, VecN<T, N>, T, N>(ScalarNode<T>(val), vec); \
} 

SCALAR_VECTOR_OPERATION(OprMult, *)
SCALAR_VECTOR_OPERATION(OprDiv, /)
}
/*
  Functions that operate on two elements at the same time.
 */

#endif
