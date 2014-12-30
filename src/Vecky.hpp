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

  
template <typename OP, typename E2, typename T, unsigned int N>
struct ExprNode <T, OP, E2, T, N> {
  T const& _e1;
  E2 const& _e2;
public:
  __host__ __device__ inline ExprNode(T const& e1, E2 const& e2)
    : _e1(e1), _e2(e2)
  {}
  
  template<unsigned int I>
  __host__ __device__ inline T get() const {
    return OP::template apply<T, N, I>(_e1, _e2.template get<I>());
  }
};

template <typename E1, typename OP, typename T, unsigned int N>
struct ExprNode <E1, OP, T, T, N> {
  E1 const& _e1;
  T const& _e2;
public:
  __host__ __device__ inline ExprNode(E1 const& e1, T const& e2)
    : _e1(e1), _e2(e2)
  {}
  
  template<unsigned int I>
  __host__ __device__ inline T get() const {
    return OP::template apply<T, N, I>(_e1.template get<I>(), _e2);
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
  __host__ __device__ static void 
  apply(T& dst, T const& src){
    dst[N] = Opr::template apply<T, 0, 0>(dst.template get<N>(), src.template get<N>());
    Binary_Map<T, Opr, N - 1>::apply(dst, src);
  }
};

template<typename T, typename Opr>
struct Binary_Map <T, Opr, 0> {
  __host__ __device__ inline static void
  apply(T& dst, T const& src){
    dst[0] = Opr::template apply<T, 0, 0>(dst.template get<0>(), src.template get<0>());
  }
};

template<typename E1, typename OP, typename E2, typename T, unsigned int N, unsigned int i>
struct Eval {
  __host__ __device__ inline static void
  apply(T* _data, ExprNode<E1, OP, E2, T, N> const& expr){
    _data[i] = expr.template get<i>();
    Eval<E1, OP, E2, T, N, i + 1>::apply(_data, expr);
  }
};

template<typename E1, typename OP, typename E2, typename T, unsigned int N>
struct Eval<E1, OP, E2, T, N, N> {
  __host__ __device__ inline static void
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

  __host__ __device__ inline
  VecN(T const& val){
    for (unsigned int i = 0; i < N; i++){
      _data[i] = val;
    }
  }

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
operator SYMBOL (ExprNode<E1, PrevOp1, E2, T, N> const& expNode1, ExprNode<E3, PrevOp2, E4, T, N> const& expNode2){ \
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
__host__ __device__ inline  ExprNode<ExprNode<E1, PrevOp, E2, T, N>, OP_NAME, T, T, N> const \
operator SYMBOL (ExprNode<E1, PrevOp, E2, T, N> const& expNode, T const& val){	\
  return ExprNode<ExprNode<E1, PrevOp, E2, T, N>, OP_NAME, T, T, N>(expNode, val); \
}    \
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>  \
__host__ __device__ inline ExprNode<T, OP_NAME, ExprNode<E1, PrevOp, E2, T, N>, T, N> const \
operator SYMBOL (T const& val, ExprNode<E1, PrevOp, E2, T, N> const& expNode){  \
  return ExprNode<T, OP_NAME, ExprNode<E1, PrevOp, E2, T, N>, T, N>(val, expNode); \
}  \
template<typename T, unsigned int N>			\
__host__ __device__ inline  ExprNode<VecN<T, N>, OP_NAME, T, T, N> const	\
operator SYMBOL (VecN<T, N> const& vec, T const& val){	\
  return ExprNode<VecN<T, N>, OP_NAME, T, T, N>(vec, val); \
}    \
template<typename T, unsigned int N>			\
__host__ __device__ inline  ExprNode<T, OP_NAME, VecN<T, N>, T, N> const	\
operator SYMBOL (T const& val, VecN<T, N> const& vec){		\
  return ExprNode<T, OP_NAME, VecN<T, N>, T, N>(val, vec); \
} 

SCALAR_VECTOR_OPERATION(OprMult, *)
SCALAR_VECTOR_OPERATION(OprDiv, /)

#define BINARY_VECTOR_FUNCTION(F_OP, F_NAME, RET_T)			\
template<typename E1, typename E2, typename E3, typename E4, typename PrevOp1, typename PrevOp2, typename T, unsigned int N> \
__host__ __device__ inline ExprNode<ExprNode<E1, PrevOp1, E2, T, N>, F_OP, ExprNode<E3, PrevOp2, E4, T, N>, RET_T, N> const \
F_NAME(ExprNode<E1, PrevOp1, E2, T, N> const& expNode1, ExprNode<E3, PrevOp2, E4, T, N> const& expNode2){ \
  return ExprNode<ExprNode<E1, PrevOp1, E2, T, N>, F_OP, ExprNode<E3, PrevOp2, E4, T, N>, RET_T, N>(expNode1, expNode2); \
}   \
    \
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>  \
__host__ __device__ inline ExprNode<ExprNode<E1, PrevOp, E2, T, N>, F_OP, VecN<T, N>, RET_T, N> const   \
F_NAME (ExprNode<E1, PrevOp, E2, T, N> const& expNode, VecN<T, N> const& vec){  \
  return ExprNode<ExprNode<E1, PrevOp, E2, T, N>, F_OP, VecN<T, N>, RET_T, N>(expNode, vec);  \
}  \
  \
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>  \
__host__ __device__ inline ExprNode<VecN<T, N>, F_OP, ExprNode<E1, PrevOp, E2, T, N>, RET_T, N> const  \
F_NAME (VecN<T, N> const& vec, ExprNode<E1, PrevOp, E2, T, N> const& expNode){ \
  return ExprNode<VecN<T, N>, F_OP, ExprNode<E1, PrevOp, E2, T, N>, RET_T, N>(vec, expNode);  \
}  \
  \
template<typename T, unsigned int N>  \
__host__ __device__ inline ExprNode<VecN<T, N>, F_OP, VecN<T, N>, RET_T, N> const  \
F_NAME (VecN<T, N> const& vec1, VecN<T, N> const& vec2){   \
  return ExprNode<VecN<T, N>, F_OP, VecN<T, N>, RET_T, N>(vec1, vec2);  \
}

struct CrossProd {  
  template <typename T, unsigned int N, unsigned int I>
  __host__ __device__ inline static T apply(T const& v1, T const& v2) {
    return v1[(I + 1) % 3] * v2[(I + 2) % 3] - v1[(I + 2) % 3] * v2[(I + 1) % 3];
  }
};

template<typename E1, typename E2, typename E3, typename E4, typename PrevOp1, typename PrevOp2, typename T, unsigned int 3> 
__host__ __device__ inline Expr3ode<Expr3ode<E1, PrevOp1, E2, T, 3>, CrossProd, Expr3ode<E3, PrevOp2, E4, T, 3>, T, 3> const 
cross(Expr3ode<E1, PrevOp1, E2, T, 3> const& exp3ode1, Expr3ode<E3, PrevOp2, E4, T, 3> const& exp3ode2){ 
  return Expr3ode<Expr3ode<E1, PrevOp1, E2, T, 3>, CrossProd, Expr3ode<E3, PrevOp2, E4, T, 3>, T, 3>(exp3ode1, exp3ode2); 
}   
    
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int 3>  
__host__ __device__ inline Expr3ode<Expr3ode<E1, PrevOp, E2, T, 3>, CrossProd, Vec3<T, 3>, T, 3> const   
cross (Expr3ode<E1, PrevOp, E2, T, 3> const& exp3ode, Vec3<T, 3> const& vec){  
  return Expr3ode<Expr3ode<E1, PrevOp, E2, T, 3>, CrossProd, Vec3<T, 3>, T, 3>(exp3ode, vec);  
}  
  
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int 3>  
__host__ __device__ inline Expr3ode<Vec3<T, 3>, CrossProd, Expr3ode<E1, PrevOp, E2, T, 3>, T, 3> const  
cross (Vec3<T, 3> const& vec, Expr3ode<E1, PrevOp, E2, T, 3> const& exp3ode){ 
  return Expr3ode<Vec3<T, 3>, CrossProd, Expr3ode<E1, PrevOp, E2, T, 3>, T, 3>(vec, exp3ode);  
}  
  
template<typename T, unsigned int 3>  
__host__ __device__ inline Expr3ode<Vec3<T, 3>, CrossProd, Vec3<T, 3>, T, 3> const  
cross (Vec3<T, 3> const& vec1, Vec3<T, 3> const& vec2){   
  return Expr3ode<Vec3<T, 3>, CrossProd, Vec3<T, 3>, T, 3>(vec1, vec2);  
}

template<typename E1, typename E2, typename T, unsigned int from>
struct DOT {
  __host__ __device__ inline static T
  apply(E1 const& e1, E2 const& e2) {
    return e1.template get<from>() * e2.template get<from>() + DOT<E1, E2, T, from-1>::apply(e1, e2);
  }
};

template<typename E1, typename E2, typename T>
struct DOT<E1, E2, T, 0> {
  __host__ __device__ inline static T
  apply(E1 const& e1, E2 const& e2) {
    return e1.template get<0>() * e2.template get<0>();
  }
};

template<typename E1, typename E2, typename E3, typename E4, typename PrevOp1, typename PrevOp2, typename T, unsigned int N> 
__host__ __device__ inline T const 
dot (ExprNode<E1, PrevOp1, E2, T, N> const& expNode1, ExprNode<E3, PrevOp2, E4, T, N> const& expNode2){ 
  return DOT<ExprNode<E1, PrevOp1, E2, T, N>, ExprNode<E3, PrevOp2, E4, T, N>, T, N-1>::apply(expNode1, expNode2); 
}   
    
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>  
__host__ __device__ inline T const   
dot (ExprNode<E1, PrevOp, E2, T, N> const& expNode, VecN<T, N> const& vec){  
  return DOT<ExprNode<E1, PrevOp, E2, T, N>, VecN<T, N>, T, N-1>::apply(expNode, vec);  
}  
  
template<typename E1, typename E2, typename PrevOp, typename T, unsigned int N>  
__host__ __device__ inline T const  
dot (VecN<T, N> const& vec, ExprNode<E1, PrevOp, E2, T, N> const& expNode){  
  return DOT<VecN<T, N>, ExprNode<E1, PrevOp, E2, T, N>, T, N-1>::apply(vec, expNode);  
}  
  
template<typename T, unsigned int N>  
__host__ __device__ inline T const  
dot (VecN<T, N> const& vec1, VecN<T, N> const& vec2){   
  return DOT<VecN<T, N>, VecN<T, N>, T, N-1>::apply(vec1, vec2);
}

typedef VecN<Real_t, 4> Vec4;
typedef VecN<Real_t, 3> Vec3;
typedef VecN<Real_t, 2> Vec2;

}
#endif
