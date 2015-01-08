CXX=g++

CXX_FLAGS=-g
TEST_FILES_CXX=dot_product.cpp
EXEC_CXX_NAMES=$(TEST_FILES_CXX%.cpp=)
INCLUDE_FILES=-I../src/

dot_product: dot_product.cpp
	$(CXX)  $(CXX_FLAGS) $(INCLUDE_FILES) dot_product.cpp -o dot_product
