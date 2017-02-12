find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(../include)

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

SET(STEREOSRCS 
	census.cu 
	matching_cost.cu 
	scan_scost.cu 
	winner_takes_all.cu 
	median_filter.cu 
	check_consistency.cu 
	cuda_utils.cu
)

CUDA_ADD_LIBRARY(sgm stereo_sgm.cpp ${STEREOSRCS} STATIC)
target_link_libraries(sgm PRIVATE ${OpenCV_LIBRARUES})
target_include_directories(sgm PRIVATE ${OpenCV_INCLUDE_DIRS})
