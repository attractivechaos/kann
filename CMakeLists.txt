cmake_minimum_required( VERSION 3.2 )

project( kann )

find_package( BLAS QUIET )
find_package( Threads QUIET )

add_library( kann kann.c kautodiff.c kann.h kautodiff.h )
target_link_libraries( kann m )
if( BLAS_FOUND )
  target_compile_definitions( kann PRIVATE HAVE_CBLAS )
  target_include_directories( kann PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/blas )
  target_link_libraries( kann ${BLAS_LIBRARIES} )
endif( BLAS_FOUND )

if( Threads_FOUND AND CMAKE_USE_PTHREADS_INIT )
  target_compile_definitions( kann PRIVATE HAVE_PTHREAD )
  target_link_libraries( kann ${CMAKE_THREAD_LIBS_INIT} )
endif( Threads_FOUND AND CMAKE_USE_PTHREADS_INIT )



add_subdirectory( kann_extra )
add_subdirectory( examples )

# Installation 
install( TARGETS kann DESTINATION lib )
install( FILES kann.h kautodiff.h DESTINATION include )
