project(odm_orthophoto)
cmake_minimum_required(VERSION 2.8)

set(OPENCV_DIR "OPENCV_DIR-NOTFOUND" CACHE "OPENCV_DIR" "Path to the OPENCV installation directory")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# Add compiler options.
if (NOT WIN32)
add_definitions(-Wall -Wextra)
set(STDPPFS_LIBRARY stdc++fs)
endif()

find_package(GDAL REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${GDAL_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})

# Find OpenCV at the default location
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)

# Only link with required opencv modules.
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui)

include_directories(${OpenCV_INCLUDE_DIRS})

# Add source directory
aux_source_directory("./src" SRC_LIST)

# Add exectuteable
add_executable(${PROJECT_NAME} ${SRC_LIST})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
target_link_libraries(odm_orthophoto ${OpenCV_LIBS} ${GDAL_LIBRARY} ${STDPPFS_LIBRARY} )

install(TARGETS odm_orthophoto RUNTIME DESTINATION bin)
