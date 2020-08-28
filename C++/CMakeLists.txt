cmake_minimum_required(VERSION 3.8)

add_definitions(-w)

option(CMAKE_VERBOSE_MAKEFILE "Create verbose makefile" OFF)
option(BUILD_SHARED_LIBS "Create dark as a shared library" ON)
option(BUILD_AS_CPP "Build Darknet using C++ compiler also for C files" ON)
option(ENABLE_OPENCV "Enable OpenCV integration" ON)

project(Darknet)

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Install prefix" FORCE)
endif()

set(INSTALL_BIN_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where lib will be installed")
set(INSTALL_INCLUDE_DIR  "include/darknet"           CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "share/darknet"             CACHE PATH "Path where cmake configs will be installed")

if(USE_INTEGRATED_LIBS)
  set(PThreads_windows_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/pthreads CACHE PATH "Path where pthreads for windows can be located")
endif()
set(Stb_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb CACHE PATH "Path where Stb image library can be located")

set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
if(MSVC)
  find_package(PThreads_windows REQUIRED)
endif()
if(ENABLE_OPENCV)
  find_package(OpenCV)
  if(OpenCV_FOUND)
    if(SELECT_OPENCV_MODULES)
      if(TARGET opencv_world)
        list(APPEND OpenCV_LINKED_COMPONENTS "opencv_world")
      else()
        if(TARGET opencv_core)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_core")
        endif()
        if(TARGET opencv_highgui)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_highgui")
        endif()
        if(TARGET opencv_imgproc)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_imgproc")
        endif()
        if(TARGET opencv_video)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_video")
        endif()
        if(TARGET opencv_videoio)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_videoio")
        endif()
        if(TARGET opencv_imgcodecs)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_imgcodecs")
        endif()
        if(TARGET opencv_text)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_text")
        endif()
      endif()
    else()
      list(APPEND OpenCV_LINKED_COMPONENTS ${OpenCV_LIBS})
    endif()
  endif()
endif()

list(APPEND OpenCV_LINKED_COMPONENTS "opencv_freetype")

find_package(Stb REQUIRED)
if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
  message(WARNING "To build with OpenMP support you need CMake 3.11.0+")
else()
  find_package(OpenMP)
endif()

set(ADDITIONAL_CXX_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings")
set(ADDITIONAL_C_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings")

if(MSVC)
  set(ADDITIONAL_CXX_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
  set(ADDITIONAL_C_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
  set(CMAKE_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${ADDITIONAL_C_FLAGS} ${CMAKE_C_FLAGS}")
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
endif()

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (UNIX AND NOT APPLE)
      set(CMAKE_CXX_FLAGS "-pthread ${CMAKE_CXX_FLAGS}")  #force pthread to avoid bugs in some cmake setups
      set(CMAKE_C_FLAGS "-pthread ${CMAKE_C_FLAGS}")
    endif()
  endif()
  set(CMAKE_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${ADDITIONAL_C_FLAGS} ${CMAKE_C_FLAGS}")
  string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REGEX REPLACE "-O0" "-Og" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
endif()

set(DARKNET_INSTALL_INCLUDE_DIR ${INSTALL_INCLUDE_DIR})
# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

#look for all *.h files in src folder
file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/src/*.h")
file(GLOB lprheaders "${CMAKE_CURRENT_LIST_DIR}/include/lpr/*.h")
#add also files in the include folder
list(APPEND headers
        ${CMAKE_CURRENT_LIST_DIR}/include/darknet.h
        "${CMAKE_CURRENT_LIST_DIR}/src/VID.h"
        )

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

#set(exported_headers ${headers})

#look for all *.c files in src folder
file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/src/*.c")
file(GLOB lprsources "${CMAKE_CURRENT_LIST_DIR}/src/lpr/*.cpp")

#add also .cpp files
list(APPEND sources
        ${CMAKE_CURRENT_LIST_DIR}/src/plateRecognition.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/speedEstimation.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/camcal.cpp
        "${CMAKE_CURRENT_LIST_DIR}/src/VID.cpp"
        )
#remove darknet.c file which is necessary only for the executable, not for the lib
list(REMOVE_ITEM sources
        ${CMAKE_CURRENT_LIST_DIR}/src/darknet.c
        )
#remove windows only files
if(NOT WIN32)
  list(REMOVE_ITEM sources
          ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.c
          ${CMAKE_CURRENT_LIST_DIR}/src/getopt.c
          )
endif()


#remove windows only files
if(NOT WIN32)
  list(REMOVE_ITEM headers
          ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.h
          ${CMAKE_CURRENT_LIST_DIR}/src/getopt.h
          )
endif()

if(BUILD_AS_CPP)
  set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
endif()

add_library(dark ${headers} ${sources} ${lprheaders} ${lprsources})
add_executable(retest ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp ${headers} ${sources} ${lprheaders} ${lprsources})
set_target_properties(dark PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(dark PRIVATE LIB_EXPORTS=1)
endif()
if(BUILD_AS_CPP)
  set_target_properties(dark PROPERTIES LINKER_LANGUAGE CXX)
endif()


# add_executable(darknet ${headers} ${sources} ${lprheaders} ${lprsources})
#add_executable(drawMask ${CMAKE_CURRENT_LIST_DIR}/src/drawROI.cpp)
#add_executable(camcal ${CMAKE_CURRENT_LIST_DIR}/src/camCal.cpp)
add_executable(prepare ${CMAKE_CURRENT_LIST_DIR}/src/prepare.cpp ${CMAKE_CURRENT_LIST_DIR}/src/camcal.cpp ${CMAKE_CURRENT_LIST_DIR}/src/drawROI.cpp)

# SET_TARGET_PROPERTIES(darknet PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../")
SET_TARGET_PROPERTIES(prepare PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../")
SET_TARGET_PROPERTIES(retest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../")

# target_include_directories(darknet PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:${DARKNET_INSTALL_INCLUDE_DIR}> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
target_include_directories(dark PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:${DARKNET_INSTALL_INCLUDE_DIR}> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
# target_include_directories(uselib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:${DARKNET_INSTALL_INCLUDE_DIR}> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)

# target_compile_definitions(darknet PRIVATE -DUSE_CMAKE_LIBS)
target_compile_definitions(dark PRIVATE -DUSE_CMAKE_LIBS)
# target_compile_definitions(uselib PRIVATE -DUSE_CMAKE_LIBS)

if(OpenCV_FOUND)
  #   target_link_libraries(darknet PRIVATE ${OpenCV_LINKED_COMPONENTS})
  target_link_libraries(prepare PRIVATE ${OpenCV_LINKED_COMPONENTS})
  #  target_link_libraries(uselib PRIVATE ${OpenCV_LINKED_COMPONENTS})
  target_link_libraries(dark PUBLIC ${OpenCV_LINKED_COMPONENTS})

  target_link_libraries(retest PUBLIC ${OpenCV_LINKED_COMPONENTS})

  target_include_directories(dark PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_include_directories(retest PUBLIC ${OpenCV_INCLUDE_DIRS})
  #   target_compile_definitions(darknet PRIVATE -DOPENCV)
  target_compile_definitions(dark PUBLIC -DOPENCV)
  target_compile_definitions(retest PUBLIC -DOPENCV)
endif()

if(OPENMP_FOUND)
  #   target_link_libraries(darknet PRIVATE OpenMP::OpenMP_CXX)
  #   target_link_libraries(darknet PRIVATE OpenMP::OpenMP_C)
  target_link_libraries(dark PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(dark PUBLIC OpenMP::OpenMP_C)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  #   target_link_libraries(darknet PRIVATE m)
  target_link_libraries(dark PUBLIC m)
endif()

if(MSVC)
  target_link_libraries(darknet PRIVATE PThreads_windows::PThreads_windows)
  target_link_libraries(darknet PRIVATE wsock32 ws2_32)
  target_link_libraries(dark PUBLIC PThreads_windows::PThreads_windows)
  target_link_libraries(dark PUBLIC wsock32 ws2_32)
  target_link_libraries(uselib PRIVATE PThreads_windows::PThreads_windows)
  target_compile_definitions(darknet PRIVATE -D_CRT_RAND_S -DNOMINMAX -D_USE_MATH_DEFINES)
  target_compile_definitions(dark PRIVATE -D_CRT_RAND_S -DNOMINMAX -D_USE_MATH_DEFINES)
  target_compile_definitions(dark PUBLIC -D_CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(uselib PRIVATE -D_CRT_RAND_S -DNOMINMAX -D_USE_MATH_DEFINES)
endif()

# target_link_libraries(darknet PRIVATE Threads::Threads)
target_link_libraries(dark PUBLIC Threads::Threads)
# target_link_libraries(uselib PRIVATE Threads::Threads)


if(USE_INTEGRATED_LIBS)
  target_compile_definitions(darknet PRIVATE -D_TIMESPEC_DEFINED)
  target_compile_definitions(dark PRIVATE -D_TIMESPEC_DEFINED)
endif()

# target_link_libraries(uselib PRIVATE dark)

#set_target_properties(dark PROPERTIES PUBLIC_HEADER "${exported_headers};${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp")
set_target_properties(dark PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_LIST_DIR}/include/darknet.h")

set_target_properties(dark PROPERTIES CXX_VISIBILITY_PRESET hidden)

install(TARGETS dark EXPORT DarknetTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
        COMPONENT dev
        )

if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND BUILD_USELIB_TRACK)
  install(TARGETS uselib_track
          DESTINATION "${INSTALL_BIN_DIR}"
          )
endif()

install(EXPORT DarknetTargets
        FILE DarknetTargets.cmake
        NAMESPACE Darknet::
        DESTINATION "${INSTALL_CMAKE_DIR}"
        )

# Export the package for use from the build-tree (this registers the build-tree with a global CMake-registry)
export(PACKAGE Darknet)

# Create the DarknetConfig.cmake
# First of all we compute the relative path between the cmake config file and the include path
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
# configure_file(DarknetConfig.cmake.in "${PROJECT_BINARY_DIR}/DarknetConfig.cmake" @ONLY)
set(CONF_INCLUDE_DIRS "\${Darknet_CMAKE_DIR}/${REL_INCLUDE_DIR}")