project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

option(test "Build all tests." OFF) # Makes boolean 'test' available.

add_definitions(-std=c++0x)

set(sources
   src/ukf.cpp
   src/main.cpp
   src/tools.cpp)

add_executable(UnscentedKF ${sources})

################################
# Testing
################################
if (test)
  add_library(UKF_lib ./src/ukf.cpp)
  add_library(Tools_lib ./src/Tools.cpp)
  target_link_libraries(UnscentedKF UKF_lib)
  target_link_libraries(UnscentedKF Tools_lib)

  # This adds another subdirectory, which has 'project(gtest)'.
  add_subdirectory(test/googletest-release-1.8.0)

  enable_testing()

  # Include the gtest library. gtest_SOURCE_DIR is available due to
  # 'project(gtest)' above.
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

  ##############
  # Unit Tests
  ##############
  add_executable(RunUnitTests ./test/main.cpp)
  add_executable(ToolsTest ./test/tools_test.cpp)
  add_executable(UKFTest ./test/ukf_test.cpp)

  # Standard linking to gtest stuff.
  target_link_libraries(RunUnitTests gtest gtest_main)
  target_link_libraries(ToolsTest gtest gtest_main)
  target_link_libraries(UKFTest gtest gtest_main)

  # Extra linking for the project.
  target_link_libraries(RunUnitTests UKF_lib)
  target_link_libraries(RunUnitTests Tools_lib)
  target_link_libraries(ToolsTest Tools_lib)
  target_link_libraries(UKFTest UKF_lib)
  target_link_libraries(UKFTest Tools_lib)

  add_test(NAME ToolsTest COMMAND ToolsTest)
  add_test(NAME UKFTest COMMAND UKFTest)
endif()
