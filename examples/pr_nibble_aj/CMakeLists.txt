# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Local Graph Clustering (PR Nibble)
# ------------------------------------------------------------------------
project(pr_nibble_aj)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_PR_NIBBLE COMMAND pr_nibble_aj  market
	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx) 
set_tests_properties(TEST_PR_NIBBLE PROPERTIES PASS_REGULAR_EXPRESSION "0 errors occurred")
