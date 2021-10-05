add_library(
  torcharrow_udfs OBJECT
  functions.h
  string_functions.h
  register_udf.cpp
)

target_link_libraries(
  torcharrow_udfs
  velox_functions_string
  velox_functions_prestosql
)
