add_library(AnimalLib OBJECT cat.cpp dog.cpp)

target_include_directories(AnimalLib PUBLIC .)