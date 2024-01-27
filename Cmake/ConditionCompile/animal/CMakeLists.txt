option(USE_CATTWO "Use cat two" ON)

if(USE_CATTWO)
  set(SRC cat.cpp dog.cpp cattwo.cpp)
else()
  set(SRC cat.cpp dog.cpp)
endif()



add_library(AnimalLib ${SRC})

if(USE_CATTWO)
  target_compile_definitions(AnimalLib PRIVATE "USE_CATTWO")
endif()

