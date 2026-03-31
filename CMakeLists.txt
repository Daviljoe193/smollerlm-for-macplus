cmake_minimum_required(VERSION 3.5)

# Enable both C and C++ languages in the project
project(SmolMac C CXX)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -m68000 -fomit-frame-pointer -funroll-loops")

add_application(SmolMac runsmol.c)

# Force CMake to use the C++ linker (g++) so RetroConsole's C++ dependencies are satisfied
set_target_properties(SmolMac PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(SmolMac RetroConsole m)
