cmake_minimum_required(VERSION 3.20)
project(Mali64 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Dependências do Sistema ---
find_package(PkgConfig REQUIRED)
find_package(SDL2 REQUIRED)
pkg_check_modules(GLESv2 REQUIRED glesv2)

# --- Submódulos Automáticos (FetchContent) ---
include(FetchContent)

FetchContent_Declare(
    N64ModernRuntime
    GIT_REPOSITORY https://github.com/N64Recomp/N64ModernRuntime.git
    GIT_TAG main
)
FetchContent_MakeAvailable(N64ModernRuntime)

# --- Biblioteca Mali64 ---
file(GLOB MALI64_SOURCES "src/*.cpp")
add_library(Mali64 STATIC ${MALI64_SOURCES})

target_include_directories(Mali64 PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GLESv2_INCLUDE_DIRS}
)

target_link_libraries(Mali64 PUBLIC 
    SDL2::SDL2 
    ${GLESv2_LIBRARIES}
    ultramodern
    librecomp
)

# --- Executável do Jogo ---
file(GLOB GAME_SOURCES "game/*.cpp")
add_executable(Mali64_Game ${GAME_SOURCES})

target_link_libraries(Mali64_Game PRIVATE 
    Mali64
    SDL2::SDL2 
    ${GLESv2_LIBRARIES}
    ultramodern
    librecomp
)

# Otimizações para a CPU do Anbernic (64-bit)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_options(Mali64 PRIVATE -O3 -mcpu=cortex-a53)
endif()
