cmake_minimum_required(VERSION 3.20)
project(Mali64 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Dependências do Sistema ---
find_package(SDL2 REQUIRED)
find_package(OpenGLES3 REQUIRED)

# --- Submódulos Automáticos (FetchContent) ---
include(FetchContent)

# Baixa o N64Recomp
FetchContent_Declare(
    N64Recomp
    GIT_REPOSITORY https://github.com/N64Recomp/N64Recomp.git
    GIT_TAG main
)
FetchContent_MakeAvailable(N64Recomp)

# Baixa o N64ModernRuntime
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
)

target_link_libraries(Mali64 PUBLIC
    SDL2::SDL2
    OpenGL::GLES3
    ultramodern
    librecomp
)

# --- Executável do Jogo ---
file(GLOB GAME_SOURCES "game/*.cpp")
add_executable(Mali64_Game ${GAME_SOURCES})

target_link_libraries(Mali64_Game PRIVATE
    Mali64
    SDL2::SDL2
    OpenGL::GLES3
    ultramodern
    librecomp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_options(Mali64 PRIVATE -O3 -mcpu=cortex-a53 -mfpu=neon-fp-armv8)
endif()
