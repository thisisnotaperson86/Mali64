#include <SDL2/SDL.h>
#include <GLES3/gl3.h>
#include <cstdio>
#include <memory>

#include "ultramodern/ultramodern.hpp"
#include "ultramodern/renderer_context.hpp"
#include "mali64/renderer_context.h"

int main(int argc, char** argv) {
    // 1. Inicializa o SDL2
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) != 0) {
        std::fprintf(stderr, "Erro ao inicializar SDL: %s\n", SDL_GetError());
        return 1;
    }

    // 2. Configura os atributos do OpenGL ES 3.0
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    // Cria a janela (320x240 é a resolução nativa do N64)
    SDL_Window* window = SDL_CreateWindow(
        "Mali64 - N64 Modern Runtime",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        320, 240,
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI
    );

    if (!window) {
        std::fprintf(stderr, "Erro ao criar janela: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // 3. Cria o contexto OpenGL
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        std::fprintf(stderr, "Erro ao criar contexto OpenGL: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_GL_SetSwapInterval(1); // V-Sync ligado

    std::printf("OpenGL Info:\n");
    std::printf("  Vendor: %s\n", (const char*)glGetString(GL_VENDOR));
    std::printf("  Renderer: %s\n", (const char*)glGetString(GL_RENDERER));
    std::printf("  Version: %s\n", (const char*)glGetString(GL_VERSION));

    // 4. Registra o Mali64 no N64ModernRuntime
    ultramodern::renderer::callbacks_t renderer_callbacks;
    
    // NOTA: Especificamos o tipo de retorno explicitamente para o C++ saber que é um RendererContext
    renderer_callbacks.create_render_context = [](uint8_t* rdram, ultramodern::renderer::WindowHandle window_handle, bool developer_mode) -> std::unique_ptr<ultramodern::renderer::RendererContext> {
        return std::make_unique<mali64::RendererContext>(rdram, window_handle, developer_mode);
    };
    
    ultramodern::renderer::set_callbacks(renderer_callbacks);

    std::printf("Mali64: Runtime inicializado com sucesso.\n");

    // 5. Loop principal
    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }
        SDL_Delay(16);
    }

    // 6. Limpeza
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
