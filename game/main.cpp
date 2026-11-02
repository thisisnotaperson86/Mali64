#include <SDL2/SDL.h>
#include <cstdio>
#include <memory>

#include "ultramodern/ultramodern.hpp"
#include "ultramodern/renderer_context.hpp"
#include "mali64/renderer_context.h"

// Se você tem o código recompilado do jogo, ele geralmente expõe funções
// de inicialização no header principal (ex: "recomp.h").
// #include "recomp.h"

int main(int argc, char** argv) {
    // 1. Inicializa o SDL2
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) != 0) {
        std::fprintf(stderr, "Erro ao inicializar SDL: %s\n", SDL_GetError());
        return 1;
    }

    // 2. Configura os atributos do OpenGL ES 3.0 antes de criar a janela
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

    // Ativa o V-Sync (1 = ligado, 0 = desligado)
    SDL_GL_SetSwapInterval(1);

    std::printf("OpenGL Info:\n");
    std::printf("  Vendor: %s\n", glGetString(GL_VENDOR));
    std::printf("  Renderer: %s\n", glGetString(GL_RENDERER));
    std::printf("  Version: %s\n", glGetString(GL_VERSION));

    // 4. Registra o Mali64 no N64ModernRuntime
    // Isso substitui completamente o RT64!
    ultramodern::renderer::callbacks_t renderer_callbacks;
    renderer_callbacks.create_render_context = [](uint8_t* rdram, ultramodern::renderer::WindowHandle window_handle, bool developer_mode) {
        return std::make_unique<mali64::RendererContext>(rdram, window_handle, developer_mode);
    };
    ultramodern::renderer::set_callbacks(renderer_callbacks);

    // 5. Inicializa o Runtime e o Jogo Recompilado
    // ATENÇÃO: O código abaixo é um exemplo genérico. Você precisará adaptar
    // para a função de inicialização que o N64Recomp gerou para o seu jogo.
    /*
     *   uint8_t* rdram = new uint8_t[8 * 1024 * 1024]; // 8MB de RAM do N64
     *   ultramodern::recomp::start(rdram, 0x80100400, "mods/"); // Endereço de entrypoint genérico
     */

    std::printf("Mali64: Runtime inicializado com sucesso.\n");

    // 6. Loop principal (Simulação se o runtime não assumir o controle)
    // Em um jogo real, o N64ModernRuntime tem seu próprio loop de threads.
    // Este loop abaixo só é necessário se você for controlar a janela manualmente.
    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        // O N64ModernRuntime (através do RendererContext::update_screen)
        // cuidará do SDL_GL_SwapWindow internamente.
        SDL_Delay(16); // Apenas para não fritar a CPU no loop de simulação
    }

    // 7. Limpeza
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
