#include "mali64/renderer_context.h"
#include <SDL2/SDL.h>
#include <cstdio>

// Inclui os headers do runtime para acessar as definições do N64
#include "ultramodern/ultra64.h"

namespace mali64 {

    RendererContext::RendererContext(uint8_t* rdram, ultramodern::renderer::WindowHandle window_handle, bool developer_mode)
    : rdram_(rdram), window_(window_handle) {

        renderer_ = std::make_unique<Renderer>();
        if (renderer_->init()) {
            valid_ = true;
            std::printf("Mali64: RendererContext inicializado com sucesso.\n");
        } else {
            std::fprintf(stderr, "Mali64: Falha ao inicializar o OpenGL ES.\n");
        }
    }

    RendererContext::~RendererContext() {
        shutdown();
    }

    bool RendererContext::valid() {
        return valid_;
    }

    bool RendererContext::update_config(const ultramodern::renderer::GraphicsConfig& old_config, const ultramodern::renderer::GraphicsConfig& new_config) {
        // Se o jogo mudar a resolução no meio da execução, podemos processar aqui.
        return true;
    }

    void RendererContext::enable_instant_present() {
        // Desativa o VSync se o jogo pedir (útil para jogos de luta/timing)
        SDL_GL_SetSwapInterval(0);
    }

    void RendererContext::send_dl(const OSTask* task) {
        if (task && task->t.type == M_GFXTASK) {
            // Passa o endereço físico da display list e do microcódigo para o nosso renderizador
            renderer_->run_task(rdram_, task->t.data_ptr, task->t.ucode_data);
        }
    }

    void RendererContext::send_dummy_workload(uint32_t fb_address) {
        // Chamado quando o jogo quer limpar a tela ou não envia uma task, mas quer um frame novo.
    }

    void RendererContext::update_screen() {
        // O runtime chama isso a cada V-Blank (60 vezes por segundo).
        // Precisamos pegar o VI_ORIGIN atual (o framebuffer que o jogo quer mostrar).

        // NOTA: O N64ModernRuntime expõe a origem do VI através do barramento.
        // Assumindo que essa função está disponível no header do runtime:
    uint32_t vi_origin = 0;
    auto vi_regs = ultramodern::renderer::get_vi_regs();
    if (vi_regs) vi_origin = vi_regs->VI_ORIGIN_REG;

        int w = 0, h = 0;
        // No Linux, window_handle é um SDL_Window*
        SDL_Window* sdl_window = reinterpret_cast<SDL_Window*>(window_);
        if (sdl_window) {
            SDL_GetWindowSize(sdl_window, &w, &h);
        }

        // Apresenta o frame na tela
        renderer_->present(w, h, vi_origin);

        // Troca os buffers do SDL (isso exibe a imagem que acabamos de desenhar)
        if (sdl_window) {
            SDL_GL_SwapWindow(sdl_window);
        }
    }

    void RendererContext::shutdown() {
        if (renderer_) {
            renderer_->shutdown();
            renderer_.reset();
        }
        valid_ = false;
    }

    uint32_t RendererContext::get_display_framerate() const {
        return 60; // O N64 roda a 60Hz (ou 50Hz em PAL)
    }

    float RendererContext::get_resolution_scale() const {
        return 1.0f; // Resolução nativa do N64 (320x240), escalada para a tela do Anbernic
    }

} // namespace mali64
