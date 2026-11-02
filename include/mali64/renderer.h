#pragma once

#include <cstdint>
#include <memory>

namespace mali64 {

    // Estatísticas para debug e profiling
    struct Stats {
        uint64_t tasks = 0;
        uint64_t triangles = 0;
        uint64_t draw_calls = 0;
        uint64_t textures_created = 0;
        uint64_t programs_created = 0;
    };

    class Renderer {
    public:
        Renderer();
        ~Renderer();

        // Inicializa o contexto OpenGL ES, shaders básicos e buffers
        bool init();

        // Libera todos os recursos da GPU
        void shutdown();

        // Executa uma display list enviada pelo RSP do N64
        // rdram: ponteiro para a memória RAM do N64
        // dl_addr: endereço físico da display list
        // ucode_data_addr: endereço físico dos dados do microcódigo
        void run_task(uint8_t* rdram, uint32_t dl_addr, uint32_t ucode_data_addr);

        // Apresenta o frame atual na tela (chamado no V-Blank)
        bool present(int win_w, int win_h, uint32_t vi_origin);

        // Restaura o estado do OpenGL para o SDL2 não quebrar
        void restore_gl_state();

        // Lê um framebuffer da GPU de volta para a CPU (usado para efeitos de transição de tela do N64)
        bool read_framebuffer(uint32_t addr, uint8_t* rgba);

        // Retorna as estatísticas atuais
        const Stats& stats() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

} // namespace mali64
