#pragma once

#include <memory>
#include "ultramodern/renderer_context.hpp"
#include "mali64/renderer.h"

namespace mali64 {

    class RendererContext : public ultramodern::renderer::RendererContext {
    public:
        RendererContext(uint8_t* rdram, ultramodern::renderer::WindowHandle window_handle, bool developer_mode);
        ~RendererContext() override;

        bool valid() override;
        bool update_config(const ultramodern::renderer::GraphicsConfig& old_config, const ultramodern::renderer::GraphicsConfig& new_config) override;
        void enable_instant_present() override;
        void send_dl(const OSTask* task) override;
        void send_dummy_workload(uint32_t fb_address) override;
        void update_screen() override;
        void shutdown() override;
        uint32_t get_display_framerate() const override;
        float get_resolution_scale() const override;

    private:
        std::unique_ptr<Renderer> renderer_;
        uint8_t* rdram_;
        ultramodern::renderer::WindowHandle window_;
        bool valid_ = false;
    };

} // namespace mali64
