#include "mali64/renderer.h"
#include "gles_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace mali64 {
    namespace {

        // ---------------------------------------------------------------------------------------------------------------
        // Constants
        // ---------------------------------------------------------------------------------------------------------------

        constexpr uint32_t kPhysMask = 0x1FFFFFFF;
        constexpr int kMaxVertices = 64;
        constexpr int kMaxLights = 8;
        constexpr int kMaxTargets = 6;
        constexpr int kTmemSize = 4096;

        enum Op : uint8_t {
            G_SPNOOP = 0x00, G_MTX = 0x01, G_MOVEMEM = 0x03, G_VTX = 0x04, G_DL = 0x06,
            G_LOAD_UCODE = 0xAF, G_BRANCH_Z = 0xB0, G_TRI2 = 0xB1, G_MODIFYVTX = 0xB2, G_RDPHALF_2 = 0xB3, G_RDPHALF_1 = 0xB4,
            G_QUAD = 0xB5, G_CLEARGEOMETRYMODE = 0xB6, G_SETGEOMETRYMODE = 0xB7, G_ENDDL = 0xB8, G_SETOTHERMODE_L = 0xB9,
            G_SETOTHERMODE_H = 0xBA, G_TEXTURE = 0xBB, G_MOVEWORD = 0xBC, G_POPMTX = 0xBD, G_CULLDL = 0xBE, G_TRI1 = 0xBF,
            G_NOOP = 0xC0, G_TEXRECT = 0xE4, G_TEXRECTFLIP = 0xE5, G_RDPLOADSYNC = 0xE6, G_RDPPIPESYNC = 0xE7,
            G_RDPTILESYNC = 0xE8, G_RDPFULLSYNC = 0xE9, G_SETKEYGB = 0xEA, G_SETKEYR = 0xEB, G_SETCONVERT = 0xEC,
            G_SETSCISSOR = 0xED, G_SETPRIMDEPTH = 0xEE, G_RDPSETOTHERMODE = 0xEF, G_LOADTLUT = 0xF0, G_SETTILESIZE = 0xF2,
            G_LOADBLOCK = 0xF3, G_LOADTILE = 0xF4, G_SETTILE = 0xF5, G_FILLRECT = 0xF6, G_SETFILLCOLOR = 0xF7,
            G_SETFOGCOLOR = 0xF8, G_SETBLENDCOLOR = 0xF9, G_SETPRIMCOLOR = 0xFA, G_SETENVCOLOR = 0xFB, G_SETCOMBINE = 0xFC,
            G_SETTIMG = 0xFD, G_SETZIMG = 0xFE, G_SETCIMG = 0xFF,
        };

        constexpr uint32_t GM_ZBUFFER = 0x00000001;
        constexpr uint32_t GM_SHADE = 0x00000004;
        constexpr uint32_t GM_SHADING_SMOOTH = 0x00000200;
        constexpr uint32_t GM_CULL_FRONT = 0x00001000;
        constexpr uint32_t GM_CULL_BACK = 0x00002000;
        constexpr uint32_t GM_FOG = 0x00010000;
        constexpr uint32_t GM_LIGHTING = 0x00020000;
        constexpr uint32_t GM_TEXTURE_GEN = 0x00040000;
        constexpr uint32_t GM_TEXTURE_GEN_LINEAR = 0x00080000;

        constexpr uint32_t OML_ALPHA_COMPARE_MASK = 0x3;
        constexpr uint32_t OML_Z_CMP = 0x00000010;
        constexpr uint32_t OML_Z_UPD = 0x00000020;
        constexpr uint32_t OML_ZMODE_SHIFT = 10;
        constexpr uint32_t OML_CVG_X_ALPHA = 0x00001000;
        constexpr uint32_t OML_FORCE_BL = 0x00004000;

        enum CycleType { CYC_1 = 0, CYC_2 = 1, CYC_COPY = 2, CYC_FILL = 3 };

        constexpr uint32_t kLogMax = 400;

        // ---------------------------------------------------------------------------------------------------------------
        // Small helpers
        // ---------------------------------------------------------------------------------------------------------------

        struct Mat { float m[4][4]; };

        Mat identity() {
            Mat r{};
            for (int i = 0; i < 4; i++) r.m[i][i] = 1.0f;
            return r;
        }

        Mat mul(const Mat &a, const Mat &b) {
            Mat r{};
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    float s = 0.0f;
                    for (int k = 0; k < 4; k++) s += a.m[i][k] * b.m[k][j];
                    r.m[i][j] = s;
                }
                return r;
        }

        uint64_t fnv1a(const void *data, size_t size, uint64_t h = 1469598103934665603ull) {
            const uint8_t *p = static_cast<const uint8_t *>(data);
            for (size_t i = 0; i < size; i++) {
                h ^= p[i];
                h *= 1099511628211ull;
            }
            return h;
        }

        float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

        struct Mem {
            uint8_t *base = nullptr;
            uint8_t u8(uint32_t a) const { return base[(a & kPhysMask) ^ 3]; }
            int8_t s8(uint32_t a) const { return static_cast<int8_t>(u8(a)); }
            uint16_t u16(uint32_t a) const {
                uint16_t v;
                std::memcpy(&v, base + ((a & kPhysMask) ^ 2), 2);
                return v;
            }
            int16_t s16(uint32_t a) const { return static_cast<int16_t>(u16(a)); }
            uint32_t u32(uint32_t a) const {
                uint32_t v;
                std::memcpy(&v, base + (a & kPhysMask), 4);
                return v;
            }
            void put32(uint32_t a, uint32_t v) { std::memcpy(base + (a & kPhysMask), &v, 4); }
        };

        // ---------------------------------------------------------------------------------------------------------------
        // GL objects
        // ---------------------------------------------------------------------------------------------------------------

        struct Target {
            uint32_t addr = 0;
            uint32_t bytes = 0;
            GLuint fbo = 0;
            GLuint color = 0;
            GLuint depth = 0;
            uint64_t last_use = 0;
        };

        struct Program {
            GLuint id = 0;
            GLint u_prim = -1, u_env = -1, u_fog = -1, u_blend = -1, u_misc = -1;
            GLint u_tile0 = -1, u_tile1 = -1, u_flip = -1, u_tex0 = -1, u_tex1 = -1;
            bool use0 = false, use1 = false;
        };

        struct TexEntry {
            GLuint id = 0;
            uint64_t last_use = 0;
        };

        struct DVert {
            float x, y, z, w;
            uint8_t r, g, b, a;
            float s, t;
        };

        struct DrawKey {
            uint32_t comb0 = 0, comb1 = 0;
            uint32_t mode_h = 0, mode_l = 0;
            uint32_t kind = 0;
            uint32_t tex0 = 0, tex1 = 0;
            uint32_t target = 0;
            uint32_t depth_test = 0, depth_write = 0, depth_decal = 0;
            uint32_t blend_src = 0, blend_dst = 0, blend_on = 0;
            int32_t viewport[4] = {0, 0, 0, 0};
            int32_t scissor[4] = {0, 0, 0, 0};
            float depth_range[2] = {0, 1};
            float flip_y = 1.0f;
            float prim[4] = {0, 0, 0, 0};
            float env[4] = {0, 0, 0, 0};
            float fog[4] = {0, 0, 0, 0};
            float blend[4] = {0, 0, 0, 0};
            float misc[4] = {0, 0, 0, 0};
            float tile0[4] = {0, 0, 0, 0};
            float tile1[4] = {0, 0, 0, 0};
        };

        struct ProgKey {
            uint32_t comb0, comb1, blender, cycle, flags;
            bool operator<(const ProgKey &o) const { return std::memcmp(this, &o, sizeof(*this)) < 0; }
        };

        struct Tile {
            uint8_t fmt = 0, siz = 0;
            uint16_t line = 0, tmem = 0;
            uint8_t pal = 0, cmt = 0, cms = 0, maskt = 0, masks = 0, shiftt = 0, shifts = 0;
            uint16_t uls = 0, ult = 0, lrs = 0, lrt = 0;
            bool size_set = false;
        };

        struct Light {
            uint8_t col[3] = {0, 0, 0};
            int8_t dir[3] = {0, 0, 0};
        };

        struct VtxOut {
            float clip[4] = {0, 0, 0, 1};
            float s = 0, t = 0;
            uint8_t rgba[4] = {255, 255, 255, 255};
        };

        // ---------------------------------------------------------------------------------------------------------------
        // Shader generation
        // ---------------------------------------------------------------------------------------------------------------

        const char *kVertexShader = R"(#version 300 es
precision highp float;
layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in vec2 aST;
uniform vec4 uTile0;
uniform vec4 uTile1;
uniform float uFlipY;
out vec4 vColor;
out vec2 vUV0;
out vec2 vUV1;
void main() {
    gl_Position = vec4(aPos.x, aPos.y * uFlipY, aPos.z, aPos.w);
    vColor = aColor;
    vUV0 = aST * uTile0.xy - uTile0.zw;
    vUV1 = aST * uTile1.xy - uTile1.zw;
}
)";

const char *kPresentVS = R"(#version 300 es
precision highp float;
layout(location = 0) in vec4 aPos;
layout(location = 2) in vec2 aST;
out vec2 vUV;
void main() { gl_Position = vec4(aPos.xy, 0.0, 1.0); vUV = aST; }
)";

const char *kPresentFS = R"(#version 300 es
precision mediump float;
in vec2 vUV;
uniform sampler2D uTex0;
out vec4 oColor;
void main() { oColor = vec4(texture(uTex0, vUV).rgb, 1.0); }
)";

std::string rgbInput(int sel, int kind, bool swap) {
    const char *t0 = swap ? "texel1" : "texel0";
    const char *t1 = swap ? "texel0" : "texel1";
    auto tex = [&](int n, bool alpha) {
        return std::string(n == 0 ? t0 : t1) + (alpha ? ".a" : ".rgb");
    };
    switch (kind) {
        case 0:
            switch (sel) {
                case 0: return "combined.rgb";
                case 1: return tex(0, false);
                case 2: return tex(1, false);
                case 3: return "uPrim.rgb";
                case 4: return "shade.rgb";
                case 5: return "uEnv.rgb";
                case 6: return "vec3(1.0)";
                case 7: return "vec3(0.5)";
                default: return "vec3(0.0)";
            }
                case 1:
                    switch (sel) {
                        case 0: return "combined.rgb";
                        case 1: return tex(0, false);
                        case 2: return tex(1, false);
                        case 3: return "uPrim.rgb";
                        case 4: return "shade.rgb";
                        case 5: return "uEnv.rgb";
                        case 6: return "vec3(0.0)";
                        case 7: return "vec3(uMisc.y)";
                        default: return "vec3(0.0)";
                    }
                        case 2:
                            switch (sel) {
                                case 0: return "combined.rgb";
                                case 1: return tex(0, false);
                                case 2: return tex(1, false);
                                case 3: return "uPrim.rgb";
                                case 4: return "shade.rgb";
                                case 5: return "uEnv.rgb";
                                case 6: return "vec3(1.0)";
                                case 7: return "vec3(combined.a)";
                                case 8: return "vec3(" + tex(0, true) + ")";
                                case 9: return "vec3(" + tex(1, true) + ")";
                                case 10: return "vec3(uPrim.a)";
                                case 11: return "vec3(shade.a)";
                                case 12: return "vec3(uEnv.a)";
                                case 13: return "vec3(0.0)";
                                case 14: return "vec3(uMisc.x)";
                                case 15: return "vec3(uMisc.z)";
                                default: return "vec3(0.0)";
                            }
                                default:
                                    switch (sel) {
                                        case 0: return "combined.rgb";
                                        case 1: return tex(0, false);
                                        case 2: return tex(1, false);
                                        case 3: return "uPrim.rgb";
                                        case 4: return "shade.rgb";
                                        case 5: return "uEnv.rgb";
                                        case 6: return "vec3(1.0)";
                                        default: return "vec3(0.0)";
                                    }
    }
}

std::string alphaInput(int sel, bool is_c, bool swap) {
    const char *t0 = swap ? "texel1.a" : "texel0.a";
    const char *t1 = swap ? "texel0.a" : "texel1.a";
    if (is_c) {
        switch (sel) {
            case 0: return "0.0";
            case 1: return t0;
            case 2: return t1;
            case 3: return "uPrim.a";
            case 4: return "shade.a";
            case 5: return "uEnv.a";
            case 6: return "uMisc.x";
            default: return "0.0";
        }
    }
    switch (sel) {
        case 0: return "combined.a";
        case 1: return t0;
        case 2: return t1;
        case 3: return "uPrim.a";
        case 4: return "shade.a";
        case 5: return "uEnv.a";
        case 6: return "1.0";
        default: return "0.0";
    }
}

struct CombineSlot {
    int a, b, c, d, aa, ab, ac, ad;
};

CombineSlot decodeSlot(uint32_t w0, uint32_t w1, int slot) {
    CombineSlot s;
    if (slot == 0) {
        s.a = (w0 >> 20) & 0xF;
        s.c = (w0 >> 15) & 0x1F;
        s.aa = (w0 >> 12) & 7;
        s.ac = (w0 >> 9) & 7;
        s.b = (w1 >> 28) & 0xF;
        s.d = (w1 >> 15) & 7;
        s.ab = (w1 >> 12) & 7;
        s.ad = (w1 >> 9) & 7;
    } else {
        s.a = (w0 >> 5) & 0xF;
        s.c = w0 & 0x1F;
        s.b = (w1 >> 24) & 0xF;
        s.aa = (w1 >> 21) & 7;
        s.ac = (w1 >> 18) & 7;
        s.d = (w1 >> 6) & 7;
        s.ab = (w1 >> 3) & 7;
        s.ad = w1 & 7;
    }
    return s;
}

void combinerTextureUse(uint32_t w0, uint32_t w1, int cycle, bool &use0, bool &use1) {
    use0 = use1 = false;
    auto note = [&](int texel, bool swap) {
        int slot = (texel != 0) != swap ? 1 : 0;
        if (cycle == CYC_1 && texel == 1) slot = 0;
        (slot == 0 ? use0 : use1) = true;
    };
    int first = (cycle == CYC_2) ? 0 : 1;
    for (int slot = first; slot < 2; slot++) {
        bool swap = (cycle == CYC_2 && slot == 1);
        CombineSlot s = decodeSlot(w0, w1, slot);
        auto rgb = [&](int sel, int kind) {
            if (kind == 2) {
                if (sel == 1 || sel == 8) note(0, swap);
                if (sel == 2 || sel == 9) note(1, swap);
            } else {
                if (sel == 1) note(0, swap);
                if (sel == 2) note(1, swap);
            }
        };
        rgb(s.a, 0);
        rgb(s.b, 1);
        rgb(s.c, 2);
        rgb(s.d, 3);
        for (int sel : {s.aa, s.ab, s.ad}) {
            if (sel == 1) note(0, swap);
            if (sel == 2) note(1, swap);
        }
        if (s.ac == 1) note(0, swap);
        if (s.ac == 2) note(1, swap);
    }
}

std::string blendColor(int sel, const char *in_color) {
    switch (sel) {
        case 0: return in_color;
        case 1: return in_color;
        case 2: return "uBlend.rgb";
        default: return "uFog.rgb";
    }
}

std::string blendAlphaA(int sel, const char *in_alpha) {
    switch (sel) {
        case 0: return in_alpha;
        case 1: return "uFog.a";
        case 2: return "shade.a";
        default: return "0.0";
    }
}

void blenderStage(std::string &src, const char *tag, int m1a, int m1b, int m2a, int m2b, const char *in_color,
                  const char *in_alpha, const std::string &out_color, const std::string &out_alpha, bool &reads_mem) {
    std::string p = blendColor(m1a, in_color);
    std::string a = blendAlphaA(m1b, in_alpha);
    reads_mem = (m2a == 1);
    src += "    float " + std::string(tag) + "A = " + a + ";\n";
    if (reads_mem) {
        src += "    vec3 " + out_color + " = " + p + ";\n";
        src += "    float " + out_alpha + " = " + std::string(tag) + "A;\n";
        return;
    }
    std::string m = blendColor(m2a, in_color);
    std::string b;
    switch (m2b) {
        case 0: b = "(1.0 - " + std::string(tag) + "A)"; break;
        case 1: b = "1.0"; break;
        case 2: b = "1.0"; break;
        default: b = "0.0"; break;
    }
    src += "    float " + std::string(tag) + "B = " + b + ";\n";
    src += "    float " + std::string(tag) + "D = " + std::string(tag) + "A + " + std::string(tag) + "B;\n";
    src += "    vec3 " + out_color + " = " + std::string(tag) + "D > 0.0 ? (" + p + " * " + std::string(tag) + "A + " + m +
    " * " + std::string(tag) + "B) / " + std::string(tag) + "D : " + p + ";\n";
    src += "    float " + out_alpha + " = " + in_alpha + ";\n";
                  }

                  std::string buildFragmentShader(const ProgKey &key, bool &use0, bool &use1) {
                      const int cycle = key.cycle;
                      const bool fill = cycle == CYC_FILL;
                      const bool copy = cycle == CYC_COPY;
                      const uint32_t L = key.blender;
                      const uint32_t flags = key.flags;
                      const int alpha_mode = flags & 3;
                      const bool force_bl = L & OML_FORCE_BL;

                      std::string src = "#version 300 es\nprecision highp float;\n";
                      src += "in vec4 vColor;\nin vec2 vUV0;\nin vec2 vUV1;\n";
                      src += "uniform sampler2D uTex0;\nuniform sampler2D uTex1;\n";
                      src += "uniform vec4 uPrim;\nuniform vec4 uEnv;\nuniform vec4 uFog;\nuniform vec4 uBlend;\nuniform vec4 uMisc;\n";
                      src += "out vec4 oColor;\nvoid main() {\n";

                      use0 = use1 = false;
                      if (fill) {
                          src += "    oColor = vec4(uPrim.rgb, 1.0);\n}\n";
                          return src;
                      }
                      if (copy) {
                          use0 = true;
                          src += "    vec4 texel0 = texture(uTex0, vUV0);\n";
                          if (L & OML_ALPHA_COMPARE_MASK) src += "    if (texel0.a < ((" + std::to_string(L & 1) + ") != 0 ? uBlend.a : 0.5)) discard;\n";
                          src += "    oColor = vec4(texel0.rgb, 1.0);\n}\n";
                          return src;
                      }

                      combinerTextureUse(key.comb0, key.comb1, cycle, use0, use1);
                      if (use0) src += "    vec4 texel0 = texture(uTex0, vUV0);\n";
                      else src += "    vec4 texel0 = vec4(0.0);\n";
                      if (use1) src += "    vec4 texel1 = texture(uTex1, vUV1);\n";
                      else src += "    vec4 texel1 = vec4(0.0);\n";
                      src += "    vec4 shade = vColor;\n    vec4 combined = vec4(0.0);\n";

                      int first = (cycle == CYC_2) ? 0 : 1;
                      for (int slot = first; slot < 2; slot++) {
                          bool swap = (cycle == CYC_2 && slot == 1);
                          CombineSlot s = decodeSlot(key.comb0, key.comb1, slot);
                          std::string A = rgbInput(s.a, 0, swap), B = rgbInput(s.b, 1, swap), C = rgbInput(s.c, 2, swap), D = rgbInput(s.d, 3, swap);
                          std::string aA = alphaInput(s.aa, false, swap), aB = alphaInput(s.ab, false, swap);
                          std::string aC = alphaInput(s.ac, true, swap), aD = alphaInput(s.ad, false, swap);
                          src += "    {\n";
                          src += "        vec3 rgb = (" + A + " - " + B + ") * " + C + " + " + D + ";\n";
                          src += "        float al = (" + aA + " - " + aB + ") * " + aC + " + " + aD + ";\n";
                          src += "        combined = clamp(vec4(rgb, al), 0.0, 1.0);\n";
                          src += "    }\n";
                      }

                      std::string in_color = "combined.rgb";
                      std::string in_alpha = "combined.a";
                      const int s0[4] = {int((L >> 30) & 3), int((L >> 26) & 3), int((L >> 22) & 3), int((L >> 18) & 3)};
                      const int s1[4] = {int((L >> 28) & 3), int((L >> 24) & 3), int((L >> 20) & 3), int((L >> 16) & 3)};
                      bool reads_mem = false;
                      std::string final_color = "combined.rgb";
                      std::string final_alpha = "combined.a";
                      if (cycle == CYC_2) {
                          bool mem0 = false;
                          blenderStage(src, "b0", s0[0], s0[1], s0[2], s0[3], "combined.rgb", "combined.a", "c1rgb", "c1a", mem0);
                          if (mem0) src += "    c1rgb = combined.rgb;\n";
                          src += "    vec3 c1c = c1rgb;\n    float c1al = combined.a;\n";
                          in_color = "c1c";
                          in_alpha = "c1al";
                      }
                      if (force_bl) {
                          const int *s = s1;
                          blenderStage(src, "b1", s[0], s[1], s[2], s[3], in_color.c_str(), in_alpha.c_str(), "fcol", "fal", reads_mem);
                          final_color = "fcol";
                          final_alpha = "fal";
                      } else {
                          src += "    vec3 fcol = " + in_color + ";\n    float fal = " + in_alpha + ";\n";
                          final_color = "fcol";
                          final_alpha = "fal";
                      }

                      if (alpha_mode == 1) src += "    if (combined.a < uBlend.a) discard;\n";
                      else if (alpha_mode == 2) src += "    if (combined.a < 0.5) discard;\n";

                      if (reads_mem) src += "    oColor = vec4(" + final_color + ", " + final_alpha + ");\n";
                      else src += "    oColor = vec4(" + final_color + ", 1.0);\n";
                      src += "}\n";
                      return src;
                  }

    } // namespace

    struct Renderer::Impl {
        Stats stats;
        bool ready = false;
        bool trace = false;
        uint32_t diag_presents = 0, diag_missing = 0, diag_setcimg = 0, diag_tex = 0, diag_prog = 0, auto_trace_budget = 0;
        uint64_t n_fillrect = 0, n_texrect = 0, n_fill_z = 0;
        bool cull_flip = false;
        uint32_t trace_budget = 0;

        Mem mem;

        // GL
        GLuint vao = 0, vbo = 0;
        // [FIX 2] VAO + VBO dedicados ao fullscreen quad de apresentação.
        GLuint present_vao = 0, present_vbo = 0;
        Program present_prog;
        std::vector<Target> targets;
        std::map<ProgKey, Program> programs;
        std::unordered_map<uint64_t, TexEntry> textures;
        uint64_t frame_counter = 0;
        uint32_t last_target_addr = 0;
        bool have_frame = false;
        // [FIX 4] Buffer reaproveitado pelo hash de resolve_tile
        std::vector<uint8_t> hash_buf;

        // RSP state
        uint32_t segments[16] = {};
        Mat mv_stack[16];
        int mv_depth = 0;
        Mat proj;
        Mat mvp;
        bool mvp_dirty = true;
        uint32_t geom = 0;
        VtxOut vtx[kMaxVertices];
        Light lights[kMaxLights + 1];
        int num_lights = 0;
        float lookat_x[3] = {1, 0, 0}, lookat_y[3] = {0, 1, 0};
        int16_t fog_mul = 256, fog_off = 0;
        uint16_t tex_scale_s = 0xFFFF, tex_scale_t = 0xFFFF;
        uint8_t tex_tile = 0;
        bool tex_on = false;
        float vp_scale[3] = {160, 120, 511}, vp_trans[3] = {160, 120, 511};
        uint32_t half1 = 0;

        // RDP state
        uint32_t mode_h = 0, mode_l = 0;
        uint32_t comb0 = 0, comb1 = 0;
        uint32_t prim_color = 0, env_color = 0, fog_color = 0, blend_color = 0, fill_color = 0;
        uint8_t prim_lod_frac = 0;
        Tile tiles[8];
        uint8_t tmem[kTmemSize] = {};
        uint8_t timg_fmt = 0, timg_siz = 0;
        uint16_t timg_width = 1;
        uint32_t timg_addr = 0;
        uint32_t cimg_addr = 0, zimg_addr = 0;
        uint8_t cimg_siz = 2;
        int32_t scissor[4] = {0, 0, 320 * 4, 240 * 4};
        uint32_t tex_version = 1;
        uint32_t tile_resolved_version[8] = {};
        struct Resolved {
            GLuint id = 0;
            float w = 1, h = 1;
            float mul[2] = {1, 1}, off[2] = {0, 0};
        } resolved[8];

        // batching
        std::vector<DVert> batch;
        DrawKey batch_key;
        DrawKey cur_key;
        bool key_dirty = true;
        bool batch_has_key = false;
        Target *cur_target = nullptr;

        // diagnostics
        bool seen_op[256] = {};
        char ucode_name[64] = {};
        bool ucode_logged = false;

        // -------------------------------------------------------------------------------------------------------
        // setup
        // -------------------------------------------------------------------------------------------------------

        bool init() {
            trace = std::getenv("MALI64_GFX_TRACE") != nullptr;
            cull_flip = std::getenv("MALI64_CULL_FLIP") != nullptr;

            targets.reserve(kMaxTargets);

            // --- VAO/VBO de geometria ---
            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);
            glGenBuffers(1, &vbo);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(DVert), reinterpret_cast<void *>(offsetof(DVert, x)));
            glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(DVert), reinterpret_cast<void *>(offsetof(DVert, r)));
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(DVert), reinterpret_cast<void *>(offsetof(DVert, s)));

            // --- [FIX 2] VAO/VBO dedicados ao present ---
            glGenVertexArrays(1, &present_vao);
            glBindVertexArray(present_vao);
            glGenBuffers(1, &present_vbo);
            glBindBuffer(GL_ARRAY_BUFFER, present_vbo);
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(DVert), reinterpret_cast<void *>(offsetof(DVert, x)));
            glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(DVert), reinterpret_cast<void *>(offsetof(DVert, r)));
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(DVert), reinterpret_cast<void *>(offsetof(DVert, s)));
            {
                // Upload do quad fullscreen uma única vez.
                static const float xy[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
                static const float uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
                static const int order[6] = {0, 1, 2, 0, 2, 3};
                DVert quad[6];
                for (int i = 0; i < 6; i++) {
                    const int k = order[i];
                    quad[i] = DVert{xy[k][0], xy[k][1], 0.0f, 1.0f, 255, 255, 255, 255, uv[k][0], uv[k][1]};
                }
                glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(sizeof(quad)), quad, GL_STATIC_DRAW);
            }

            const char *attribs[3] = {"aPos", "aColor", "aST"};
            present_prog.id = gles_build_program(kPresentVS, kPresentFS, attribs, 3);
            if (present_prog.id == 0) return false;
            present_prog.u_tex0 = glGetUniformLocation(present_prog.id, "uTex0");

            reset_rsp();
            ready = true;
            return true;
        }

        void shutdown() {
            for (auto &t : targets) {
                glDeleteFramebuffers(1, &t.fbo);
                glDeleteTextures(1, &t.color);
                glDeleteRenderbuffers(1, &t.depth);
            }
            targets.clear();
            for (auto &p : programs) glDeleteProgram(p.second.id);
            programs.clear();
            for (auto &t : textures) glDeleteTextures(1, &t.second.id);
            textures.clear();
            if (present_prog.id) glDeleteProgram(present_prog.id);
            present_prog.id = 0;
            if (vbo) glDeleteBuffers(1, &vbo);
            if (vao) glDeleteVertexArrays(1, &vao);
            vbo = vao = 0;
            if (present_vbo) glDeleteBuffers(1, &present_vbo);
            if (present_vao) glDeleteVertexArrays(1, &present_vao);
            present_vbo = present_vao = 0;
            ready = false;
        }

        void reset_rsp() {
            for (auto &s : segments) s = 0;
            mv_depth = 0;
            mv_stack[0] = identity();
            proj = identity();
            mvp_dirty = true;
            geom = 0;
            num_lights = 0;
            for (auto &l : lights) l = Light{};
            lookat_x[0] = 1; lookat_x[1] = 0; lookat_x[2] = 0;
            lookat_y[0] = 0; lookat_y[1] = 1; lookat_y[2] = 0;
            fog_mul = 256;
            fog_off = 0;
            tex_scale_s = tex_scale_t = 0xFFFF;
            tex_tile = 0;
            tex_on = false;
            vp_scale[0] = 160; vp_scale[1] = 120; vp_scale[2] = 511;
            vp_trans[0] = 160; vp_trans[1] = 120; vp_trans[2] = 511;
            for (auto &v : vtx) v = VtxOut{};
            key_dirty = true;
        }

        // -------------------------------------------------------------------------------------------------------
        // render targets
        // -------------------------------------------------------------------------------------------------------

        Target *get_target(uint32_t addr) {
            addr &= kPhysMask;
            for (auto &t : targets) {
                if (t.addr == addr) {
                    t.last_use = ++frame_counter;
                    return &t;
                }
            }
            Target *slot = nullptr;
            if (targets.size() < size_t(kMaxTargets)) {
                targets.emplace_back();
                slot = &targets.back();
            } else {
                slot = &targets[0];
                for (auto &t : targets)
                    if (t.last_use < slot->last_use) slot = &t;
                    glDeleteFramebuffers(1, &slot->fbo);
                glDeleteTextures(1, &slot->color);
                glDeleteRenderbuffers(1, &slot->depth);
            }
            slot->addr = addr;
            slot->bytes = 0;
            slot->last_use = ++frame_counter;
            glGenTextures(1, &slot->color);
            glBindTexture(GL_TEXTURE_2D, slot->color);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glGenRenderbuffers(1, &slot->depth);
            glBindRenderbuffer(GL_RENDERBUFFER, slot->depth);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 640, 480);
            glGenFramebuffers(1, &slot->fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, slot->fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, slot->color, 0);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, slot->depth);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                std::fprintf(stderr, "Mali64: framebuffer incomplete\n");
            }
            glDisable(GL_SCISSOR_TEST);
            glDepthMask(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glClearColor(0, 0, 0, 1);
            glClearDepthf(1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            return slot;
        }

        void select_target(uint32_t addr) {
            flush();
            Target *t = get_target(addr);
            // [FIX 1] cimg_siz: 0=4bpp, 1=8bpp, 2=16bpp, 3=32bpp.
            t->bytes = uint32_t(640) * uint32_t(480) * (1u << cimg_siz) / 2u;
            cur_target = t;
            last_target_addr = t->addr;
            key_dirty = true;
            if (z_clear_pending) {
                glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
                glDisable(GL_SCISSOR_TEST);
                glDepthMask(GL_TRUE);
                glClearDepthf(z_clear_value);
                glClear(GL_DEPTH_BUFFER_BIT);
                z_clear_pending = false;
            }
        }

        // -------------------------------------------------------------------------------------------------------
        // textures
        // -------------------------------------------------------------------------------------------------------

        static int bitsPerTexel(int siz) { return 4 << siz; }

        uint8_t tm(int a) const { return tmem[a & (kTmemSize - 1)]; }

        void resolve_tile(int ti) {
            if (tile_resolved_version[ti] == tex_version) return;
            tile_resolved_version[ti] = tex_version;
            Resolved &r = resolved[ti];
            const Tile &t = tiles[ti];

            int w, h;
            const int bits = bitsPerTexel(t.siz);
            if (t.size_set) {
                w = ((t.lrs - t.uls) >> 2) + 1;
                h = ((t.lrt - t.ult) >> 2) + 1;
            } else {
                w = std::max(1, (t.line * 64) / bits);
                h = t.maskt ? (1 << t.maskt) : 32;
            }
            if (t.masks && (1 << t.masks) < w) w = 1 << t.masks;
            if (t.maskt && (1 << t.maskt) < h) h = 1 << t.maskt;
            w = std::max(1, std::min(w, 1024));
            h = std::max(1, std::min(h, 1024));

            const int row_bytes = t.line ? t.line * 8 : ((w * bits + 63) / 64) * 8;
            const uint32_t tlut_mode = (mode_h >> 14) & 3;
            const bool ci = t.fmt == 2;
            const uint32_t filter = (mode_h >> 12) & 3;

            uint64_t hash = 1469598103934665603ull;
            uint32_t meta[12] = {t.fmt, t.siz, uint32_t(w), uint32_t(h), uint32_t(row_bytes), t.cms, t.cmt,
                filter != 0, tlut_mode, t.pal, t.masks, t.maskt};
                hash = fnv1a(meta, sizeof(meta), hash);

                // [FIX 4] Snapshot do trecho de TMEM que este tile ocupa
                const int tbase = t.tmem * 8;
                const int len_bytes = (w * bits + 7) / 8;
                const int per_row = len_bytes * (t.siz == 3 ? 2 : 1);
                hash_buf.clear();
                hash_buf.reserve(size_t(h) * per_row);
                for (int y = 0; y < h; y++) {
                    const int start = tbase + y * row_bytes;
                    const int sw = (y & 1) ? 4 : 0;
                    for (int i = 0; i < len_bytes; i++) {
                        hash_buf.push_back(tm((start + i) ^ sw));
                        if (t.siz == 3) hash_buf.push_back(tm(((start + i) ^ sw) + 0x800));
                    }
                }
                if (!hash_buf.empty()) hash = fnv1a(hash_buf.data(), hash_buf.size(), hash);
                if (ci) hash = fnv1a(tmem + 0x800, 0x800, hash);

                auto it = textures.find(hash);
            if (it == textures.end()) {
                std::vector<uint32_t> px(size_t(w) * h);
                decode_tile(t, w, h, row_bytes, tlut_mode, px.data());
                if (diag_tex < 40) {
                    diag_tex++;
                    size_t nz_rgb = 0, nz_a = 0;
                    for (uint32_t v : px) {
                        if (v & 0x00FFFFFFu) nz_rgb++;
                        if (v & 0xFF000000u) nz_a++;
                    }
                    size_t tm_nz = 0;
                    for (int i = 0; i < kTmemSize; i++) if (tmem[i]) tm_nz++;
                    std::fprintf(stderr, "Mali64: texture #%u tile %d fmt %d siz %d %dx%d line %d tmem %d tlut_mode %u: %zu/%zu texels with colour, %zu with alpha, first %08x; TMEM non-zero bytes %zu\n",
                                 diag_tex, ti, t.fmt, t.siz, w, h, t.line, t.tmem, tlut_mode, nz_rgb, px.size(), nz_a, px[0], tm_nz);
                }
                TexEntry e;
                glGenTextures(1, &e.id);
                glBindTexture(GL_TEXTURE_2D, e.id);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
                const GLint mag = filter ? GL_LINEAR : GL_NEAREST;
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mag);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag);
                auto wrap = [&](int cm, int mask) -> GLint {
                    if (cm & 2) return GL_CLAMP_TO_EDGE;
                    if (!mask) return GL_CLAMP_TO_EDGE;
                    return (cm & 1) ? GL_MIRRORED_REPEAT : GL_REPEAT;
                };
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap(t.cms, t.masks));
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap(t.cmt, t.maskt));
                stats.textures_created++;
                it = textures.emplace(hash, e).first;
            }
            it->second.last_use = frame_counter;

            r.id = it->second.id;
            r.w = float(w);
            r.h = float(h);
            auto shiftFactor = [](int sh) -> float {
                if (sh == 0) return 1.0f;
                if (sh <= 10) return 1.0f / float(1 << sh);
                return float(1 << (16 - sh));
            };
            r.mul[0] = shiftFactor(t.shifts) / r.w;
            r.mul[1] = shiftFactor(t.shiftt) / r.h;
            r.off[0] = (float(t.uls) / 4.0f) / r.w;
            r.off[1] = (float(t.ult) / 4.0f) / r.h;
        }

        void decode_tile(const Tile &t, int w, int h, int row_bytes, uint32_t tlut_mode, uint32_t *out) {
            const int tbase = t.tmem * 8;
            auto expand5 = [](uint32_t v) { return uint8_t((v << 3) | (v >> 2)); };
            auto rgba = [](uint32_t r, uint32_t g, uint32_t b, uint32_t a) { return r | (g << 8) | (b << 16) | (a << 24); };
            auto fromRGBA16 = [&](uint16_t v) {
                return rgba(expand5((v >> 11) & 31), expand5((v >> 6) & 31), expand5((v >> 1) & 31), (v & 1) ? 255 : 0);
            };
            auto fromIA16 = [&](uint16_t v) {
                uint32_t i = v >> 8, a = v & 255;
                return rgba(i, i, i, a);
            };
            auto palette = [&](int index) -> uint32_t {
                int a = 0x800 + (index & 255) * 8;
                uint16_t v = uint16_t((tm(a) << 8) | tm(a + 1));
                return tlut_mode == 3 ? fromIA16(v) : fromRGBA16(v);
            };
            for (int y = 0; y < h; y++) {
                const int swap = (y & 1) ? 4 : 0;
                const int row = tbase + y * row_bytes;
                for (int x = 0; x < w; x++) {
                    uint32_t c = 0;
                    switch (t.siz) {
                        case 0: {
                            int a = (row + (x >> 1)) ^ swap;
                            uint8_t b = tm(a);
                            uint8_t n = (x & 1) ? (b & 15) : (b >> 4);
                            if (t.fmt == 2) c = palette(t.pal * 16 + n);
                            else if (t.fmt == 3) {
                                uint32_t i = (n >> 1) & 7;
                                uint8_t i8 = uint8_t((i << 5) | (i << 2) | (i >> 1));
                                c = rgba(i8, i8, i8, (n & 1) ? 255 : 0);
                            } else { uint32_t i = n * 17; c = rgba(i, i, i, i); }
                            break;
                        }
                        case 1: {
                            int a = (row + x) ^ swap;
                            uint8_t b = tm(a);
                            if (t.fmt == 2) c = palette(b);
                            else if (t.fmt == 3) {
                                uint32_t i = (b >> 4) * 17, al = (b & 15) * 17;
                                c = rgba(i, i, i, al);
                            } else c = rgba(b, b, b, b);
                            break;
                        }
                        case 2: {
                            int a = (row + x * 2) ^ swap;
                            uint16_t v = uint16_t((tm(a) << 8) | tm(a + 1));
                            if (t.fmt == 3) c = fromIA16(v);
                            else c = fromRGBA16(v);
                            break;
                        }
                        default: {
                            int a = (row + x * 2) ^ swap;
                            c = rgba(tm(a), tm(a + 1), tm(a + 0x800), tm(a + 0x801));
                            break;
                        }
                    }
                    out[size_t(y) * w + x] = c;
                }
            }
        }

        // -------------------------------------------------------------------------------------------------------
        // TMEM loads
        // -------------------------------------------------------------------------------------------------------

        uint32_t timg_phys() const { return timg_addr & kPhysMask; }

        void load_block(int ti, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t dxt) {
            const Tile &t = tiles[ti];
            const int bits = bitsPerTexel(t.siz);
            const int src_bits = bitsPerTexel(timg_siz);
            uint32_t texels = lrs + 1;
            uint32_t src = timg_phys() + uint32_t(((uint64_t(ult) * timg_width + uls) * src_bits) / 8);
            if (t.siz == 3) {
                uint32_t words = (texels + 1) / 2;
                uint32_t acc = 0;
                for (uint32_t j = 0; j < words; j++) {
                    int swap = ((acc >> 11) & 1) ? 4 : 0;
                    for (int k = 0; k < 2; k++) {
                        uint32_t a = src + j * 8 + k * 4;
                        int dst = (t.tmem * 8 + int(j) * 4 + k * 2) ^ swap;
                        tmem[(dst) & 0x7FF] = mem.u8(a);
                        tmem[(dst + 1) & 0x7FF] = mem.u8(a + 1);
                        tmem[((dst) & 0x7FF) + 0x800] = mem.u8(a + 2);
                        tmem[(((dst + 1) & 0x7FF)) + 0x800] = mem.u8(a + 3);
                    }
                    acc += dxt;
                }
            } else {
                uint32_t bytes = (texels * uint32_t(bits) + 7) / 8;
                uint32_t words = (bytes + 7) / 8;
                uint32_t acc = 0;
                for (uint32_t j = 0; j < words; j++) {
                    int swap = ((acc >> 11) & 1) ? 4 : 0;
                    for (int k = 0; k < 8; k++) {
                        int dst = (t.tmem * 8 + int(j) * 8 + k) ^ swap;
                        tmem[dst & (kTmemSize - 1)] = mem.u8(src + j * 8 + k);
                    }
                    acc += dxt;
                }
            }
            tex_version++;
        }

        void load_tile(int ti, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t lrt) {
            const Tile &t = tiles[ti];
            const int src_bits = bitsPerTexel(timg_siz);
            int w = int(((lrs - uls) >> 2) + 1);
            int h = int(((lrt - ult) >> 2) + 1);
            const uint32_t s0 = uls >> 2, t0 = ult >> 2;
            const int row_bytes = t.line * 8;
            for (int y = 0; y < h; y++) {
                const int swap = (y & 1) ? 4 : 0;
                uint32_t src = timg_phys() + uint32_t(((uint64_t(t0 + y) * timg_width + s0) * src_bits) / 8);
                if (t.siz == 3) {
                    for (int x = 0; x < w; x++) {
                        uint32_t a = src + uint32_t(x) * 4;
                        int dst = (t.tmem * 8 + y * row_bytes + x * 2) ^ swap;
                        tmem[dst & 0x7FF] = mem.u8(a);
                        tmem[(dst + 1) & 0x7FF] = mem.u8(a + 1);
                        tmem[(dst & 0x7FF) + 0x800] = mem.u8(a + 2);
                        tmem[((dst + 1) & 0x7FF) + 0x800] = mem.u8(a + 3);
                    }
                } else {
                    int bytes = (w * src_bits + 7) / 8;
                    for (int i = 0; i < bytes; i++) {
                        int dst = (t.tmem * 8 + y * row_bytes + i) ^ swap;
                        tmem[dst & (kTmemSize - 1)] = mem.u8(src + uint32_t(i));
                    }
                }
            }
            tex_version++;
        }

        void load_tlut(int ti, uint32_t uls, uint32_t lrs) {
            const Tile &t = tiles[ti];
            int count = int(((lrs - uls) >> 2) + 1);
            uint32_t src = timg_phys() + (uls >> 2) * 2;
            for (int i = 0; i < count; i++) {
                uint8_t hi = mem.u8(src + uint32_t(i) * 2), lo = mem.u8(src + uint32_t(i) * 2 + 1);
                int dst = t.tmem * 8 + i * 8;
                for (int k = 0; k < 4; k++) {
                    tmem[(dst + k * 2) & (kTmemSize - 1)] = hi;
                    tmem[(dst + k * 2 + 1) & (kTmemSize - 1)] = lo;
                }
            }
            tex_version++;
        }
        // -------------------------------------------------------------------------------------------------------
        // programs & draw state
        // -------------------------------------------------------------------------------------------------------

        const Program *get_program(const ProgKey &key, bool &use0, bool &use1) {
            auto it = programs.find(key);
            if (it != programs.end()) {
                use0 = it->second.use0;
                use1 = it->second.use1;
                return it->second.id ? &it->second : nullptr;
            }
            bool u0 = false, u1 = false;
            std::string fs = buildFragmentShader(key, u0, u1);
            use0 = u0;
            use1 = u1;
            Program p;
            p.use0 = u0;
            p.use1 = u1;
            const char *attribs[3] = {"aPos", "aColor", "aST"};
            p.id = gles_build_program(kVertexShader, fs.c_str(), attribs, 3);
            if (p.id) {
                p.u_prim = glGetUniformLocation(p.id, "uPrim");
                p.u_env = glGetUniformLocation(p.id, "uEnv");
                p.u_fog = glGetUniformLocation(p.id, "uFog");
                p.u_blend = glGetUniformLocation(p.id, "uBlend");
                p.u_misc = glGetUniformLocation(p.id, "uMisc");
                p.u_tile0 = glGetUniformLocation(p.id, "uTile0");
                p.u_tile1 = glGetUniformLocation(p.id, "uTile1");
                p.u_flip = glGetUniformLocation(p.id, "uFlipY");
                p.u_tex0 = glGetUniformLocation(p.id, "uTex0");
                p.u_tex1 = glGetUniformLocation(p.id, "uTex1");
                stats.programs_created++;
                if (diag_prog < 40) {
                    diag_prog++;
                    std::fprintf(stderr, "Mali64: program #%u: combine %06x/%08x blender %08x cycle %u flags %08x\n", diag_prog, key.comb0, key.comb1, key.blender, key.cycle, key.flags);
                }
            } else {
                std::fprintf(stderr, "Mali64: shader failed for combine %06x/%08x mode %08x cycle %u\n", key.comb0, key.comb1,
                             key.blender, key.cycle);
            }
            auto ins = programs.emplace(key, p);
            return ins.first->second.id ? &ins.first->second : nullptr;
        }

        static void unpack(uint32_t c, float out[4]) {
            out[0] = float((c >> 24) & 255) / 255.0f;
            out[1] = float((c >> 16) & 255) / 255.0f;
            out[2] = float((c >> 8) & 255) / 255.0f;
            out[3] = float(c & 255) / 255.0f;
        }

        int cycle_type() const { return int((mode_h >> 20) & 3); }

        void build_key(int kind) {
            DrawKey k;
            k.kind = uint32_t(kind);
            k.comb0 = comb0;
            k.comb1 = comb1;
            k.mode_h = mode_h;
            k.mode_l = mode_l;
            k.target = cur_target ? cur_target->addr : 0;
            const int cycle = cycle_type();

            unpack(prim_color, k.prim);
            unpack(env_color, k.env);
            unpack(fog_color, k.fog);
            unpack(blend_color, k.blend);
            k.misc[0] = float(prim_lod_frac) / 255.0f;
            if (kind == 1) {
                float c[4];
                if (cimg_siz == 3) {
                    unpack(fill_color, c);
                } else {
                    uint16_t v = uint16_t(fill_color >> 16);
                    c[0] = float((v >> 11) & 31) / 31.0f;
                    c[1] = float((v >> 6) & 31) / 31.0f;
                    c[2] = float((v >> 1) & 31) / 31.0f;
                    c[3] = (v & 1) ? 1.0f : 0.0f;
                }
                std::memcpy(k.prim, c, sizeof(c));
            }

            int vx = int(std::lround(vp_trans[0] - std::fabs(vp_scale[0])));
            int vw = int(std::lround(2.0f * std::fabs(vp_scale[0])));
            int vh = int(std::lround(2.0f * std::fabs(vp_scale[1])));
            int vy = int(std::lround(float(240) - (vp_trans[1] + std::fabs(vp_scale[1]))));
            k.viewport[0] = vx; k.viewport[1] = vy; k.viewport[2] = std::max(vw, 1); k.viewport[3] = std::max(vh, 1);
            k.flip_y = vp_scale[1] < 0 ? -1.0f : 1.0f;
            float zn = clampf((vp_trans[2] - std::fabs(vp_scale[2])) / 1023.0f, 0.0f, 1.0f);
            float zf = clampf((vp_trans[2] + std::fabs(vp_scale[2])) / 1023.0f, 0.0f, 1.0f);
            k.depth_range[0] = zn;
            k.depth_range[1] = zf;
            if (kind != 0) {
                k.viewport[0] = 0; k.viewport[1] = 0; k.viewport[2] = 320; k.viewport[3] = 240;
                k.flip_y = 1.0f;
                k.depth_range[0] = 0.0f;
                k.depth_range[1] = 1.0f;
            }

            int sx = scissor[0] >> 2, sy = scissor[1] >> 2;
            int sw = (scissor[2] >> 2) - sx, sh = (scissor[3] >> 2) - sy;
            k.scissor[0] = sx;
            k.scissor[1] = 240 - (sy + sh);
            k.scissor[2] = std::max(sw, 0);
            k.scissor[3] = std::max(sh, 0);

            const uint32_t zmode = (mode_l >> OML_ZMODE_SHIFT) & 3;
            const bool zbuf_geom = (geom & GM_ZBUFFER) != 0;
            if (kind != 0 || cycle >= CYC_COPY) {
                k.depth_test = 0;
                k.depth_write = 0;
            } else {
                k.depth_test = (zbuf_geom && (mode_l & OML_Z_CMP)) ? 1 : 0;
                k.depth_write = (zbuf_geom && (mode_l & OML_Z_UPD)) ? 1 : 0;
            }
            k.depth_decal = zmode == 3;

            k.blend_on = 0;
            if (kind != 1 && cycle < CYC_COPY && (mode_l & OML_FORCE_BL)) {
                const int m2a = int((mode_l >> 20) & 3), m2b = int((mode_l >> 16) & 3);
                if (m2a == 1) {
                    k.blend_src = GL_SRC_ALPHA;
                    switch (m2b) {
                        case 0: k.blend_dst = GL_ONE_MINUS_SRC_ALPHA; k.blend_on = 1; break;
                        case 1: k.blend_dst = GL_ONE_MINUS_SRC_ALPHA; k.blend_on = 1; break;
                        case 2: k.blend_dst = GL_ONE; k.blend_on = 1; break;
                        default: break;
                    }
                }
            }

            if (kind != 1 && cycle != CYC_FILL) {
                int base = tex_tile;
                ProgKey pk{};
                pk.comb0 = comb0;
                pk.comb1 = comb1;
                pk.blender = mode_l;
                pk.cycle = uint32_t(cycle);
                pk.flags = alpha_mode_flags();
                bool u0 = false, u1 = false;
                combinerTextureUse(comb0, comb1, cycle, u0, u1);
                if (cycle == CYC_COPY) { u0 = true; u1 = false; }
                if (u0) {
                    resolve_tile(base & 7);
                    k.tex0 = resolved[base & 7].id;
                    set_tile_uniform(k.tile0, resolved[base & 7]);
                }
                if (u1) {
                    resolve_tile((base + 1) & 7);
                    k.tex1 = resolved[(base + 1) & 7].id;
                    set_tile_uniform(k.tile1, resolved[(base + 1) & 7]);
                }
            }
            cur_key = k;
            key_dirty = false;
        }

        static void set_tile_uniform(float out[4], const Resolved &r) {
            out[0] = r.mul[0];
            out[1] = r.mul[1];
            out[2] = r.off[0];
            out[3] = r.off[1];
        }

        uint32_t alpha_mode_flags() const {
            const bool force_bl = mode_l & OML_FORCE_BL;
            if (mode_l & 1) return 1;
            if ((mode_l & OML_CVG_X_ALPHA) && !force_bl) return 2;
            return 0;
        }

        void use_key(int kind) {
            if (key_dirty || cur_key.kind != uint32_t(kind)) build_key(kind);
            if (batch_has_key && std::memcmp(&cur_key, &batch_key, sizeof(DrawKey)) != 0) flush();
            if (!batch_has_key) {
                batch_key = cur_key;
                batch_has_key = true;
            }
        }

        void flush() {
            if (batch.empty() || !batch_has_key) {
                batch.clear();
                batch_has_key = false;
                return;
            }
            const DrawKey &k = batch_key;
            Target *tgt = nullptr;
            for (auto &t : targets)
                if (t.addr == k.target) tgt = &t;
                if (tgt == nullptr) {
                    batch.clear();
                    batch_has_key = false;
                    return;
                }

                ProgKey pk{};
            pk.comb0 = k.comb0;
            pk.comb1 = k.comb1;
            pk.blender = k.mode_l;
            pk.cycle = uint32_t((k.mode_h >> 20) & 3);
            pk.flags = (k.kind == 1) ? 0 : ((k.mode_l & 1) ? 1 : (((k.mode_l & OML_CVG_X_ALPHA) && !(k.mode_l & OML_FORCE_BL)) ? 2 : 0));
            if (k.kind == 1) pk.cycle = CYC_FILL;
            bool use0 = false, use1 = false;
            const Program *prog = get_program(pk, use0, use1);
            if (prog == nullptr) {
                batch.clear();
                batch_has_key = false;
                return;
            }

            glBindFramebuffer(GL_FRAMEBUFFER, tgt->fbo);
            glViewport(k.viewport[0], k.viewport[1], k.viewport[2], k.viewport[3]);
            glEnable(GL_SCISSOR_TEST);
            glScissor(k.scissor[0], k.scissor[1], k.scissor[2], k.scissor[3]);
            glDepthRangef(k.depth_range[0], k.depth_range[1]);
            if (k.depth_test) {
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LEQUAL);
            } else {
                glDisable(GL_DEPTH_TEST);
            }
            glDepthMask(k.depth_write ? GL_TRUE : GL_FALSE);
            if (k.depth_decal) {
                glEnable(GL_POLYGON_OFFSET_FILL);
                glPolygonOffset(-2.0f, -2.0f);
            } else {
                glDisable(GL_POLYGON_OFFSET_FILL);
            }
            if (k.blend_on) {
                glEnable(GL_BLEND);
                glBlendFunc(k.blend_src, k.blend_dst);
            } else {
                glDisable(GL_BLEND);
            }
            glDisable(GL_CULL_FACE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            glUseProgram(prog->id);
            glUniform4fv(prog->u_prim, 1, k.prim);
            glUniform4fv(prog->u_env, 1, k.env);
            glUniform4fv(prog->u_fog, 1, k.fog);
            glUniform4fv(prog->u_blend, 1, k.blend);
            glUniform4fv(prog->u_misc, 1, k.misc);
            glUniform4fv(prog->u_tile0, 1, k.tile0);
            glUniform4fv(prog->u_tile1, 1, k.tile1);
            glUniform1f(prog->u_flip, k.flip_y);
            glUniform1i(prog->u_tex0, 0);
            glUniform1i(prog->u_tex1, 1);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, use0 ? k.tex0 : 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, use1 ? k.tex1 : 0);
            glActiveTexture(GL_TEXTURE0);

            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            // [FIX 3] Uso direto do glBufferData pois glBufferSubData não está no gles_api
            glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(batch.size() * sizeof(DVert)), batch.data(), GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, GLsizei(batch.size()));
            stats.draw_calls++;
            batch.clear();
            batch_has_key = false;

            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthMask(GL_TRUE);
        }

        // -------------------------------------------------------------------------------------------------------
        // RSP commands
        // -------------------------------------------------------------------------------------------------------

        uint32_t phys(uint32_t a) const {
            if (a & 0x80000000u) return a & kPhysMask;
            return (segments[(a >> 24) & 0xF] + (a & 0x00FFFFFF)) & kPhysMask;
        }

        Mat load_matrix(uint32_t addr) const {
            Mat m{};
            for (int i = 0; i < 16; i++) {
                int16_t ip = mem.s16(addr + uint32_t(i) * 2);
                uint16_t fp = mem.u16(addr + 32 + uint32_t(i) * 2);
                m.m[i / 4][i % 4] = float(ip) + float(fp) / 65536.0f;
            }
            return m;
        }

        void cmd_mtx(uint32_t w0, uint32_t w1) {
            const uint8_t params = (w0 >> 16) & 0xFF;
            const bool is_proj = params & 0x01;
            const bool load = params & 0x02;
            const bool push = params & 0x04;
            Mat m = load_matrix(phys(w1));
            if (is_proj) {
                proj = load ? m : mul(m, proj);
            } else {
                if (push && mv_depth < 15) {
                    mv_stack[mv_depth + 1] = mv_stack[mv_depth];
                    mv_depth++;
                }
                mv_stack[mv_depth] = load ? m : mul(m, mv_stack[mv_depth]);
            }
            mvp_dirty = true;
        }

        void update_mvp() {
            if (mvp_dirty) {
                mvp = mul(mv_stack[mv_depth], proj);
                mvp_dirty = false;
            }
        }

        void cmd_movemem(uint32_t w0, uint32_t w1) {
            const uint8_t idx = (w0 >> 16) & 0xFF;
            const uint32_t a = phys(w1);
            if (idx == 0x80) {
                vp_scale[0] = float(mem.s16(a)) / 4.0f;
                vp_scale[1] = float(mem.s16(a + 2)) / 4.0f;
                vp_scale[2] = float(mem.s16(a + 4));
                vp_trans[0] = float(mem.s16(a + 8)) / 4.0f;
                vp_trans[1] = float(mem.s16(a + 10)) / 4.0f;
                vp_trans[2] = float(mem.s16(a + 12));
                key_dirty = true;
            } else if (idx >= 0x86 && idx <= 0x94 && ((idx - 0x86) % 2) == 0) {
                int n = (idx - 0x86) / 2;
                Light &l = lights[n];
                for (int i = 0; i < 3; i++) {
                    l.col[i] = mem.u8(a + uint32_t(i));
                    l.dir[i] = mem.s8(a + 8 + uint32_t(i));
                }
            } else if (idx == 0x82 || idx == 0x84) {
                float *dst = (idx == 0x84) ? lookat_x : lookat_y;
                float v[3] = {float(mem.s8(a + 8)), float(mem.s8(a + 9)), float(mem.s8(a + 10))};
                float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                for (int i = 0; i < 3; i++) dst[i] = len > 0 ? v[i] / len : 0;
            } else if (idx == 0x9E) {
                mv_stack[mv_depth] = load_matrix(a);
                mvp_dirty = true;
            }
        }

        void cmd_moveword(uint32_t w0, uint32_t w1) {
            const uint8_t type = w0 & 0xFF;
            switch (type) {
                case 0x02:
                    num_lights = int(((w1 - 0x80000000u) >> 5)) - 1;
                    num_lights = std::max(0, std::min(num_lights, kMaxLights));
                    break;
                case 0x06:
                    segments[(w0 >> 10) & 0xF] = w1 & kPhysMask;
                    break;
                case 0x08:
                    fog_mul = int16_t(w1 >> 16);
                    fog_off = int16_t(w1 & 0xFFFF);
                    break;
                default: break;
            }
        }

        void cmd_vertex(uint32_t w0, uint32_t w1) {
            const int n = int((w0 >> 10) & 0x3F);
            const int dst = int((w0 >> 17) & 0x7F);
            if (dst + n > kMaxVertices) return;
            update_mvp();
            const uint32_t a = phys(w1);
            const bool lighting = geom & GM_LIGHTING;
            const bool texgen = lighting && (geom & GM_TEXTURE_GEN);
            const float scale_s = float(tex_scale_s) / 65536.0f;
            const float scale_t = float(tex_scale_t) / 65536.0f;
            const Mat &mv = mv_stack[mv_depth];
            for (int i = 0; i < n; i++) {
                const uint32_t va = a + uint32_t(i) * 16;
                const float x = float(mem.s16(va)), y = float(mem.s16(va + 2)), z = float(mem.s16(va + 4));
                VtxOut &o = vtx[dst + i];
                for (int c = 0; c < 4; c++) {
                    o.clip[c] = x * mvp.m[0][c] + y * mvp.m[1][c] + z * mvp.m[2][c] + mvp.m[3][c];
                }
                float s_fixed = float(mem.s16(va + 8)), t_fixed = float(mem.s16(va + 10));
                float r, g, b;
                const uint8_t ca = mem.u8(va + 15);
                float alpha = float(ca);
                if (lighting) {
                    const float nx = float(mem.s8(va + 12)) / 127.0f;
                    const float ny = float(mem.s8(va + 13)) / 127.0f;
                    const float nz = float(mem.s8(va + 14)) / 127.0f;
                    float vn[3];
                    for (int c = 0; c < 3; c++) vn[c] = nx * mv.m[0][c] + ny * mv.m[1][c] + nz * mv.m[2][c];
                    r = lights[num_lights].col[0];
                    g = lights[num_lights].col[1];
                    b = lights[num_lights].col[2];
                    for (int l = 0; l < num_lights; l++) {
                        const Light &L = lights[l];
                        float d = (vn[0] * float(L.dir[0]) + vn[1] * float(L.dir[1]) + vn[2] * float(L.dir[2])) / 127.0f;
                        if (d > 0.0f) {
                            r += d * L.col[0];
                            g += d * L.col[1];
                            b += d * L.col[2];
                        }
                    }
                    r = std::min(r, 255.0f);
                    g = std::min(g, 255.0f);
                    b = std::min(b, 255.0f);
                    if (texgen) {
                        float dx = vn[0] * lookat_x[0] + vn[1] * lookat_x[1] + vn[2] * lookat_x[2];
                        float dy = vn[0] * lookat_y[0] + vn[1] * lookat_y[1] + vn[2] * lookat_y[2];
                        dx = clampf(dx, -1.0f, 1.0f);
                        dy = clampf(dy, -1.0f, 1.0f);
                        if (geom & GM_TEXTURE_GEN_LINEAR) {
                            s_fixed = std::acos(dx) * 325.94931f;
                            t_fixed = std::acos(dy) * 325.94931f;
                        } else {
                            s_fixed = (dx + 1.0f) * 512.0f;
                            t_fixed = (dy + 1.0f) * 512.0f;
                        }
                    }
                } else {
                    r = float(mem.u8(va + 12));
                    g = float(mem.u8(va + 13));
                    b = float(mem.u8(va + 14));
                }
                if (geom & GM_FOG) {
                    float w = o.clip[3];
                    float winv = w != 0.0f ? 1.0f / w : 0.0f;
                    float f = o.clip[2] * winv * float(fog_mul) + float(fog_off);
                    alpha = clampf(f, 0.0f, 255.0f);
                }
                o.rgba[0] = uint8_t(r);
                o.rgba[1] = uint8_t(g);
                o.rgba[2] = uint8_t(b);
                o.rgba[3] = uint8_t(alpha);
                o.s = s_fixed * scale_s / 32.0f;
                o.t = t_fixed * scale_t / 32.0f;
            }
        }

        void cmd_modify_vertex(uint32_t w0, uint32_t w1) {
            const int index = int((w0 >> 1) & 0x7FFF);
            const int what = int((w0 >> 16) & 0xFF);
            if (index >= kMaxVertices) return;
            VtxOut &o = vtx[index];
            if (what == 0x14) {
                o.s = float(int16_t(w1 >> 16)) / 32.0f;
                o.t = float(int16_t(w1 & 0xFFFF)) / 32.0f;
            } else if (what == 0x10) {
                o.rgba[0] = uint8_t(w1 >> 24);
                o.rgba[1] = uint8_t(w1 >> 16);
                o.rgba[2] = uint8_t(w1 >> 8);
                o.rgba[3] = uint8_t(w1);
            }
        }

        void push_vertex(const VtxOut &v, const uint8_t rgba[4]) {
            DVert d;
            d.x = v.clip[0]; d.y = v.clip[1]; d.z = v.clip[2]; d.w = v.clip[3];
            d.r = rgba[0]; d.g = rgba[1]; d.b = rgba[2]; d.a = rgba[3];
            d.s = v.s; d.t = v.t;
            batch.push_back(d);
        }

        void draw_triangle(int a, int b, int c, int provoking) {
            if (a >= kMaxVertices || b >= kMaxVertices || c >= kMaxVertices) return;
            const uint32_t cull = geom & (GM_CULL_FRONT | GM_CULL_BACK);
            if (cull == (GM_CULL_FRONT | GM_CULL_BACK)) return;
            const VtxOut &va = vtx[a], &vb = vtx[b], &vc = vtx[c];

            if (cull && va.clip[3] > 0 && vb.clip[3] > 0 && vc.clip[3] > 0) {
                float x0 = va.clip[0] / va.clip[3], y0 = va.clip[1] / va.clip[3];
                float x1 = vb.clip[0] / vb.clip[3], y1 = vb.clip[1] / vb.clip[3];
                float x2 = vc.clip[0] / vc.clip[3], y2 = vc.clip[1] / vc.clip[3];
                float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
                if (cull_flip) area = -area;
                if (area == 0.0f) return;
                if ((area < 0 && (cull & GM_CULL_BACK)) || (area > 0 && (cull & GM_CULL_FRONT))) return;
            }

            use_key(0);

            const VtxOut *tri[3] = {&va, &vb, &vc};
            const VtxOut &flat = *tri[std::max(0, std::min(provoking, 2))];
            const bool smooth = geom & GM_SHADING_SMOOTH;
            for (int i = 0; i < 3; i++) {
                const uint8_t *col = smooth ? tri[i]->rgba : flat.rgba;
                push_vertex(*tri[i], col);
            }
            stats.triangles++;
            if (batch.size() > 30000) flush();
        }

        void emit_rect(float x0, float y0, float x1, float y1, const float ss[4], const float tt[4]) {
            auto cx = [](float px) { return px / float(320) * 2.0f - 1.0f; };
            auto cy = [](float py) { return 1.0f - py / float(240) * 2.0f; };
            VtxOut v[4];
            const float xs[4] = {x0, x1, x1, x0};
            const float ys[4] = {y0, y0, y1, y1};
            for (int i = 0; i < 4; i++) {
                v[i].clip[0] = cx(xs[i]);
                v[i].clip[1] = cy(ys[i]);
                v[i].clip[2] = 0.0f;
                v[i].clip[3] = 1.0f;
                v[i].s = ss[i];
                v[i].t = tt[i];
            }
            static const uint8_t white[4] = {255, 255, 255, 255};
            static const int order[6] = {0, 1, 2, 0, 2, 3};
            for (int i : order) push_vertex(v[i], white);
            stats.triangles += 2;
        }

        void fillrect(int ulx, int uly, int lrx, int lry) {
            if (cimg_is_z) {
                n_fill_z++;
                const uint16_t z16 = uint16_t(fill_color >> 16);
                z_clear_value = float(z16 >> 2) / 16383.0f;
                z_clear_pending = true;
                return;
            }
            n_fillrect++;
            const int cyc = cycle_type();
            float x0 = float(ulx) / 4.0f, y0 = float(uly) / 4.0f, x1 = float(lrx) / 4.0f, y1 = float(lry) / 4.0f;
            if (cyc >= CYC_COPY) {
                x1 += 1.0f;
                y1 += 1.0f;
            }
            const float zero[4] = {0, 0, 0, 0};
            use_key(cyc == CYC_FILL ? 1 : 2);
            emit_rect(x0, y0, x1, y1, zero, zero);
        }

        void texrect(int ulx, int uly, int lrx, int lry, int tile, uint32_t half1_w1, uint32_t half2_w1, bool flip) {
            n_texrect++;
            const float s0 = float(int16_t(half1_w1 >> 16)) / 32.0f;
            const float t0 = float(int16_t(half1_w1 & 0xFFFF)) / 32.0f;
            float ds = float(int16_t(half2_w1 >> 16)) / 1024.0f;
            float dt = float(int16_t(half2_w1 & 0xFFFF)) / 1024.0f;
            const int cyc = cycle_type();
            float x0 = float(ulx) / 4.0f, y0 = float(uly) / 4.0f, x1 = float(lrx) / 4.0f, y1 = float(lry) / 4.0f;
            if (cyc >= CYC_COPY) {
                x1 += 1.0f;
                y1 += 1.0f;
                ds *= 0.25f;
            }
            const float w = x1 - x0, h = y1 - y0;
            float ss[4], tt[4];
            if (!flip) {
                ss[0] = s0;           ss[1] = s0 + w * ds; ss[2] = s0 + w * ds; ss[3] = s0;
                tt[0] = t0;           tt[1] = t0;          tt[2] = t0 + h * dt; tt[3] = t0 + h * dt;
            } else {
                ss[0] = s0;           ss[1] = s0;          ss[2] = s0 + h * dt; ss[3] = s0 + h * dt;
                tt[0] = t0;           tt[1] = t0 + w * ds; tt[2] = t0 + w * ds; tt[3] = t0;
            }
            const uint8_t saved_tile = tex_tile;
            tex_tile = uint8_t(tile);
            key_dirty = true;
            use_key(2);
            emit_rect(x0, y0, x1, y1, ss, tt);
            tex_tile = saved_tile;
            key_dirty = true;
        }

        // -------------------------------------------------------------------------------------------------------
        // RDP state commands
        // -------------------------------------------------------------------------------------------------------

        bool cimg_is_z = false;
        bool z_clear_pending = false;
        float z_clear_value = 1.0f;

        void cmd_setcimg(uint32_t w0, uint32_t w1) {
            const uint32_t addr = phys(w1);
            if (diag_setcimg < 40) {
                diag_setcimg++;
                std::fprintf(stderr, "Mali64: SETCIMG %08x (raw %08x) zimg %08x width %u siz %u\n", addr, w1, zimg_addr,
                             (w0 & 0xFFF) + 1, (w0 >> 19) & 3);
            }
            if (zimg_addr != 0 && addr == zimg_addr) {
                cimg_is_z = true;
                return;
            }
            cimg_is_z = false;
            cimg_siz = uint8_t((w0 >> 19) & 3);
            cimg_addr = addr;
            select_target(addr);
        }

        void cmd_settile(uint32_t w0, uint32_t w1) {
            Tile &t = tiles[(w1 >> 24) & 7];
            t.fmt = uint8_t((w0 >> 21) & 7);
            t.siz = uint8_t((w0 >> 19) & 3);
            t.line = uint16_t((w0 >> 9) & 0x1FF);
            t.tmem = uint16_t(w0 & 0x1FF);
            t.pal = uint8_t((w1 >> 20) & 0xF);
            t.cmt = uint8_t((w1 >> 18) & 3);
            t.maskt = uint8_t((w1 >> 14) & 0xF);
            t.shiftt = uint8_t((w1 >> 10) & 0xF);
            t.cms = uint8_t((w1 >> 8) & 3);
            t.masks = uint8_t((w1 >> 4) & 0xF);
            t.shifts = uint8_t(w1 & 0xF);
            t.size_set = false;
            tex_version++;
        }

        void cmd_settilesize(uint32_t w0, uint32_t w1) {
            Tile &t = tiles[(w1 >> 24) & 7];
            t.uls = uint16_t((w0 >> 12) & 0xFFF);
            t.ult = uint16_t(w0 & 0xFFF);
            t.lrs = uint16_t((w1 >> 12) & 0xFFF);
            t.lrt = uint16_t(w1 & 0xFFF);
            t.size_set = true;
            tex_version++;
        }

        void set_other_mode(uint32_t &reg, uint32_t w0, uint32_t w1) {
            const uint32_t len = w0 & 0xFF, sft = (w0 >> 8) & 0xFF;
            if (sft >= 32) return;
            const uint32_t mask = len >= 32 ? 0xFFFFFFFFu : (((1u << len) - 1u) << sft);
            reg = (reg & ~mask) | (w1 & mask);
        }

        // -------------------------------------------------------------------------------------------------------
        // display list interpreter
        // -------------------------------------------------------------------------------------------------------

        void detect_ucode(uint32_t data_addr) {
            static const char kTag[] = "RSP Gfx ucode ";
            const size_t tag_len = sizeof(kTag) - 1;
            for (uint32_t off = 0; off + tag_len + 8 < 0x1000; off++) {
                size_t i = 0;
                while (i < tag_len && mem.u8(data_addr + off + uint32_t(i)) == uint8_t(kTag[i])) i++;
                if (i != tag_len) continue;
                char name[sizeof(ucode_name)] = {};
                for (size_t k = 0; k + 1 < sizeof(name); k++) {
                    const uint8_t c = mem.u8(data_addr + off + uint32_t(tag_len + k));
                    if (c < 0x20 || c > 0x7E) break;
                    name[k] = char(c);
                }
                if (std::strncmp(name, ucode_name, sizeof(name)) != 0) {
                    std::memcpy(ucode_name, name, sizeof(name));
                    std::printf("Mali64: microcode \"%s\"\n", ucode_name);
                    std::fflush(stdout);
                }
                return;
            }
        }

        static constexpr int kStackDepth = 18;

        void run_dl(uint32_t start) {
            uint32_t stack[kStackDepth];
            int sp = 0;
            uint32_t pc = start & kPhysMask;
            for (uint32_t count = 0; count < 4000000u; count++) {
                const uint32_t w0 = mem.u32(pc), w1 = mem.u32(pc + 4);
                pc += 8;
                const uint8_t op = uint8_t(w0 >> 24);
                if (trace && trace_budget > 0) {
                    trace_budget--;
                    std::printf("Mali64: %08x: %08x %08x\n", pc - 8, w0, w1);
                }
                switch (op) {
                    case G_SPNOOP:
                        if ((w0 & 0xFFFFFF) == 0x525464) {
                            const uint32_t hook_op = w1 >> 28;
                            const uint32_t value = w1 & 0x0FFFFFFF;
                            if (hook_op == 0) {
                                mem.put32(phys(value), 1);
                            } else if (hook_op == 3 || hook_op == 4) {
                                if (hook_op == 3 && sp < kStackDepth) stack[sp++] = pc;
                                pc = phys(value);
                            }
                        }
                        break;
                    case G_NOOP: case G_CULLDL: case G_RDPLOADSYNC: case G_RDPPIPESYNC:
                    case G_RDPTILESYNC: case G_RDPFULLSYNC: case G_SETKEYGB: case G_SETKEYR:
                    case G_SETCONVERT: case G_SETPRIMDEPTH: case G_LOAD_UCODE:
                        break;

                    case G_MTX: cmd_mtx(w0, w1); break;
                    case G_POPMTX:
                        if (w1 == 0 && mv_depth > 0) {
                            mv_depth--;
                            mvp_dirty = true;
                        }
                        break;
                    case G_MOVEMEM:
                        cmd_movemem(w0, w1);
                        if (((w0 >> 16) & 0xFF) == 0x9E) pc += 24;
                        break;
                    case G_MOVEWORD: cmd_moveword(w0, w1); break;
                    case G_VTX: cmd_vertex(w0, w1); break;
                    case G_MODIFYVTX: cmd_modify_vertex(w0, w1); break;
                    case G_TRI1: {
                        const int flag = int((w1 >> 24) & 0xFF);
                        draw_triangle(int((w1 >> 17) & 0x7F), int((w1 >> 9) & 0x7F), int((w1 >> 1) & 0x7F), flag > 2 ? 0 : flag);
                        break;
                    }
                    case G_TRI2:
                        draw_triangle(int((w0 >> 17) & 0x7F), int((w0 >> 9) & 0x7F), int((w0 >> 1) & 0x7F), 0);
                        draw_triangle(int((w1 >> 17) & 0x7F), int((w1 >> 9) & 0x7F), int((w1 >> 1) & 0x7F), 0);
                        break;
                    case G_QUAD: {
                        const int a = int((w1 >> 25) & 0x7F), b = int((w1 >> 17) & 0x7F), c = int((w1 >> 9) & 0x7F), d = int((w1 >> 1) & 0x7F);
                        draw_triangle(a, b, c, 0);
                        draw_triangle(a, c, d, 0);
                        break;
                    }
                    case G_SETGEOMETRYMODE: geom |= w1; key_dirty = true; break;
                    case G_CLEARGEOMETRYMODE: geom &= ~w1; key_dirty = true; break;
                    case G_TEXTURE:
                        tex_tile = uint8_t((w0 >> 8) & 7);
                        tex_on = (w0 & 0xFF) != 0;
                        tex_scale_s = uint16_t(w1 >> 16);
                        tex_scale_t = uint16_t(w1 & 0xFFFF);
                        key_dirty = true;
                        break;
                    case G_RDPHALF_1: half1 = w1; break;
                    case G_RDPHALF_2: break;

                    case 0x64: { // RT64 Extended GBI (útil para mods)
                        const uint32_t id = w0 & 0xFFFFFF;
                        switch (id) {
                            case 0x03: {
                                const uint32_t a = mem.u32(pc), b = mem.u32(pc + 4);
                                pc += 8;
                                fillrect(int16_t(a >> 16), int16_t(a & 0xFFFF), int16_t(b >> 16), int16_t(b & 0xFFFF));
                                break;
                            }
                            case 0x02: {
                                const int tile = int(w1 & 7);
                                const bool flip = ((w1 >> 7) & 1) != 0;
                                const uint32_t a = mem.u32(pc), b = mem.u32(pc + 4);
                                const uint32_t c = mem.u32(pc + 8), d = mem.u32(pc + 12);
                                pc += 16;
                                texrect(int16_t(a >> 16), int16_t(a & 0xFFFF), int16_t(b >> 16), int16_t(b & 0xFFFF), tile, c, d, flip);
                                break;
                            }
                            case 0x04: case 0x05: case 0x06: case 0x07: case 0x0C: case 0x13: case 0x14: case 0x31: case 0x30:
                                pc += 8;
                                break;
                            case 0x08:
                                pc += 16;
                                break;
                            default: break;
                        }
                        break;
                    }

                            case G_DL: {
                                const uint32_t target = phys(w1);
                                const bool branch = ((w0 >> 16) & 1) != 0;
                                if (!branch) {
                                    if (sp < kStackDepth) stack[sp++] = pc;
                                }
                                pc = target;
                                break;
                            }
                            case G_ENDDL:
                                if (sp == 0) return;
                                pc = stack[--sp];
                    break;
                            case G_BRANCH_Z:
                                pc = phys(half1);
                                break;

                            case G_SETOTHERMODE_L: set_other_mode(mode_l, w0, w1); key_dirty = true; break;
                            case G_SETOTHERMODE_H: set_other_mode(mode_h, w0, w1); tex_version++; key_dirty = true; break;
                            case G_RDPSETOTHERMODE:
                                mode_h = w0 & 0xFFFFFF; mode_l = w1;
                                tex_version++; key_dirty = true;
                                break;
                            case G_SETCOMBINE: comb0 = w0 & 0xFFFFFF; comb1 = w1; key_dirty = true; break;
                            case G_SETPRIMCOLOR: prim_lod_frac = uint8_t(w0 & 0xFF); prim_color = w1; key_dirty = true; break;
                            case G_SETENVCOLOR: env_color = w1; key_dirty = true; break;
                            case G_SETFOGCOLOR: fog_color = w1; key_dirty = true; break;
                            case G_SETBLENDCOLOR: blend_color = w1; key_dirty = true; break;
                            case G_SETFILLCOLOR: fill_color = w1; key_dirty = true; break;
                            case G_SETSCISSOR:
                                scissor[0] = int32_t((w0 >> 12) & 0xFFF);
                                scissor[1] = int32_t(w0 & 0xFFF);
                                scissor[2] = int32_t((w1 >> 12) & 0xFFF);
                                scissor[3] = int32_t(w1 & 0xFFF);
                                key_dirty = true;
                                break;
                            case G_SETCIMG: cmd_setcimg(w0, w1); break;
                            case G_SETZIMG: zimg_addr = phys(w1); break;
                            case G_SETTIMG:
                                timg_fmt = uint8_t((w0 >> 21) & 7);
                                timg_siz = uint8_t((w0 >> 19) & 3);
                                timg_width = uint16_t((w0 & 0xFFF) + 1);
                                timg_addr = phys(w1);
                                break;
                            case G_SETTILE: cmd_settile(w0, w1); key_dirty = true; break;
                            case G_SETTILESIZE: cmd_settilesize(w0, w1); key_dirty = true; break;
                            case G_LOADBLOCK:
                                load_block(int((w1 >> 24) & 7), (w0 >> 12) & 0xFFF, w0 & 0xFFF, (w1 >> 12) & 0xFFF, w1 & 0xFFF);
                                key_dirty = true;
                                break;
                            case G_LOADTILE:
                                load_tile(int((w1 >> 24) & 7), (w0 >> 12) & 0xFFF, w0 & 0xFFF, (w1 >> 12) & 0xFFF, w1 & 0xFFF);
                                key_dirty = true;
                                break;
                            case G_LOADTLUT:
                                load_tlut(int((w1 >> 24) & 7), (w0 >> 12) & 0xFFF, (w1 >> 12) & 0xFFF);
                                key_dirty = true;
                                break;
                            case G_FILLRECT: fillrect(int((w1 >> 12) & 0xFFF), int(w1 & 0xFFF), int((w0 >> 12) & 0xFFF), int(w0 & 0xFFF)); break;
                            case G_TEXRECT:
                            case G_TEXRECTFLIP: {
                                const uint32_t h1 = mem.u32(pc + 4), h2 = mem.u32(pc + 12);
                                pc += 16;
                                texrect(int((w1 >> 12) & 0xFFF), int(w1 & 0xFFF), int((w0 >> 12) & 0xFFF), int(w0 & 0xFFF), int((w1 >> 24) & 7), h1, h2, op == G_TEXRECTFLIP);
                                break;
                            }
                            default:
                                if (!seen_op[op]) {
                                    seen_op[op] = true;
                                    std::fprintf(stderr, "Mali64: unhandled opcode 0x%02X (%08x %08x) at %08x, microcode \"%s\"\n", op, w0, w1, pc - 8, ucode_name);
                                }
                                break;
                }
            }
            std::fprintf(stderr, "Mali64: display list did not end (command limit)\n");
        }

        void run_task(uint8_t *rdram, uint32_t dl_addr, uint32_t ucode_data_addr) {
            if (!ready) return;
            mem.base = rdram;
            stats.tasks++;
            frame_counter++;
            if (stats.tasks == 40 || stats.tasks == 150 || stats.tasks == 400) auto_trace_budget = 220;
            detect_ucode(ucode_data_addr & kPhysMask);
            if (trace) trace_budget = kLogMax;
            reset_rsp();
            run_dl(dl_addr);
            flush();
            if (trace && (stats.tasks % 120) == 0) {
                std::printf("Mali64: tasks %llu, triangles %llu, draw calls %llu, textures %llu, programs %llu\n",
                            (unsigned long long)stats.tasks, (unsigned long long)stats.triangles,
                            (unsigned long long)stats.draw_calls, (unsigned long long)stats.textures_created,
                            (unsigned long long)stats.programs_created);
                std::fflush(stdout);
            }
            if (textures.size() > 2048) {
                for (auto it = textures.begin(); it != textures.end();) {
                    if (frame_counter - it->second.last_use > 600) {
                        glDeleteTextures(1, &it->second.id);
                        it = textures.erase(it);
                    } else {
                        ++it;
                    }
                }
                tex_version++;
            }
        }

        // -------------------------------------------------------------------------------------------------------
        // presentation
        // -------------------------------------------------------------------------------------------------------

        Target *find_target_for_origin(uint32_t origin) {
            const uint32_t want = origin & kPhysMask;

            for (auto &t : targets)
                if (t.addr == want) return &t;

                for (auto &t : targets) {
                    uint32_t sz = t.bytes > 0 ? t.bytes : (uint32_t(320) * 240 * 2);
                    if (want >= t.addr && want < t.addr + sz) {
                        return &t;
                    }
                }

                for (auto &t : targets)
                    if (t.addr == last_target_addr) return &t;

                    if (!targets.empty()) {
                        Target *best = &targets[0];
                        for (auto &t : targets) {
                            if (t.last_use > best->last_use) best = &t;
                        }
                        return best;
                    }
                    return nullptr;
        }

        bool present(int win_w, int win_h, uint32_t vi_origin) {
            if (!ready) return false;
            if (win_w <= 0 || win_h <= 0) return false;

            flush();
            Target *t = find_target_for_origin(vi_origin);
            diag_presents++;
            if (t == nullptr) {
                if ((diag_missing++ % 60) == 0)
                    std::fprintf(stderr, "Mali64: present: no target for VI origin %08x (%zu targets, tasks %llu)\n", vi_origin,
                                 targets.size(), (unsigned long long)stats.tasks);
                    return false;
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthMask(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            // Fundo preto para o letterbox
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            int dw = win_w, dh = win_h;
            if (dw * 3 > dh * 4) dw = dh * 4 / 3;
            else dh = dw * 3 / 4;
            const int dx = (win_w - dw) / 2, dy = (win_h - dh) / 2;
            glViewport(dx, dy, dw, dh);

            glUseProgram(present_prog.id);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, t->color);
            glUniform1i(present_prog.u_tex0, 0);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            glBindVertexArray(present_vao);
            glBindBuffer(GL_ARRAY_BUFFER, present_vbo);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glUseProgram(0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, 0);

            return true;
        }

        void restore_gl_state() {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glUseProgram(0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthMask(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        }

        bool read_framebuffer(uint32_t addr, uint8_t *rgba) {
            Target *t = nullptr;
            if (addr == 0) {
                for (auto &x : targets)
                    if (x.addr == last_target_addr) t = &x;
            } else {
                for (auto &x : targets)
                    if (x.addr == (addr & kPhysMask)) t = &x;
            }
            if (t == nullptr) return false;
            flush();
            std::vector<uint8_t> tmp(size_t(320) * 240 * 4);
            glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
            glPixelStorei(GL_PACK_ALIGNMENT, 4);
            glReadPixels(0, 0, 320, 240, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
            for (int y = 0; y < 240; y++)
                std::memcpy(rgba + size_t(y) * 320 * 4,
                            tmp.data() + size_t(240 - 1 - y) * 320 * 4, size_t(320) * 4);
                return true;
        }
    };

    // ---------------------------------------------------------------------------------------------------------------
    // public interface
    // ---------------------------------------------------------------------------------------------------------------

    Renderer::Renderer() : impl(new Impl()) {}
    Renderer::~Renderer() = default;
    bool Renderer::init() { return impl->init(); }
    void Renderer::shutdown() { impl->shutdown(); }
    void Renderer::run_task(uint8_t *rdram, uint32_t dl_addr, uint32_t ucode_data_addr) { impl->run_task(rdram, dl_addr, ucode_data_addr); }
    bool Renderer::present(int win_w, int win_h, uint32_t vi_origin) { return impl->present(win_w, win_h, vi_origin); }
    void Renderer::restore_gl_state() { impl->restore_gl_state(); }
    bool Renderer::read_framebuffer(uint32_t addr, uint8_t *rgba) { return impl->read_framebuffer(addr, rgba); }
    const Stats &Renderer::stats() const { return impl->stats; }

} // namespace mali64
