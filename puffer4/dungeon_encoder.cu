// Native CUDA CNN encoder for the rotmg dungeon env, matching rotmg_rl DungeonEncoder exactly:
//   grid obs (7,31,31): conv1(7->32,k3,p1)+GELU, conv2(32->32,k3,p1)+GELU, flatten,
//                       grid_fc(32*31*31->256)+GELU
//   scalars (6):        scalar_fc(6->64)+GELU
//   fuse:               concat(256+64=320) -> fuse(320->hidden)+GELU
// Plugged into PufferLib 4.0's native _C backend via create_custom_encoder("dungeon", ...) so native
// training (no --slowly) runs WITH the convolutional encoder. Modeled on the NMMO3 encoder + the
// im2col/cuBLAS conv path in ocean.cu; this file is #included by ocean.cu just before
// create_custom_encoder (so puf_mm*, im2col helpers, Allocator, ConvWeights, etc. are in scope).
//
// The encoder is the first layer, so encoder_backward returns no input gradient — but the internal
// chain (fuse<-grid_fc<-conv2<-conv1, + scalar_fc) is fully backpropagated for the weight grads.
#ifndef DUNGEON_ENCODER_CU
#define DUNGEON_ENCODER_CU

// Architecture constants (must match src/rotmg_rl/csim/dungeon.h + puffer4/dungeon_encoder.py)
#define DG_IC 7
#define DG_GRID 31
#define DG_SCALARS 6
#define DG_GRID_FLAT (DG_IC * DG_GRID * DG_GRID)        // 6727
#define DG_C1 32
#define DG_C2 32
#define DG_K 3
#define DG_PAD 1
#define DG_SP (DG_GRID * DG_GRID)                        // 961
#define DG_CONV_FLAT (DG_C2 * DG_SP)                     // 30752
#define DG_FC 256
#define DG_SFC 64
#define DG_CONCAT (DG_FC + DG_SFC)                       // 320
#define DG_COL1 (DG_IC * DG_K * DG_K)                    // 63
#define DG_COL2 (DG_C1 * DG_K * DG_K)                    // 288

__device__ __forceinline__ float dg_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}
__device__ __forceinline__ float dg_gelu_grad(float x) {
    // d/dx [0.5 x (1+erf(x/sqrt2))] = 0.5(1+erf(x/sqrt2)) + x * (1/sqrt(2pi)) exp(-x^2/2)
    return 0.5f * (1.0f + erff(x * 0.7071067811865476f)) + x * 0.3989422804014327f * expf(-0.5f * x * x);
}

// ---- elementwise kernels ----
__global__ void dg_bias_gelu(precision_t* z, precision_t* a, const precision_t* bias, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = to_float(z[i]) + to_float(bias[i % dim]);
    z[i] = from_float(v);            // pre-gelu (post-bias), saved for backward
    a[i] = from_float(dg_gelu(v));   // post-gelu activation
}
// NCHW variant: bias indexed by output channel
__global__ void dg_bias_gelu_nchw(precision_t* z, precision_t* a, const precision_t* bias, int n, int OC, int spatial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = to_float(z[i]) + to_float(bias[(i / spatial) % OC]);
    z[i] = from_float(v);
    a[i] = from_float(dg_gelu(v));
}
// g (holds dA) <- dZ = dA * gelu'(z), in place
__global__ void dg_gelu_bwd(precision_t* g, const precision_t* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    g[i] = from_float(to_float(g[i]) * dg_gelu_grad(to_float(z[i])));
}
// bias grad for a [B, dim] linear: one block per output column, sum over batch
__global__ void dg_bias_grad_lin(precision_t* bgrad, const precision_t* g, int B, int dim) {
    int col = blockIdx.x;
    if (col >= dim) return;
    float s = 0.0f;
    for (int b = threadIdx.x; b < B; b += blockDim.x) s += to_float(g[b * dim + col]);
    __shared__ float red[256];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) bgrad[col] = from_float(red[0]);
}
// bias grad for NCHW conv [B, OC, spatial]: one block per channel, sum over batch*spatial
__global__ void dg_conv_bias_grad(precision_t* bgrad, const precision_t* g, int B, int OC, int spatial) {
    int oc = blockIdx.x;
    if (oc >= OC) return;
    float s = 0.0f;
    for (int b = 0; b < B; b++)
        for (int p = threadIdx.x; p < spatial; p += blockDim.x)
            s += to_float(g[(b * OC + oc) * spatial + p]);
    __shared__ float red[256];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) bgrad[oc] = from_float(red[0]);
}
// gather the 6 scalar features (at offset DG_GRID_FLAT of each obs row, rows obs_size apart) -> [B,6]
__global__ void dg_gather_scalars(const precision_t* obs, precision_t* dst, int B, int obs_size, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * DG_SCALARS) return;
    int s = idx % DG_SCALARS, b = idx / DG_SCALARS;
    dst[idx] = obs[b * obs_size + offset + s];
}
// padded im2col: input NCHW [B,IC,IH,IW] -> col [B*OH*OW, IC*K*K], pad with 0 (S=1, OH=IH, OW=IW).
// in_row_stride = elements between consecutive batch rows of input (= obs_size when the grid is the
// first DG_GRID_FLAT of a wider obs row; = IC*IH*IW when input is a contiguous conv activation).
__global__ void dg_im2col_pad(const precision_t* input, precision_t* col,
        int B, int IC, int IH, int IW, int K, int pad, int in_row_stride) {
    int OH = IH, OW = IW, col_w = IC * K * K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OH * OW * col_w;
    if (idx >= total) return;
    int c = idx % col_w, row = idx / col_w;
    int ow = row % OW, oh = (row / OW) % OH, b = row / (OH * OW);
    int ic = c / (K * K), kk = c % (K * K), kh = kk / K, kw = kk % K;
    int ih = oh - pad + kh, iw = ow - pad + kw;
    float v = (ih < 0 || ih >= IH || iw < 0 || iw >= IW) ? 0.0f
        : to_float(input[b * in_row_stride + (ic * IH + ih) * IW + iw]);
    col[idx] = from_float(v);
}
// padded col2im: col_grad [B*OH*OW, IC*K*K] -> grad_input NCHW [B,IC,IH,IW] (gather, no atomics)
__global__ void dg_col2im_pad(const precision_t* col, precision_t* grad_input,
        int B, int IC, int IH, int IW, int K, int pad) {
    int OH = IH, OW = IW, col_w = IC * K * K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * IC * IH * IW;
    if (idx >= total) return;
    int iw = idx % IW, ih = (idx / IW) % IH, ic = (idx / (IW * IH)) % IC, b = idx / (IW * IH * IC);
    float sum = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        int oh = ih + pad - kh;
        if (oh < 0 || oh >= OH) continue;
        for (int kw = 0; kw < K; kw++) {
            int ow = iw + pad - kw;
            if (ow < 0 || ow >= OW) continue;
            sum += to_float(col[(b * OH * OW + oh * OW + ow) * col_w + ic * K * K + kh * K + kw]);
        }
    }
    grad_input[idx] = from_float(sum);
}
// (B*spatial, OC) row-major <-> (B, OC, spatial) NCHW
__global__ void dg_rows_to_nchw(const precision_t* src, precision_t* dst, int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * OC * spatial) return;
    int s = idx % spatial, oc = (idx / spatial) % OC, b = idx / (OC * spatial);
    dst[idx] = src[(b * spatial + s) * OC + oc];
}
__global__ void dg_nchw_to_rows(const precision_t* src, precision_t* dst, int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * OC * spatial) return;
    int s = idx % spatial, oc = (idx / spatial) % OC, b = idx / (OC * spatial);
    dst[(b * spatial + s) * OC + oc] = src[idx];
}
// concat [B,256]||[B,64] -> [B,320]; and split [B,320] -> halves
__global__ void dg_concat2(precision_t* out, const precision_t* a, const precision_t* b, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * DG_CONCAT) return;
    int c = idx % DG_CONCAT, row = idx / DG_CONCAT;
    out[idx] = (c < DG_FC) ? a[row * DG_FC + c] : b[row * DG_SFC + (c - DG_FC)];
}
__global__ void dg_split2(const precision_t* in, precision_t* a, precision_t* b, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * DG_CONCAT) return;
    int c = idx % DG_CONCAT, row = idx / DG_CONCAT;
    if (c < DG_FC) a[row * DG_FC + c] = in[idx];
    else b[row * DG_SFC + (c - DG_FC)] = in[idx];
}

// ---- weights / activations ----
struct DungeonEncWeights {
    PrecisionTensor c1w, c1b, c2w, c2b, gfw, gfb, sfw, sfb, fw, fb;
    int obs_size, hidden;
};
struct DungeonEncActs {
    PrecisionTensor saved;                 // [B, obs_size]
    PrecisionTensor z1, a1, z2, a2;        // conv pre/post-gelu, NCHW [B,32,31,31] (a2 viewed [B,30752])
    PrecisionTensor zg, ag, zs, as_, concat, zf, out;
    PrecisionTensor col, mm;               // shared im2col + matmul scratch (sized for conv2)
    PrecisionTensor gconv, gcat, gh1, gh2; // backward scratch ([B,30752], [B,320], halves)
    // weight grads
    PrecisionTensor c1wg, c1bg, c2wg, c2bg, gfwg, gfbg, sfwg, sfbg, fwg, fbg;
};

static PrecisionTensor dungeon_encoder_forward(void* w, void* act, PrecisionTensor input, cudaStream_t stream) {
    DungeonEncWeights* e = (DungeonEncWeights*)w;
    DungeonEncActs* a = (DungeonEncActs*)act;
    int B = input.shape[0], H = e->hidden, conv_n = B * DG_C1 * DG_SP;
    if (a->saved.data) puf_copy(&a->saved, &input, stream);

    // conv1: im2col(grid, stride=obs_size) @ c1w^T -> nchw -> +bias -> gelu
    dg_im2col_pad<<<grid_size(B * DG_SP * DG_COL1), BLOCK_SIZE, 0, stream>>>(input.data, a->col.data, B, DG_IC, DG_GRID, DG_GRID, DG_K, DG_PAD, e->obs_size);
    PrecisionTensor col1 = {.data = a->col.data, .shape = {B * DG_SP, DG_COL1}};
    PrecisionTensor mm1 = {.data = a->mm.data, .shape = {B * DG_SP, DG_C1}};
    puf_mm(&col1, &e->c1w, &mm1, stream);
    dg_rows_to_nchw<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->mm.data, a->z1.data, B, DG_C1, DG_SP);
    dg_bias_gelu_nchw<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->z1.data, a->a1.data, e->c1b.data, conv_n, DG_C1, DG_SP);

    // conv2 (input a1 is contiguous [B,30752])
    dg_im2col_pad<<<grid_size(B * DG_SP * DG_COL2), BLOCK_SIZE, 0, stream>>>(a->a1.data, a->col.data, B, DG_C1, DG_GRID, DG_GRID, DG_K, DG_PAD, DG_CONV_FLAT);
    PrecisionTensor col2 = {.data = a->col.data, .shape = {B * DG_SP, DG_COL2}};
    PrecisionTensor mm2 = {.data = a->mm.data, .shape = {B * DG_SP, DG_C2}};
    puf_mm(&col2, &e->c2w, &mm2, stream);
    dg_rows_to_nchw<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->mm.data, a->z2.data, B, DG_C2, DG_SP);
    dg_bias_gelu_nchw<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->z2.data, a->a2.data, e->c2b.data, conv_n, DG_C2, DG_SP);

    // grid_fc: a2[B,30752] @ gfw^T -> +bias -> gelu
    PrecisionTensor a2flat = {.data = a->a2.data, .shape = {B, DG_CONV_FLAT}};
    PrecisionTensor zg = {.data = a->zg.data, .shape = {B, DG_FC}};
    puf_mm(&a2flat, &e->gfw, &zg, stream);
    dg_bias_gelu<<<grid_size(B * DG_FC), BLOCK_SIZE, 0, stream>>>(a->zg.data, a->ag.data, e->gfb.data, B * DG_FC, DG_FC);

    // scalar_fc: gather strided scalars into contiguous mm scratch [B,6], then @ sfw^T -> +bias -> gelu
    PrecisionTensor scal_dst = {.data = a->mm.data, .shape = {B, DG_SCALARS}};
    dg_gather_scalars<<<grid_size(B * DG_SCALARS), BLOCK_SIZE, 0, stream>>>(input.data, a->mm.data, B, e->obs_size, DG_GRID_FLAT);
    PrecisionTensor zs = {.data = a->zs.data, .shape = {B, DG_SFC}};
    puf_mm(&scal_dst, &e->sfw, &zs, stream);
    dg_bias_gelu<<<grid_size(B * DG_SFC), BLOCK_SIZE, 0, stream>>>(a->zs.data, a->as_.data, e->sfb.data, B * DG_SFC, DG_SFC);

    // concat + fuse
    dg_concat2<<<grid_size(B * DG_CONCAT), BLOCK_SIZE, 0, stream>>>(a->concat.data, a->ag.data, a->as_.data, B);
    PrecisionTensor concat = {.data = a->concat.data, .shape = {B, DG_CONCAT}};
    PrecisionTensor zf = {.data = a->zf.data, .shape = {B, H}};
    puf_mm(&concat, &e->fw, &zf, stream);
    dg_bias_gelu<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->zf.data, a->out.data, e->fb.data, B * H, H);
    return a->out;
}

static void dungeon_encoder_backward(void* w, void* act, PrecisionTensor grad, cudaStream_t stream) {
    DungeonEncWeights* e = (DungeonEncWeights*)w;
    DungeonEncActs* a = (DungeonEncActs*)act;
    int B = grad.shape[0], H = e->hidden, conv_n = B * DG_C1 * DG_SP;

    // fuse backward: grad(=dOut) -> dZf -> bias/wgrad, dConcat
    dg_gelu_bwd<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(grad.data, a->zf.data, B * H);
    dg_bias_grad_lin<<<H, 256, 0, stream>>>(a->fbg.data, grad.data, B, H);
    PrecisionTensor concat = {.data = a->concat.data, .shape = {B, DG_CONCAT}};
    puf_mm_tn(&grad, &concat, &a->fwg, stream);
    PrecisionTensor gcat = {.data = a->gcat.data, .shape = {B, DG_CONCAT}};
    puf_mm_nn(&grad, &e->fw, &gcat, stream);
    dg_split2<<<grid_size(B * DG_CONCAT), BLOCK_SIZE, 0, stream>>>(a->gcat.data, a->gh1.data, a->gh2.data, B);  // gh1=dAg[B,256], gh2=dAs[B,64]

    // scalar_fc backward (no input grad)
    dg_gelu_bwd<<<grid_size(B * DG_SFC), BLOCK_SIZE, 0, stream>>>(a->gh2.data, a->zs.data, B * DG_SFC);
    dg_bias_grad_lin<<<DG_SFC, 256, 0, stream>>>(a->sfbg.data, a->gh2.data, B, DG_SFC);
    PrecisionTensor gh2 = {.data = a->gh2.data, .shape = {B, DG_SFC}};
    PrecisionTensor scal_c = {.data = a->mm.data, .shape = {B, DG_SCALARS}};  // re-gather scalars into mm scratch
    dg_gather_scalars<<<grid_size(B * DG_SCALARS), BLOCK_SIZE, 0, stream>>>(a->saved.data, a->mm.data, B, e->obs_size, DG_GRID_FLAT);
    puf_mm_tn(&gh2, &scal_c, &a->sfwg, stream);

    // grid_fc backward: dAg -> dZg -> bias/wgrad, dA2flat
    dg_gelu_bwd<<<grid_size(B * DG_FC), BLOCK_SIZE, 0, stream>>>(a->gh1.data, a->zg.data, B * DG_FC);
    dg_bias_grad_lin<<<DG_FC, 256, 0, stream>>>(a->gfbg.data, a->gh1.data, B, DG_FC);
    PrecisionTensor gh1 = {.data = a->gh1.data, .shape = {B, DG_FC}};
    PrecisionTensor a2flat = {.data = a->a2.data, .shape = {B, DG_CONV_FLAT}};
    puf_mm_tn(&gh1, &a2flat, &a->gfwg, stream);
    PrecisionTensor gconv = {.data = a->gconv.data, .shape = {B, DG_CONV_FLAT}};  // dA2 (flattened NCHW)
    puf_mm_nn(&gh1, &e->gfw, &gconv, stream);

    // conv2 backward: dA2 -> dZ2 -> bias/wgrad + dA1 (input grad needed)
    dg_gelu_bwd<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->gconv.data, a->z2.data, conv_n);  // gconv now = dZ2 (NCHW)
    dg_conv_bias_grad<<<DG_C2, 256, 0, stream>>>(a->c2bg.data, a->gconv.data, B, DG_C2, DG_SP);
    // wgrad: dZ2(NCHW)->rows (mm), im2col(a1)->col, wgrad = mm^T @ col
    dg_nchw_to_rows<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->gconv.data, a->mm.data, B, DG_C2, DG_SP);
    dg_im2col_pad<<<grid_size(B * DG_SP * DG_COL2), BLOCK_SIZE, 0, stream>>>(a->a1.data, a->col.data, B, DG_C1, DG_GRID, DG_GRID, DG_K, DG_PAD, DG_CONV_FLAT);
    PrecisionTensor mm2 = {.data = a->mm.data, .shape = {B * DG_SP, DG_C2}};
    PrecisionTensor col2 = {.data = a->col.data, .shape = {B * DG_SP, DG_COL2}};
    puf_mm_tn(&mm2, &col2, &a->c2wg, stream);
    // input grad dA1: col_grad = mm @ c2w -> col2im -> gh-conv buffer (reuse gconv? need fresh); use a->z2 as scratch (consumed)
    puf_mm_nn(&mm2, &e->c2w, &col2, stream);     // col2 now = col_grad [B*SP, COL2]
    dg_col2im_pad<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->col.data, a->gconv.data, B, DG_C1, DG_GRID, DG_GRID, DG_K, DG_PAD);  // gconv = dA1 (NCHW)

    // conv1 backward: dA1 -> dZ1 -> bias/wgrad (no input grad)
    dg_gelu_bwd<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->gconv.data, a->z1.data, conv_n);
    dg_conv_bias_grad<<<DG_C1, 256, 0, stream>>>(a->c1bg.data, a->gconv.data, B, DG_C1, DG_SP);
    dg_nchw_to_rows<<<grid_size(conv_n), BLOCK_SIZE, 0, stream>>>(a->gconv.data, a->mm.data, B, DG_C1, DG_SP);
    dg_im2col_pad<<<grid_size(B * DG_SP * DG_COL1), BLOCK_SIZE, 0, stream>>>(a->saved.data, a->col.data, B, DG_IC, DG_GRID, DG_GRID, DG_K, DG_PAD, e->obs_size);
    PrecisionTensor mm1 = {.data = a->mm.data, .shape = {B * DG_SP, DG_C1}};
    PrecisionTensor col1 = {.data = a->col.data, .shape = {B * DG_SP, DG_COL1}};
    puf_mm_tn(&mm1, &col1, &a->c1wg, stream);
}

static void dungeon_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    DungeonEncWeights* e = (DungeonEncWeights*)w;
    auto k = [&](PrecisionTensor& t, int r, int c, float g) { PrecisionTensor wt = {.data = t.data, .shape = {r, c}}; puf_kaiming_init(&wt, g, (*seed)++, stream); };
    float g = std::sqrt(2.0f);
    k(e->c1w, DG_C1, DG_COL1, g);  k(e->c2w, DG_C2, DG_COL2, g);
    k(e->gfw, DG_FC, DG_CONV_FLAT, g);  k(e->sfw, DG_SFC, DG_SCALARS, g);  k(e->fw, e->hidden, DG_CONCAT, g);
    auto z = [&](PrecisionTensor& t) { cudaMemsetAsync(t.data, 0, numel(t.shape) * sizeof(precision_t), stream); };
    z(e->c1b); z(e->c2b); z(e->gfb); z(e->sfb); z(e->fb);
}

static void dungeon_encoder_reg_params(void* w, Allocator* alloc) {
    DungeonEncWeights* e = (DungeonEncWeights*)w;
    e->c1w = {.shape = {DG_C1, DG_COL1}}; e->c1b = {.shape = {DG_C1}};
    e->c2w = {.shape = {DG_C2, DG_COL2}}; e->c2b = {.shape = {DG_C2}};
    e->gfw = {.shape = {DG_FC, DG_CONV_FLAT}}; e->gfb = {.shape = {DG_FC}};
    e->sfw = {.shape = {DG_SFC, DG_SCALARS}}; e->sfb = {.shape = {DG_SFC}};
    e->fw = {.shape = {e->hidden, DG_CONCAT}}; e->fb = {.shape = {e->hidden}};
    PrecisionTensor* ts[] = {&e->c1w, &e->c1b, &e->c2w, &e->c2b, &e->gfw, &e->gfb, &e->sfw, &e->sfb, &e->fw, &e->fb};
    for (auto* t : ts) alloc_register(alloc, t);
}

static void dungeon_encoder_reg_train(void* w, void* act, Allocator* acts, Allocator* grads, int B) {
    DungeonEncWeights* e = (DungeonEncWeights*)w;
    DungeonEncActs* a = (DungeonEncActs*)act;
    *a = {};
    a->saved  = {.shape = {B, e->obs_size}};
    a->z1 = {.shape = {B * DG_CONV_FLAT}}; a->a1 = {.shape = {B * DG_CONV_FLAT}};
    a->z2 = {.shape = {B * DG_CONV_FLAT}}; a->a2 = {.shape = {B * DG_CONV_FLAT}};
    a->zg = {.shape = {B, DG_FC}}; a->ag = {.shape = {B, DG_FC}};
    a->zs = {.shape = {B, DG_SFC}}; a->as_ = {.shape = {B, DG_SFC}};
    a->concat = {.shape = {B, DG_CONCAT}}; a->zf = {.shape = {B, e->hidden}}; a->out = {.shape = {B, e->hidden}};
    a->col = {.shape = {B * DG_SP, DG_COL2}}; a->mm = {.shape = {B * DG_SP, DG_C2}};
    a->gconv = {.shape = {B * DG_CONV_FLAT}}; a->gcat = {.shape = {B, DG_CONCAT}};
    a->gh1 = {.shape = {B, DG_FC}}; a->gh2 = {.shape = {B, DG_SFC}};
    PrecisionTensor* av[] = {&a->saved, &a->z1, &a->a1, &a->z2, &a->a2, &a->zg, &a->ag, &a->zs, &a->as_,
        &a->concat, &a->zf, &a->out, &a->col, &a->mm, &a->gconv, &a->gcat, &a->gh1, &a->gh2};
    for (auto* t : av) alloc_register(acts, t);
    a->c1wg = {.shape = {DG_C1, DG_COL1}}; a->c1bg = {.shape = {DG_C1}};
    a->c2wg = {.shape = {DG_C2, DG_COL2}}; a->c2bg = {.shape = {DG_C2}};
    a->gfwg = {.shape = {DG_FC, DG_CONV_FLAT}}; a->gfbg = {.shape = {DG_FC}};
    a->sfwg = {.shape = {DG_SFC, DG_SCALARS}}; a->sfbg = {.shape = {DG_SFC}};
    a->fwg = {.shape = {e->hidden, DG_CONCAT}}; a->fbg = {.shape = {e->hidden}};
    PrecisionTensor* gv[] = {&a->c1wg, &a->c1bg, &a->c2wg, &a->c2bg, &a->gfwg, &a->gfbg, &a->sfwg, &a->sfbg, &a->fwg, &a->fbg};
    for (auto* t : gv) alloc_register(grads, t);
}

static void dungeon_encoder_reg_rollout(void* w, void* act, Allocator* alloc, int B) {
    DungeonEncWeights* e = (DungeonEncWeights*)w;
    DungeonEncActs* a = (DungeonEncActs*)act;
    *a = {};
    a->z1 = {.shape = {B * DG_CONV_FLAT}}; a->a1 = {.shape = {B * DG_CONV_FLAT}};
    a->z2 = {.shape = {B * DG_CONV_FLAT}}; a->a2 = {.shape = {B * DG_CONV_FLAT}};
    a->zg = {.shape = {B, DG_FC}}; a->ag = {.shape = {B, DG_FC}};
    a->zs = {.shape = {B, DG_SFC}}; a->as_ = {.shape = {B, DG_SFC}};
    a->concat = {.shape = {B, DG_CONCAT}}; a->zf = {.shape = {B, e->hidden}}; a->out = {.shape = {B, e->hidden}};
    a->col = {.shape = {B * DG_SP, DG_COL2}}; a->mm = {.shape = {B * DG_SP, DG_C2}};
    PrecisionTensor* av[] = {&a->z1, &a->a1, &a->z2, &a->a2, &a->zg, &a->ag, &a->zs, &a->as_,
        &a->concat, &a->zf, &a->out, &a->col, &a->mm};
    for (auto* t : av) alloc_register(alloc, t);
}

static void* dungeon_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    DungeonEncWeights* ew = (DungeonEncWeights*)calloc(1, sizeof(DungeonEncWeights));
    ew->obs_size = e->in_dim; ew->hidden = e->out_dim;
    return ew;
}
static void dungeon_encoder_free_weights(void* weights) { free(weights); }
static void dungeon_encoder_free_activations(void* act) { free(act); }

static void create_dungeon_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = dungeon_encoder_forward,
        .backward = dungeon_encoder_backward,
        .init_weights = dungeon_encoder_init_weights,
        .reg_params = dungeon_encoder_reg_params,
        .reg_train = dungeon_encoder_reg_train,
        .reg_rollout = dungeon_encoder_reg_rollout,
        .create_weights = dungeon_encoder_create_weights,
        .free_weights = dungeon_encoder_free_weights,
        .free_activations = dungeon_encoder_free_activations,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(DungeonEncActs),
    };
}
#endif
