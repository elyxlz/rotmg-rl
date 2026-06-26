// Standalone parity + gradient check for the native dungeon CNN encoder (puffer4/dungeon_encoder.cu)
// vs the torch DungeonEncoder. Loads identical weights + input from files (written by
// scripts/check_encoder_parity.py in reg_params order), runs forward (+ optional backward), and
// dumps the encoder output (and weight grads) for the Python side to compare.
//
//   nvcc -DPRECISION_FLOAT -O2 -I src test_encoder.cu -lcudart -lcublas -o test_encoder
//   ./test_encoder B H weights.bin input.bin out.bin [dout.bin grads.bin]
#include <cstdio>
#include <cstdlib>
#include "models.cu"
#include "dungeon_encoder.cu"

static long fsize(const char* f) { FILE* fp = fopen(f, "rb"); fseek(fp, 0, SEEK_END); long n = ftell(fp); fclose(fp); return n; }
static void rd(const char* f, void* p, size_t n) { FILE* fp = fopen(f, "rb"); size_t r = fread(p, 1, n, fp); (void)r; fclose(fp); }
static void wr(const char* f, const void* p, size_t n) { FILE* fp = fopen(f, "wb"); fwrite(p, 1, n, fp); fclose(fp); }

int main(int argc, char** argv) {
    int B = atoi(argv[1]), H = atoi(argv[2]);
    const char *wf = argv[3], *inf = argv[4], *outf = argv[5];
    int obs = DG_GRID_FLAT + DG_SCALARS;  // 6733
    cudaStream_t s = 0;

    Encoder enc{}; enc.in_dim = obs; enc.out_dim = H;
    create_dungeon_encoder(&enc);
    DungeonEncWeights* w = (DungeonEncWeights*)enc.create_weights(&enc);

    Allocator pa{}; enc.reg_params(w, &pa); alloc_create(&pa);
    long wbytes = fsize(wf);
    float* hw = (float*)malloc(wbytes);
    rd(wf, hw, wbytes);
    cudaMemcpy(w->c1w.data, hw, wbytes, cudaMemcpyHostToDevice);  // params are contiguous from c1w.data

    DungeonEncActs acts{};
    Allocator aa{}, ga{}; enc.reg_train(w, &acts, &aa, &ga, B); alloc_create(&aa); alloc_create(&ga);

    float* d_in; cudaMalloc(&d_in, (size_t)B * obs * sizeof(float));
    float* h_in = (float*)malloc((size_t)B * obs * sizeof(float));
    rd(inf, h_in, (size_t)B * obs * sizeof(float));
    cudaMemcpy(d_in, h_in, (size_t)B * obs * sizeof(float), cudaMemcpyHostToDevice);
    PrecisionTensor input = {.data = d_in, .shape = {B, obs}};

    PrecisionTensor out = enc.forward(w, &acts, input, s);
    cudaDeviceSynchronize();
    float* h_out = (float*)malloc((size_t)B * H * sizeof(float));
    cudaMemcpy(h_out, out.data, (size_t)B * H * sizeof(float), cudaMemcpyDeviceToHost);
    wr(outf, h_out, (size_t)B * H * sizeof(float));

    if (argc >= 8) {  // backward: load dOut, run, dump grads (contiguous from c1wg.data, same layout as params)
        float* d_dout; cudaMalloc(&d_dout, (size_t)B * H * sizeof(float));
        float* h_dout = (float*)malloc((size_t)B * H * sizeof(float));
        rd(argv[6], h_dout, (size_t)B * H * sizeof(float));
        cudaMemcpy(d_dout, h_dout, (size_t)B * H * sizeof(float), cudaMemcpyHostToDevice);
        PrecisionTensor grad = {.data = d_dout, .shape = {B, H}};
        enc.backward(w, &acts, grad, s);
        cudaDeviceSynchronize();
        float* hg = (float*)malloc(wbytes);  // grads have the same total size + layout as params
        cudaMemcpy(hg, acts.c1wg.data, wbytes, cudaMemcpyDeviceToHost);
        wr(argv[7], hg, wbytes);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    printf("OK B=%d H=%d wbytes=%ld\n", B, H, wbytes);
    return 0;
}
