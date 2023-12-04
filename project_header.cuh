
#define STR_INDIR(x) #x
#define STR(x) STR_INDIR(x)

#define CELL_DEAD 0
#define CELL_ALIVE 1

// width and height of game of life cell 2D array, has to be divisible by 32
#define WIDTH 2048
#define HEIGHT 2048
static constexpr long WORLD_BYTES = sizeof(unsigned char) * WIDTH * HEIGHT;
static constexpr dim3 BLOCKDIM_WORLD =
    dim3{32, 32, 1}; // 32 * 32 is the maximum block, don't increase!
static constexpr dim3 GRIDDIM_WORLD = dim3{WIDTH / 32, HEIGHT / 32, 1};

// cuda_routines.cu
__global__ void transform_cell(const unsigned char *const world,
                               unsigned char *write_world);

void transform_world(const unsigned char *const read_world,
                     unsigned char *const write_world);

// opengl_related.cu
unsigned int get_program();
void randomize_world(unsigned int SSBO);
void terminate_all_life_in_world(unsigned int SSBO);
void world_set_cell(unsigned int SSBO, int x, int y, unsigned char data);

#include <cstdio>
#include <cstdlib>

#define Chk(ans)                                                               \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
