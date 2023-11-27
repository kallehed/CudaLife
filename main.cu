#include <cstdio>
#include <cstdlib>
#include <ctime>

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

__global__ void array2D_set(char *a, const long width, const char val) {
  a[(threadIdx.x + blockIdx.x * blockDim.x) +
    (threadIdx.y + blockIdx.y * blockDim.y) * width] = val;
}

#define CELL_DEAD 0
#define CELL_ALIVE 1

__global__ void transform_cell(char *const world, const long width,
                               const long height) {
  const long x = blockIdx.x * blockDim.x + threadIdx.x;
  const long y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
    return;
  const long place = x + y * width;
  const char cur_state = world[place];
  const char neighbors = world[place + 1] + world[place - 1] +
                         world[place + width] + world[place - width] +
                         world[place + 1 - width] + world[place + 1 + width] +
                         world[place - 1 + width] + world[place - 1 - width];
  char next_state;
  switch (neighbors) {
  case 0:
  case 1:
    next_state = CELL_DEAD;
    break;
  case 2:
    next_state = cur_state;
    break;
  case 3:
    next_state = CELL_ALIVE;
    break;
  default:
    next_state = CELL_DEAD;
  }
  world[place] = next_state;
}

void draw_world_in_terminal(const char *const world, const long width,
                            const long height) {
  for (long i = 0; i < height; ++i) {
    for (long j = 0; j < width; ++j) {
      char out;
      switch (world[j + i * width]) {
        case CELL_DEAD:
          out = ' ';
          break;
        case CELL_ALIVE:
          out = '+';
          break;
      }
      putchar(out);
    }
    putchar('\n');
  }
}

static constexpr long WIDTH = 32;
static constexpr long HEIGHT = 32;
static constexpr long WORLD_BYTES = sizeof(char) * WIDTH * HEIGHT;

static constexpr dim3 BLOCKDIM_WORLD = dim3{32, 32, 1};
static constexpr dim3 GRIDDIM_WORLD = dim3{1, 1, 1};

// game of life, use shared memory so a 32x32 part will load into shared memory
// their values, and the middle 30x30 part will calculate but start by using
// global memory and divide thread blocks into chunks that calculate new grid

int main() {
  srand(time(NULL));
  char *h_world = (char *)malloc(WORLD_BYTES);
  char *d_world;
  cudaMalloc(&d_world, WORLD_BYTES);
  // array2D_set<<<GRIDDIM_WORLD, BLOCKDIM_WORLD>>>(d_world, WIDTH, CELL_DEAD);
  // cudaDeviceSynchronize();

  // set host to random
  for (int i = 1; i < HEIGHT - 1; ++i) {
    for (int j = 1; j < WIDTH - 1; ++j) {
      h_world[j + i * WIDTH] = rand() % 2;
    }
  }
  // upload to device from host
  cudaMemcpy(d_world, h_world, WORLD_BYTES, cudaMemcpyHostToDevice);

  for (int iter = 0; iter < 10000; ++iter) {
    transform_cell<<<GRIDDIM_WORLD, BLOCKDIM_WORLD>>>(d_world, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // copy back to host
    cudaMemcpy(h_world, d_world, WORLD_BYTES, cudaMemcpyDeviceToHost);
    draw_world_in_terminal(h_world, WIDTH, HEIGHT);
    puts("--------------------------------");
  }

  float total_error = 0;
  for (long i = 0; i < HEIGHT; ++i) {
    for (long j = 0; j < WIDTH; ++j) {
      total_error += abs(h_world[j + i * WIDTH]);
    }
    // std::cout << " " << h_a[i] << " ";
  }
  printf("total life: %f\n" , (float)total_error);
}
