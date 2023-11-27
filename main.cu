#include "raylib.h"
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
  __syncthreads();
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
  puts("--------------------------------");
}


static Texture2D MY_TEX;

static void draw_world_raylib(const char *const world, const long width,
                              const long height, const long window_width,
                              const long window_height) {
  // Vector2 size = {window_width / (float)width, window_height / (float)height};
  if (MY_TEX.width == window_width) {
    UnloadTexture(MY_TEX);
  }
  Image img = GenImageColor(window_width, window_height, WHITE); 
  MY_TEX = LoadTextureFromImage(img);
  // GenTextureMipmaps(&tex);
  DrawText("ym biot wirug", 100, 100, 20, WHITE);
  // UnloadTexture(tex);
  UnloadImage(img);
  
  // for (long i = 0; i < height; ++i) {
  //   for (long j = 0; j < width; ++j) {
  //     if (world[j + i * width]) {
  //       Vector2 v = Vector2{float(j) * size.x, float(i) * size.y};
  //
  //       DrawPixelV(v, WHITE);
  //       // DrawRectangleV(v, size, WHITE);
  //     }
  //   }
  // }
  UpdateTexture(MY_TEX, world);
  DrawTexture(MY_TEX, 0, 0, WHITE);
}

static constexpr long WIDTH = 2048;
static constexpr long HEIGHT = 2048;
static constexpr long WORLD_BYTES = sizeof(char) * WIDTH * HEIGHT;
static constexpr dim3 BLOCKDIM_WORLD = dim3{32, 32, 1};
static constexpr dim3 GRIDDIM_WORLD = dim3{64, 64, 1};

static void transform_world(char *const d_world, const long width,
                            const long height) {
  transform_cell<<<GRIDDIM_WORLD, BLOCKDIM_WORLD>>>(d_world, width, height);
  cudaDeviceSynchronize();
}

static void randomize_world(char *const h_world, char *const d_world,
                            const long width, const long height) {
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      h_world[j + i * width] = rand() % 2;
    }
  }
  // upload to device from host
  cudaMemcpy(d_world, h_world, WORLD_BYTES, cudaMemcpyHostToDevice);
}

// game of life, use shared memory so a 32x32 part will load into shared memory
// their values, and the middle 30x30 part will calculate but start by using
// global memory and divide thread blocks into chunks that calculate new grid
// array2D_set<<<GRIDDIM_WORLD, BLOCKDIM_WORLD>>>(d_world, WIDTH, CELL_DEAD);
// cudaDeviceSynchronize();
int main() {
  srand(time(NULL));
  char *h_world = (char *)malloc(WORLD_BYTES);
  char *d_world;
  cudaMalloc(&d_world, WORLD_BYTES);

  randomize_world(h_world, d_world, WIDTH, HEIGHT);

  draw_world_in_terminal(h_world, WIDTH, HEIGHT);

  const long window_width = 1024, window_height = 1024;
  InitWindow(window_width, window_height, "cudalife");
  // SetTargetFPS(60);
  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    if (IsKeyPressed(KEY_R)) {
      randomize_world(h_world, d_world, WIDTH, HEIGHT);
    }
    transform_world(d_world, WIDTH, HEIGHT);
    BeginDrawing();
    ClearBackground(BLACK);
    // copy from device to host
    cudaMemcpy(h_world, d_world, WORLD_BYTES, cudaMemcpyDeviceToHost);
    draw_world_raylib(h_world, WIDTH, HEIGHT, window_width, window_height);

    {
      float dt = GetFrameTime();
      char buf[256];
      int written = snprintf(buf, sizeof(buf) - 1, "fps: %f", 1.f/dt); 
      buf[written] = '\0';
      DrawText(buf, 10, 10, 20, GREEN);
    }
    EndDrawing();
  }
  CloseWindow(); // Close window and OpenGL context
}
