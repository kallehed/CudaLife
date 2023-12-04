
#include "project_header.cuh"

__global__ void transform_cell(const unsigned char *const world,
                               unsigned char *write_world) {
  const long x = blockIdx.x * blockDim.x + threadIdx.x;
  const long y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1)
    return;
  const long place = x + y * WIDTH;
  const unsigned char cur_state = world[place];
  const unsigned char neighbors =
      world[place + 1] + world[place - 1] + world[place + WIDTH] +
      world[place - WIDTH] + world[place + 1 - WIDTH] +
      world[place + 1 + WIDTH] + world[place - 1 + WIDTH] +
      world[place - 1 - WIDTH];
  unsigned char next_state;
  // switch (neighbors) {
  // case 2:
  //   next_state = cur_state;
  //   break;
  // case 3:
  //   next_state = CELL_ALIVE;
  //   break;
  // default:
  //   next_state = CELL_DEAD;
  // }
  switch (neighbors) {
  case 0:
    next_state = CELL_ALIVE;
  case 1:
    next_state = CELL_ALIVE;
  case 2:
    next_state = CELL_ALIVE;
    break;
  case 3:
    next_state = CELL_ALIVE;
    break;
  default:
    next_state = CELL_DEAD;
  }
  write_world[place] = next_state;
}

__global__ void array2D_set(unsigned char *a, const long width,
                            const unsigned char val) {
  a[(threadIdx.x + blockIdx.x * blockDim.x) +
    (threadIdx.y + blockIdx.y * blockDim.y) * width] = val;
}

void transform_world(const unsigned char *const read_world,
                            unsigned char *const write_world) {
  transform_cell<<<GRIDDIM_WORLD, BLOCKDIM_WORLD>>>(read_world, write_world);
  cudaDeviceSynchronize();
}


