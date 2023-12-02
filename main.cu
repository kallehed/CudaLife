#include "include/glad/glad.h"
#include <GL/glext.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#define STR_INDIR(x) #x
#define STR(x) STR_INDIR(x)

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
// look at 32x32 squares, everyone saves to shared memory, but only middle 30x30
// part writes. We also spawn overlapping 32x32 grid next to this one, which
// will handle the edge cases.
__global__ void transform_cell_better(char *const world, const long width,
                                      const long height) {
  __shared__ char shared_32x32[32][32];
  // positions on actual life grid
  const long actual_x = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
  const long actual_y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;
  const char cur_state = world[actual_y * width + actual_x]; // important write
  shared_32x32[threadIdx.y][threadIdx.x] = cur_state;

  __syncthreads(); // sync block so everyone has written to shared buffer

  if (threadIdx.x == 0 || threadIdx.x == 31 || threadIdx.y == 0 ||
      threadIdx.y == 31)
    return; // edges of thread block should disappear

  // relative x and y for brevity in neighbor calculatio
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const char neighbors =
      shared_32x32[y][x + 1] + shared_32x32[y][x - 1] + shared_32x32[y + 1][x] +
      shared_32x32[y - 1][x] + shared_32x32[y + 1][x + 1] +
      shared_32x32[y + 1][x - 1] + shared_32x32[y - 1][x + 1] +
      shared_32x32[y - 1][x - 1];

  char next_state;
  switch (neighbors) {
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
  world[actual_y * width + actual_x] = next_state; // write back to device mem
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

// width and height of game of life cell 2D array
#define WIDTH 2048
#define HEIGHT 2048
static constexpr long WORLD_BYTES = sizeof(char) * WIDTH * HEIGHT;
static constexpr dim3 BLOCKDIM_WORLD =
    dim3{32, 32, 1}; // 32 * 32 is the maximum block
static constexpr dim3 GRIDDIM_WORLD = dim3{WIDTH / 32, HEIGHT / 32, 1};

static void transform_world(char *const d_world, const long width,
                            const long height) {
  transform_cell_better<<<GRIDDIM_WORLD, BLOCKDIM_WORLD>>>(d_world, width,
                                                           height);
  cudaDeviceSynchronize();
}

// slow, copies using OpenGL, inits on CPU
static void randomize_world(unsigned int SSBO) {
  char *data = (char *)malloc(WORLD_BYTES);
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      char value;
      if (i == 0 || i == HEIGHT - 1 || j == 0 || j == WIDTH - 1) {
        value = 0;
      } else {
        value = rand() % 2;
      }
      data[i * WIDTH + j] = value;
    }
  }
  glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, WORLD_BYTES, data);
  free(data);
}
int g_current_window_width = 1024, g_current_window_height = 1024;
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
  g_current_window_width = width;
  g_current_window_height = height;
}

// game of life, use shared memory so a 32x32 part will load into shared memory
// their values, and the middle 30x30 part will calculate but start by using
// global memory and divide thread blocks into chunks that calculate new grid
// array2D_set<<<GRIDDIM_WORLD, BLOCKDIM_WORLD>>>(d_world, WIDTH, CELL_DEAD);
// cudaDeviceSynchronize();

#define SWAP_INTERVAL 0

int main() {
  srand(time(NULL));

  glfwInit();
  // glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow *window = glfwCreateWindow(
      g_current_window_width, g_current_window_height, "CudaLife", NULL, NULL);
  glfwMakeContextCurrent(window);
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
  glViewport(0, 0, g_current_window_width, g_current_window_height);
  glfwSwapInterval(SWAP_INTERVAL);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  const char *const vertex_shader_source =
      "#version 460\n"
      "float x_pos[6] = float[6](-1.f, -1.f, 1.f, -1.f, 1.f, 1.f); \n"
      "float y_pos[6] = float[6](-1.f, 1.f, 1.f, -1.f, 1.f, -1.f); \n"
      "void main()\n"
      "{\n"
      " float x = x_pos[gl_VertexID];"
      " float y = y_pos[gl_VertexID];"
      " gl_Position = vec4(x, y, 0.f, 1.0);\n"
      "}\0";
  unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
  glCompileShader(vertex_shader);
  {
    int success;
    char infoLog[512];
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(vertex_shader, 512, NULL, infoLog);
      printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n %s\n", infoLog);
    }
  }
  // clang-format off
  const char *const fragment_shader_source =
      "#version 460 core\n"
      "out vec4 FragColor;\n"
      "uniform vec4 u_pos_and_scale;"
      "layout (std430, binding = 0) buffer Colors {\n"
      "  uint color[(" STR(WIDTH) "*" STR(HEIGHT) " )/4];\n" 
      "};\n"
      "void main() {\n"
      "  uint x = uint((gl_FragCoord.x + u_pos_and_scale.x)/u_pos_and_scale.z);\n"
      "  uint y = uint((gl_FragCoord.y + u_pos_and_scale.y)/u_pos_and_scale.w);\n"
      "  uint idx = x + y * " STR(WIDTH) ";\n"
      "  uint block = idx / 4;\n"
      "  uint byte = idx % 4;\n"
      "  uint col4 = color[block];\n"
      "  uint mask = (0x000000FF << (byte * 8));\n"
      "  uint colbool =  mask & col4;\n"
      "  float col = float(colbool);\n"
      "  if (x >= " STR(WIDTH) " || y >= " STR(HEIGHT) ") {col = 0.5;}"
      "  FragColor = vec4(vec3(float(col)), 1.0f);\n"
      "}\n";
  // clang-format on
  unsigned int fragment_shader;
  fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
  glCompileShader(fragment_shader);
  {
    int success;
    char infoLog[512];
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(fragment_shader, 512, NULL, infoLog);
      printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n %s\n", infoLog);
    }
  }
  unsigned int shader_program;
  shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glLinkProgram(shader_program);
  {
    int success;
    char infoLog[512];
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader_program, 512, NULL, infoLog);
      printf("ERROR::PROGRAM::COMPILATION_FAILED\n %s\n", infoLog);
    }
  }
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  unsigned int VAO;
  glGenVertexArrays(1, &VAO);

  unsigned int SSBO;
  glGenBuffers(1, &SSBO);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
  glBufferData(GL_SHADER_STORAGE_BUFFER, WORLD_BYTES, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, SSBO);
  randomize_world(SSBO);

  struct cudaGraphicsResource *SSBO_CUDA;
  Chk(cudaGraphicsGLRegisterBuffer(&SSBO_CUDA, SSBO, cudaGraphicsMapFlagsNone));

  int u_pos_and_scale_location =
      glGetUniformLocation(shader_program, "u_pos_and_scale");
  float pos_x = 0.f, pos_y = 0.f, scale = 1.f;

  double dt = 0.16, prev_time = 0.0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
      randomize_world(SSBO);
    }
    float speed = 500.f * dt;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
      pos_x += speed;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
      pos_x -= speed;
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
      pos_y += speed;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
      pos_y -= speed;
    }
    float scale_speed = 1 * dt * scale;
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
      scale += scale_speed;
      pos_x += (pos_x + (float)(g_current_window_width >> 1)) * scale_speed *
               (1.f / scale);
      pos_y += (pos_y + (float)(g_current_window_height >> 1)) * scale_speed *
               (1.f / scale);
    }
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
      scale -= scale_speed;
      pos_x -= (pos_x + (float)(g_current_window_width >> 1)) * scale_speed *
               (1.f / scale);
      pos_y -= (pos_y + (float)(g_current_window_height >> 1)) * scale_speed *
               (1.f / scale);
    }

    // Current bottleneck
    // if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
    if (!false) {
      Chk(cudaGraphicsMapResources(1, &SSBO_CUDA, 0));
      void *ssbo_mapped_to_cuda;
      Chk(cudaGraphicsResourceGetMappedPointer(&ssbo_mapped_to_cuda, NULL,
                                               SSBO_CUDA));
      transform_world((char *)ssbo_mapped_to_cuda, WIDTH, HEIGHT);
      cudaGraphicsUnmapResources(1, &SSBO_CUDA);
    }

    {
      double time = glfwGetTime();
      dt = time - prev_time;
      prev_time = time;
    }
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (!false) {
      glUseProgram(shader_program);
      glUniform4f(u_pos_and_scale_location, pos_x, pos_y, scale, scale);
      glBindVertexArray(VAO);
      glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    if (!false) {
      char buf[256];
      int written =
          snprintf(buf, sizeof(buf) - 1, "CudaLife: fps: %f", 1.f / dt);
      buf[written] = '\0';
      glfwSetWindowTitle(window, buf);
    }

    glfwSwapBuffers(window);
    // glFlush();
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}
