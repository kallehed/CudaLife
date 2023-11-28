#include "include/glad/glad.h"
#include <GL/glext.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

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
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
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

  // draw_world_in_terminal(h_world, WIDTH, HEIGHT);

  // const long window_width = 2048, window_height = 2048;
  const long window_width = 1024, window_height = 1024;
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow *window =
      glfwCreateWindow(window_width, window_height, "CudaLife", NULL, NULL);
  glfwMakeContextCurrent(window);
  // gladLoadGL(glfwGetProcAddress());
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
  glViewport(0, 0, 800, 800);
  glfwSwapInterval(1);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  float vertices[] = {
      -1.f, -1.f, 0.f, 0.f, 1.f, 0.f, 1.f, -1.f, 0.f,
  };
  unsigned int VBO;
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  const char *const vertex_shader_source =
      "#version 460\n"
      "layout (location = 0) in vec3 aPos;\n"
      "void main()\n"
      "{\n"
      " gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
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
  const char *const fragment_shader_source =
    "#version 460 core\n"
    "out vec4 FragColor;\n"
    "void main() {\n"
    " FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n";
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
    if(!success) {
        glGetProgramInfoLog(shader_program, 512, NULL, infoLog);
        printf("ERROR::PROGRAM::COMPILATION_FAILED\n %s\n", infoLog);
    }
  }
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);  
  // SetTargetFPS(60);
  double dt = 0.16, prev_time = 0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    {
      double time = glfwGetTime();
      dt = time - prev_time;
      prev_time = time;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
      randomize_world(h_world, d_world, WIDTH, HEIGHT);
    }
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    // transform_world(d_world, WIDTH, HEIGHT);
    // copy from device to host
    cudaMemcpy(h_world, d_world, WORLD_BYTES, cudaMemcpyDeviceToHost);

    {
      char buf[256];
      int written = snprintf(buf, sizeof(buf) - 1, "fps: %f", 1.f / dt);
      buf[written] = '\0';
      printf("%s", buf);
    }
    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}
