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
  char *h_world = (char *)malloc(WORLD_BYTES);
  char *d_world;
  cudaMalloc(&d_world, WORLD_BYTES);

  randomize_world(h_world, d_world, WIDTH, HEIGHT);

  // draw_world_in_terminal(h_world, WIDTH, HEIGHT);

  // const long window_width = 2048, window_height = 2048;
  glfwInit();
  glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);
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
      "  uint color[(2048 * 2048)/4];\n"
      "};\n"
      "void main() {\n"
      "  uint x = uint((gl_FragCoord.x + u_pos_and_scale.x)/u_pos_and_scale.z);\n"
      "  uint y = uint((gl_FragCoord.y + u_pos_and_scale.y)/u_pos_and_scale.w);\n"
      "  uint idx = x + y * 2048;\n"
      "  uint block = idx / 4;\n"
      "  uint byte = idx % 4;\n"
      "  uint col4 = color[block];\n"
      "  uint mask = (0x000000FF << (byte * 8));\n"
      "  uint colbool =  mask & col4;\n"
      "  float col = float(colbool);\n"
      "  if (x >= 2048 || y >= 2048) {col = 0.5;}"
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

  int u_pos_and_scale_location =
      glGetUniformLocation(shader_program, "u_pos_and_scale");
  float pos_x = 0.f, pos_y = 0.f, scale = 1.f;

  double dt = 0.16, prev_time = 0.0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
      randomize_world(h_world, d_world, WIDTH, HEIGHT);
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

    transform_world(d_world, WIDTH, HEIGHT);
    // copy from device to host
    cudaMemcpy(h_world, d_world, WORLD_BYTES, cudaMemcpyDeviceToHost);

    {
      double time = glfwGetTime();
      dt = time - prev_time;
      prev_time = time;
    }
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    {
      glUseProgram(shader_program);
      glUniform4f(u_pos_and_scale_location, pos_x, pos_y, scale, scale);
      glBindVertexArray(VAO);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, WORLD_BYTES, h_world);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
      glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    {
      char buf[256];
      int written =
          snprintf(buf, sizeof(buf) - 1, "CudaLife: fps: %f", 1.f / dt);
      buf[written] = '\0';
      glfwSetWindowTitle(window, buf);
      // printf("%s", buf);
    }

    // glfwSwapBuffers(window);
    glFlush();
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}
