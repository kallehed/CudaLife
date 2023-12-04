#include "include/glad/glad.h"
#include <GL/glext.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "project_header.cuh"

void draw_world_in_terminal(const unsigned char *const world) {
  for (long i = 0; i < HEIGHT; ++i) {
    for (long j = 0; j < WIDTH; ++j) {
      unsigned char out;
      switch (world[j + i * WIDTH]) {
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

static bool g_space_just_pressed;
static void key_callback(GLFWwindow *window, int key, int scancode, int action,
                         int mods) {
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
    g_space_just_pressed = true;
  }
}
static int g_current_window_width = 1024, g_current_window_height = 1024;
static void framebuffer_size_callback(GLFWwindow *window, int width,
                                      int height) {
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
  glfwSetKeyCallback(window, key_callback);

  unsigned int shader_program = get_program();

  unsigned int VAO;
  glGenVertexArrays(1, &VAO);

  unsigned int SSBO_first;
  glGenBuffers(1, &SSBO_first);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO_first);
  glBufferData(GL_SHADER_STORAGE_BUFFER, WORLD_BYTES, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, SSBO_first);
  randomize_world(SSBO_first);
  unsigned int SSBO_second;
  glGenBuffers(1, &SSBO_second);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO_second);
  glBufferData(GL_SHADER_STORAGE_BUFFER, WORLD_BYTES, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, SSBO_second);

  struct cudaGraphicsResource *SSBO_CUDA_first;
  Chk(cudaGraphicsGLRegisterBuffer(&SSBO_CUDA_first, SSBO_first,
                                   cudaGraphicsMapFlagsNone));
  struct cudaGraphicsResource *SSBO_CUDA_second;
  Chk(cudaGraphicsGLRegisterBuffer(&SSBO_CUDA_second, SSBO_second,
                                   cudaGraphicsMapFlagsNone));

  int u_pos_and_scale_location =
      glGetUniformLocation(shader_program, "u_pos_and_scale");
  float pos_x = 0.f, pos_y = 0.f, scale = 1.f;
  bool should_transform = true;

  double dt = 0.16, prev_time = 0.0;
  while (!glfwWindowShouldClose(window)) {
    g_space_just_pressed = false;
    glfwPollEvents();
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
      randomize_world(SSBO_first);
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
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
      terminate_all_life_in_world(SSBO_first);
    }
    if (g_space_just_pressed) {
      should_transform = !should_transform;
    }

    // mouse input
    {
      bool left_click =
          glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
      bool right_click =
          glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
      if (left_click || right_click) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        y = HEIGHT / 2 - y;
        // printf("pos: %f, %f, real: %f, %f\n", x, y, pos_x, pos_y);

        double real_x = (pos_x + x) / scale;
        double real_y = (pos_y + y) / scale;
        // printf("actual: %f, %f\n", real_x, real_y);
        {
          int cell_x = real_x, cell_y = real_y;
          world_set_cell(SSBO_first, cell_x, cell_y, left_click ? 1 : 0);
        }
      }
    }

    // Current bottleneck
    if (should_transform) { // transform world
      Chk(cudaGraphicsMapResources(1, &SSBO_CUDA_first, 0));
      void *ssbo_first_mapped_to_cuda;
      Chk(cudaGraphicsResourceGetMappedPointer(&ssbo_first_mapped_to_cuda, NULL,
                                               SSBO_CUDA_first));

      Chk(cudaGraphicsMapResources(1, &SSBO_CUDA_second, 0));
      void *ssbo_second_mapped_to_cuda;
      Chk(cudaGraphicsResourceGetMappedPointer(&ssbo_second_mapped_to_cuda,
                                               NULL, SSBO_CUDA_second));

      transform_world(
          (unsigned char *)ssbo_first_mapped_to_cuda, // IMPORTANT LINE
          (unsigned char *)ssbo_second_mapped_to_cuda);
      cudaMemcpy(ssbo_first_mapped_to_cuda, ssbo_second_mapped_to_cuda,
                 WORLD_BYTES, cudaMemcpyDeviceToDevice);

      cudaGraphicsUnmapResources(1, &SSBO_CUDA_first);
      cudaGraphicsUnmapResources(1, &SSBO_CUDA_second);
    }

    { // set delta time
      double time = glfwGetTime();
      dt = time - prev_time;
      prev_time = time;
    }
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    { // draw to screen
      glUseProgram(shader_program);
      glUniform4f(u_pos_and_scale_location, pos_x, pos_y, scale, scale);
      glBindVertexArray(VAO);
      glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    { // set window title to framerate
      char buf[256];
      int written =
          snprintf(buf, sizeof(buf) - 1, "CudaLife: fps: %f", 1.f / dt);
      buf[written] = '\0';
      glfwSetWindowTitle(window, buf);
    }

    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}
